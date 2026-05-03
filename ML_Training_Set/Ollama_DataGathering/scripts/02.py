#!/usr/bin/env python3
"""
02_extract_snippets_and_rules.py  (v6 — post-5-pass-audit final)

Reads raw GitHub Actions job logs from the index produced by script 01.
For each log:
  1. Cleans noise (ANSI, timestamps, group markers; preserves ##[error] lines)
  2. Extracts multiple named failure windows
  3. Infers context from metadata + text (including failing_step boosting)
  4. Scores each window against a research-backed pattern library
  5. Assigns a fine-grained primary_label from the 84-node CI failure ontology
  6. Maps to one of 5 training labels:
       compilation | test_failure | flaky_test | infrastructure | configuration
  7. Flags uncertain rows (needs_review=True) for Ollama (script 03)

Changes from v5 (post 5-pass audit):
  REMOVED: cache.miss (normal behaviour, not a failure cause)
  REMOVED: timeout.stale_or_archived (too many false positives)
  REMOVED: test.environment_external (wrong bucket mapping, redis/db refused
            is infrastructure not test_failure)
  REMOVED: build.script_or_packaging (redundant, covered by compile.* rules)
  REDUCED: cache.corrupt weight 3.5→2.5
  FIXED:   typo staleelementreferenceexception
  FIXED:   playwright regex tightened
  ADDED:   exit 143/125/100, SIGTERM, Gradle wrapper timeout, dial unix
           docker.sock, npm etarget/eunsupportedprotocol, Vitest,
           bundler connection resets, non-resolvable parent pom,
           broker URL patterns, step-name boosting in context inference
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH = Path("data/records/jobs_index.jsonl")
OUT_PATH   = Path("data/snippets/snippets_with_rules.jsonl")
STATS_PATH = Path("data/snippets/snippet_rule_stats.json")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EXTRACT_WORKERS   = 8
MAX_SNIPPET_CHARS = 18_000
MIN_TEXT_CHARS    = 50
WINDOW_BEFORE     = 280
WINDOW_AFTER      = 90
OLLAMA_THRESHOLD  = 0.82

VALID_LABELS = {
    "compilation", "test_failure", "flaky_test",
    "infrastructure", "configuration",
}

FAMILY_TO_BUCKET: dict[str, str | None] = {
    "build_compile":             "compilation",
    "test_deterministic":        "test_failure",
    "test_intermittent":         "flaky_test",
    "infrastructure":            "infrastructure",
    "timeout_cancel_state":      "infrastructure",
    "config_schema_env_secret":  "configuration",
    "dependency_resolution":     "configuration",
    "static_analysis":           "configuration",
    "auth_permission_policy":    "configuration",
    "source_checkout":           "configuration",
    "artifacts_cache_workspace": "configuration",
    "deploy_release":            "configuration",
    "runtime_execution":         "configuration",
    "insufficient_evidence":     None,
}

FAMILY_BY_PREFIX: dict[str, str] = {
    "compile.":   "build_compile",
    "static.":    "static_analysis",
    "test.":      "test_deterministic",
    "flaky.":     "test_intermittent",
    "infra.":     "infrastructure",
    "timeout.":   "timeout_cancel_state",
    "cancel.":    "timeout_cancel_state",
    "config.":    "config_schema_env_secret",
    "dep.":       "dependency_resolution",
    "auth.":      "auth_permission_policy",
    "policy.":    "auth_permission_policy",
    "scm.":       "source_checkout",
    "artifact.":  "artifacts_cache_workspace",
    "cache.":     "artifacts_cache_workspace",
    "workspace.": "artifacts_cache_workspace",
    "deploy.":    "deploy_release",
    "runtime.":   "runtime_execution",
    "unknown.":   "insufficient_evidence",
}


def label_family(primary: str) -> str:
    for prefix, family in FAMILY_BY_PREFIX.items():
        if primary.startswith(prefix):
            return family
    return "insufficient_evidence"


def to_bucket(primary: str) -> str | None:
    return FAMILY_TO_BUCKET.get(label_family(primary))


# ── Rule dataclass ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Rule:
    label:           str
    pattern:         str
    weight:          float = 1.0
    tags:            tuple[str, ...] = ()
    forbids_any:     tuple[str, ...] = ()
    confidence_hint: float | None = None
    note:            str = ""
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "regex", re.compile(self.pattern, re.I | re.M))


def R(label, pattern, weight=1.0, tags=(), forbids_any=(),
      confidence_hint=None, note="") -> Rule:
    return Rule(label, pattern, weight, tuple(tags),
                tuple(forbids_any), confidence_hint, note)


# ── Context detectors ─────────────────────────────────────────────────────────
_CTX = {
    "test": re.compile(
        r"\b(pytest|junit|testng|nunit|xunit|rspec|jest|mocha|karma|vitest|"
        r"nose|cypress|playwright|selenium|unittest|surefire|failsafe|"
        r"go test|cargo test|mvn test|gradle test|testrunner)\b", re.I),
    "build": re.compile(
        r"\b(compile|compiler|javac|gcc|g\+\+|clang|rustc|tsc|webpack|"
        r"kotlinc|go build|cargo build|make|cmake|ninja|maven-compiler|"
        r"msbuild|babel|rollup)\b", re.I),
    "dependency": re.compile(
        r"\b(npm|yarn|pnpm|pip|poetry|pipenv|conda|maven|gradle|nuget|gem|"
        r"bundler|cargo|go mod|composer|apt-get|apt|yum|apk|brew|registry|"
        r"artifact|dependency|dependencies|repository|pypi|npmjs|crates\.io)\b",
        re.I),
    "network": re.compile(
        r"\b(timeout|timed out|connection|dns|socket|tls|ssl|http|"
        r"503|502|504|429|refused|reset|unreachable|econnreset|"
        r"etimedout|enotfound|eai_again|network)\b", re.I),
    "container": re.compile(
        r"\b(docker|containerd|podman|image|registry|kubernetes|k8s|pod|"
        r"helm|buildkit|buildx|container|kubectl|runc|oci)\b", re.I),
    "infra": re.compile(
        r"\b(runner|agent|executor|pod|docker|kubernetes|oom|disk|no space|"
        r"queued|scheduler|shutdown signal|lost communication|worker process|"
        r"buildkitd|containerd)\b", re.I),
    "scm": re.compile(
        r"\b(git|checkout|clone|submodule|lfs|remote repository|"
        r"github\.com|gitlab\.com|bitbucket)\b", re.I),
    "deploy": re.compile(
        r"\b(deploy|release|publish|upload|kubectl|helm|terraform|"
        r"cloudformation|pulumi|serverless|heroku)\b", re.I),
    "artifact": re.compile(
        r"\b(artifact|cache|workspace|upload-artifact|download-artifact|"
        r"restore-cache|save-cache|archive)\b", re.I),
}

# Step name → label family boost mapping
# If failing_step contains these keywords, boost the corresponding family
_STEP_BOOSTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(pytest|unittest|nose|rspec|jest|mocha|vitest|karma|"
                r"go test|cargo test|surefire|failsafe|nunit|xunit)\b", re.I),
     "test"),
    (re.compile(r"\b(compile|build|javac|tsc|rustc|gcc|cmake|make|gradle|"
                r"maven.*compil|kotlin)\b", re.I),
     "build"),
    (re.compile(r"\b(set.?up|install|setup|bootstrap|init|configure|"
                r"pip install|npm install|yarn|bundle install)\b", re.I),
     "dependency"),
    (re.compile(r"\b(docker.*build|docker.*push|buildx|buildkit|"
                r"container|image)\b", re.I),
     "container"),
    (re.compile(r"\b(deploy|release|publish|helm|kubectl|terraform)\b", re.I),
     "deploy"),
    (re.compile(r"\b(lint|format|flake8|mypy|ruff|eslint|checkstyle|"
                r"spotless|black|isort|prettier|rubocop)\b", re.I),
     "static_analysis_step"),
    (re.compile(r"\b(checkout|clone|fetch|git)\b", re.I),
     "scm"),
]


def infer_context(text: str, meta: str, failing_step: str) -> dict[str, bool]:
    joined = f"{meta}\n{text[:8000]}"
    ctx = {k: bool(rx.search(joined)) for k, rx in _CTX.items()}
    # Step-name boosting — adds synthetic context keys
    for rx, key in _STEP_BOOSTS:
        if rx.search(failing_step):
            ctx[key] = True
            # Propagate: static_analysis_step → not test, not compile
            if key == "static_analysis_step":
                ctx["static_lint_step"] = True
    return ctx


# ── Failure anchor markers ────────────────────────────────────────────────────
FAILURE_MARKERS = [
    "error:", "failed", "failure", "exception", "traceback", "fatal:",
    "process completed with exit code", "exit code", "assertionerror",
    "build failed", "test failed", "cannot find symbol", "syntaxerror",
    "permission denied", "timed out", "timeout", "connection refused",
    "no space left on device", "oomkilled", "segmentation fault", "panic:",
    "no files to upload", "failed to pull image", "manifest unknown",
    "##[error]", "::error ::", "unexpected http response",
    "lost communication", "shutdown signal", "worker process exited",
    "no route to host", "name resolution", "certificate verify",
    "imagepullbackoff", "crashloopbackoff", "killed", "enospc",
]


# ══════════════════════════════════════════════════════════════════════════════
#  RULE LIBRARY  (v6 — post 5-pass audit)
# ══════════════════════════════════════════════════════════════════════════════
RULES: list[Rule] = [

    # ══════════════════════════════════════════════════════════════════════
    #  SOURCE CHECKOUT  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("scm.auth",
      r"permission denied \(publickey\)"
      r"|could not read from remote repository"
      r"|authentication failed for.{0,80}(github|gitlab|bitbucket)"
      r"|fatal: authentication failed"
      r"|ssh: handshake failed"
      r"|bad credentials.{0,60}git"
      r"|remote: invalid username or password"
      r"|git@.{0,80}: permission denied"
      r"|fatal: could not read username for",
      4.2, ["scm", "auth"]),

    R("scm.ref_missing",
      r"fatal: couldn't find remote ref"
      r"|pathspec .{0,100} did not match any file"
      r"|reference is not a tree"
      r"|invalid reference"
      r"|could not find ref"
      r"|not our ref"
      r"|no such ref was fetched",
      4.0, ["scm", "ref"]),

    R("scm.network",
      r"fatal: unable to access .{0,120}(could not resolve host|failed to connect|connection timed out|connection reset|gnutls recv error)"
      r"|fatal: the remote end hung up unexpectedly"
      r"|error: rpc failed; (http|curl)"
      r"|fatal: early eof"
      r"|fetch-pack: unexpected disconnect"
      r"|fatal: fetch-pack: invalid index-pack output"
      r"|remote: http 503",
      3.8, ["scm", "network"]),

    R("scm.submodule",
      r"submodule.{0,80}(failed|error)"
      r"|no url found for submodule"
      r"|fatal: clone of '.{0,200}' into submodule path"
      r"|submodule update.*failed",
      3.5, ["scm", "submodule"]),

    R("scm.git_lfs",
      r"git.?lfs"
      r"|smudge filter lfs failed"
      r"|lfs:.{0,80}(authentication|bad credentials|not found)"
      r"|error downloading object.{0,80}smudge error"
      r"|batch request.*lfs.*failed",
      3.5, ["scm", "lfs"]),

    # ══════════════════════════════════════════════════════════════════════
    #  COMPILATION  →  compilation
    # ══════════════════════════════════════════════════════════════════════
    R("compile.syntax",
      r"syntaxerror:"
      r"|indentationerror:"
      r"|taberror:"
      r"|invalid syntax"
      r"|parse error"
      r"|error: ';' expected"
      r"|error: '.' expected"
      r"|unterminated string literal"
      r"|unexpected token"
      r"|expected expression"
      r"|unexpected end of input",
      4.2, ["compile", "syntax"]),

    R("compile.symbol_resolution",
      r"cannot find symbol"
      r"|unresolved reference:"
      r"|nameerror: name .{0,100} is not defined"
      r"|undefined reference to"
      r"|cannot resolve symbol"
      r"|use of undeclared identifier"
      r"|not declared in this scope"
      r"|no module named .{0,80}"
      r"|modulenotfounderror"
      r"|importerror: cannot import name"
      r"|cannot import name .{0,80} from",
      4.0, ["compile", "symbol"]),

    R("compile.java_kotlin",
      r"\[error\]\s+compilation error"
      r"|compilation failed; see the compiler error output"
      r"|\[error\].+\.java:\d+"
      r"|maven-compiler-plugin.{0,80}failure"
      r"|javac.{0,40}error"
      r"|e: .+\.kt:\(\d+,\d+\):"
      r"|kotlin compilation error"
      r"|execution failed for task.{0,60}compil"
      r"|> task :.{0,80}compil.{0,40} failed"
      r"|package .{0,100} does not exist"
      r"|error: aborting due to \d+ previous errors?",
      4.2, ["compile", "jvm"]),

    R("compile.typescript",
      r"error ts\d+:"
      r"|ts\d+: error ts"
      r"|tsc.{0,60}found \d+ errors?"
      r"|found \d+ errors? in \d+ files?"
      r"|type error:"
      r"|typescript.*\d+ error",
      3.8, ["compile", "typescript"]),

    R("compile.rust_go",
      r"rustc.{0,40}error\["
      r"|error\[e\d+\]:"
      r"|^error: aborting due to"
      r"|could not compile\s+`"
      r"|go build.{0,60}failed"
      r"|build constraints exclude all go files"
      r"|could not find crate"
      r"|cargo.*aborting due to"
      r"|^fail\s+\S+\s+\[build failed\]",
      3.8, ["compile", "rust_go"]),

    R("compile.linker_native",
      r"undefined reference to"
      r"|ld returned \d+ exit status"
      r"|linker command failed"
      r"|collect2: error: ld returned"
      r"|ld: library not found"
      r"|cannot find -l\w+"
      r"|undefined symbols for architecture"
      r"|link\.exe.*fatal error lnk",
      3.8, ["compile", "linker"]),

    R("compile.msvc",
      r"error c\d{4}:"
      r"|fatal error c\d{4}:"
      r"|msbuild.{0,40}error"
      r"|build failed.*msbuild"
      r"|\.vcxproj.{0,40}failed",
      3.5, ["compile", "msvc"]),

    # ══════════════════════════════════════════════════════════════════════
    #  STATIC ANALYSIS  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("static.lint",
      r"flake8.{0,60}error"
      r"|ruff.{0,60}error"
      r"|pylint.{0,60}(error|your code has been rated)"
      r"|eslint.{0,60}error"
      r"|tslint.{0,60}error"
      r"|mypy.{0,60}error"
      r"|checkstyle.{0,60}violations were found"
      r"|spotbugs.{0,60}violations"
      r"|golangci-lint.{0,60}issues"
      r"|shellcheck.{0,60}error"
      r"|hadolint.{0,60}error"
      r"|yamllint.{0,60}error"
      r"|actionlint.{0,60}error"
      r"|markdownlint.{0,60}error"
      r"|rubocop.{0,60}(offense|failed)"
      r"|clippy.*error",
      3.5, ["static", "lint"]),

    R("static.format",
      r"black --check"
      r"|black.{0,60}would reformat"
      r"|\d+ files? would be reformatted"
      r"|spotless check failed"
      r"|prettier.{0,60}check.{0,60}failed"
      r"|isort.{0,60}check.{0,60}failed"
      r"|gofmt.{0,60}not formatted"
      r"|cargo fmt.{0,60}check"
      r"|reformatted"
      r"|would reformat",
      3.5, ["static", "format"]),

    # ══════════════════════════════════════════════════════════════════════
    #  DETERMINISTIC TEST FAILURES  →  test_failure
    # ══════════════════════════════════════════════════════════════════════
    R("test.assertion",
      r"assertionerror"
      r"|assertionfailederror"
      r"|comparisonfailure"
      r"|expected:<.{0,300}> but was:<"
      r"|expected: .{0,200} but was:"
      r"|expected \[.{0,200}\] but (got|found|was) \["
      r"|assert.{0,80}failed"
      r"|^\s*e\s+assert\b"
      r"|^\s*e\s+assertionerror"
      r"|not equal: expected"
      r"|org\.opentest4j\.assertionfailederror"
      r"|chai\.assertionerror"
      r"|equalexception"
      r"|expectationnotmeterror"
      r"|assertion `left == right` failed"
      r"|left:.*\n.*right:",
      4.2, ["test", "assertion"]),

    R("test.framework_failures",
      r"there are test failures"
      r"|tests run: \d+.{0,40}failures: [1-9]"
      r"|tests run: \d+.{0,40}errors: [1-9]"
      r"|\d+ tests? failed"
      r"|=+ \d+ failed"
      r"|short test summary info"
      r"|failures!!+"
      r"|pytest.{0,60}\d+ failed"
      r"|rspec.{0,60}failure"
      r"|mocha.{0,60}failing"
      r"|jest.{0,60}failed"
      r"|karma.{0,60}failed"
      r"|vitest.{0,60}failed"
      r"|--- fail:"
      r"|fail\t.{0,60}time="
      r"|<failure "
      r"|<error "
      r"|> task :test failed"
      r"|execution failed for task.{0,40}test"
      r"|gradle.{0,60}test.{0,60}failed"
      r"|test result: failed"
      r"|failures!"
      r"|\d+ test.{0,30}completed.{0,30}\d+ failed",
      3.8, ["test"]),

    R("test.exception",
      r"traceback \(most recent call last\):"
      r"|java\.lang\.[a-z]+exception"
      r"|uncaught exception"
      r"|panic: .{0,300}testing\.trunner"
      r"|thread .{0,80} panicked at",
      2.5, ["test", "exception"], forbids_any=("dependency", "network")),

    R("test.crash",
      r"segmentation fault"
      r"|sigsegv"
      r"|sigbus"
      r"|core dumped"
      r"|fatal python error"
      r"|exit code 139",
      3.2, ["test", "crash"]),

    R("test.timeout_local",
      r"testtimeoutexception"
      r"|test timed out after \d+"
      r"|timeout.*waiting for.{0,80}(test|assert|condition)"
      r"|async.*timeout"
      r"|timed out retrying"
      r"|exceeded timeout of \d+ ms for a test"
      r"|timeout - async callback was not invoked"
      r"|timeout of \d+ms exceeded"
      r"|error: timeout: timed out",
      3.2, ["test", "timeout"], forbids_any=("network", "container", "dependency")),

    R("test.data_fixture",
      r"fixture .{0,80} not found"
      r"|golden file mismatch"
      r"|snapshot .{0,60} failed"
      r"|could not find test data"
      r"|test resource .{0,80} not found",
      3.0, ["test", "fixture"]),

    R("test.xunit_nunit",
      r"xunit\..*failed"
      r"|nunit.*failed"
      r"|dotnet test.*failed"
      r"|failed!\s*-\s*failed: \d+"
      r"|expected:.*\n.*actual:"
      r"|should\.equal|should\.be",
      3.2, ["test", "dotnet"]),

    R("test.playwright_cypress",
      r"playwright.{0,60}failed"
      r"|cypress.{0,60}failed"
      r"|timed out retrying.{0,100}assertion"
      r"|page\.goto.{0,80}failed"
      r"|expect.{0,80}tobevisible.{0,80}timeout"
      r"|cypresserror",
      3.2, ["test", "e2e"]),

    # ══════════════════════════════════════════════════════════════════════
    #  FLAKY / INTERMITTENT  →  flaky_test
    # ══════════════════════════════════════════════════════════════════════
    R("flaky.confirmed_rerun_recovery",
      r"passed on rerun"
      r"|rerun succeeded"
      r"|failed then passed"
      r"|same commit.*pass"
      r"|re-run.*passed",
      5.0, ["flaky", "rerun"], confidence_hint=0.98),

    R("flaky.suspected_retry_sensitive",
      r"rerun failures"
      r"|retrying.*test"
      r"|retry.*attempt \d+"
      r"|passed \d+ times.*failed \d+ times"
      r"|failed \d+ times.*passed \d+ times"
      r"|flakytest"
      r"|@flaky"
      r"|\bflaky\b"
      r"|intermittent"
      r"|non.?deterministic"
      r"|sporadic"
      r"|unstable test"
      r"|flakes: [1-9]"
      r"|this test is flaky",
      3.8, ["flaky", "retry"]),

    R("flaky.order_or_race",
      r"race condition"
      r"|data race"
      r"|deadlock detected"
      r"|concurrentmodificationexception"
      r"|order dependent"
      r"|random seed"
      r"|thread.{0,60}timed out"
      r"|warning: data race"
      r"|found \d+ data race"
      r"|threadsanitizer: data race"
      r"|lock wait timeout exceeded"
      r"|sqlstate\[40p01\]"
      r"|deadlock found when trying to get lock"
      r"|could not obtain lock on row",
      3.5, ["flaky", "race"]),

    R("flaky.time_or_async",
      r"eventually.*failed"
      r"|awaitility"
      r"|timing dependent"
      r"|clock skew.*test"
      r"|sockettimeoutexception"
      r"|readtimeoutexception"
      r"|connectiontimeoutexception"
      r"|jest did not exit one second after"
      r"|jest has detected.*open handle"
      r"|coroutine .{0,80} was never awaited"
      r"|task was destroyed but it is pending"
      r"|staleelementreferenceexception"
      r"|elementnotinteractableexception"
      r"|elementclickinterceptedexception"
      r"|invalid session id"
      r"|chrome not reachable"
      r"|unhandledpromiserejectionwarning"
      r"|unhandled promise rejection"
      r"|disconnected: not connected to devtools",
      2.8, ["flaky", "time"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — RUNNER CONTROL PLANE  →  infrastructure
    #  Always infrastructure. Highest weights in the library.
    # ══════════════════════════════════════════════════════════════════════
    R("infra.runner_lost",
      r"the runner has received a shutdown signal"
      r"|runner.{0,60}lost communication"
      r"|runner.{0,60}lost connection"
      r"|lost communication with the server"
      r"|worker process exited with code"
      r"|runner.{0,60}was removed"
      r"|runner.{0,60}went offline"
      r"|channel is closing down"
      r"|connection to agent was broken"
      r"|hudson\.remoting"
      r"|jenkins agent.{0,60}terminated"
      r"|post request to .{0,200}renewjob timed out"
      r"|get request to .{0,200}broker\.actions\.githubusercontent\.com.{0,100}timed out"
      r"|pipelines.{0,60}\.actions\.githubusercontent\.com.{0,100}timed out",
      5.0, ["infra", "runner"], confidence_hint=0.97),

    R("infra.runner_offline",
      r"no runners available"
      r"|waiting for a runner"
      r"|runner.{0,60}offline"
      r"|stuck because.{0,80}no active runners"
      r"|queued for.{0,80}waiting for executor",
      3.8, ["infra", "runner", "queue"]),

    R("infra.scheduler_queue",
      r"job is stuck"
      r"|stuck_or_timeout_failure"
      r"|scheduler_failure"
      r"|pending for too long"
      r"|queued for too long"
      r"|resource_group.*waiting",
      3.5, ["infra", "scheduler"]),

    R("infra.startup_failure",
      r"startup_failure"
      r"|failed to initialize runner"
      r"|prepare environment failed"
      r"|failed to prepare environment"
      r"|executor failed to start"
      r"|prepare environment:.*context deadline exceeded",
      3.8, ["infra", "startup"]),

    R("infra.agent_connection",
      r"hudson\.remoting"
      r"|channel is closing down"
      r"|connection to agent was broken"
      r"|jenkins agent.{0,60}terminated"
      r"|agent.*unexpectedly terminated",
      3.8, ["infra", "agent"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — OOM / DISK / RESOURCE  →  infrastructure
    # ══════════════════════════════════════════════════════════════════════
    R("infra.resources.oom",
      r"process completed with exit code 137"
      r"|process completed with exit code 143"
      r"|oomkilled"
      r"|outofmemoryerror"
      r"|java\.lang\.outofmemoryerror"
      r"|cannot allocate memory"
      r"|killed.*oom"
      r"|memory cgroup out of memory"
      r"|out of memory: kill"
      r"|fatal error: runtime: out of memory"
      r"|javascript heap out of memory"
      r"|fatal error:.*ineffective mark-compacts"
      r"|node.*heap out of memory"
      r"|signal: killed"
      r"|signal: terminated"
      r"|updated oom_score_adj",
      4.5, ["infra", "resource", "oom"]),

    R("infra.resources.disk_full",
      r"no space left on device"
      r"|enospc"
      r"|disk.?pressure"
      r"|ephemeral-storage"
      r"|not enough space"
      r"|disk quota exceeded"
      r"|write.{0,60}no space left"
      r"|edquot"
      r"|node had condition: \[diskpressure\]",
      4.2, ["infra", "resource", "disk"]),

    R("infra.resources.quota_fd",
      r"too many open files"
      r"|emfile"
      r"|resource temporarily unavailable"
      r"|max user processes"
      r"|cannot fork",
      3.5, ["infra", "resource", "quota"]),

    R("infra.resources.cpu_memory_pressure",
      r"cpu throttling"
      r"|evicted.*memorypressure"
      r"|node.*memorypressure"
      r"|insufficient cpu"
      r"|insufficient memory"
      r"|context deadline exceeded.*resource",
      3.2, ["infra", "resource"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — GITHUB ACTIONS HTTP ERRORS  →  infrastructure
    #  Covers tool/action downloads failing mid-run with 502/503,
    #  broker/renewjob timeouts, cache service errors.
    # ══════════════════════════════════════════════════════════════════════
    R("infra.network.service_5xx",
      r"unexpected http response: 5\d\d"
      r"|httperror: unexpected http response"
      r"|httpstatuscode: 50[234]"
      r"|return code is: 50[234]"
      r"|response status code does not indicate success: 50[234]"
      r"|503 service unavailable"
      r"|502 bad gateway"
      r"|504 gateway timeout"
      r"|http error 5\d\d"
      r"|waiting \d+ seconds before trying again"
      r"|##\[error\]unhandled error: error: unexpected http"
      r"|cache service responded with [45]\d\d"
      r"|failed to save cache"
      r"|failed to restore cache"
      r"|error: unable to download"
      r"|tool-cache.*error"
      r"|downloading.{0,80}(502|503)"
      r"|##\[error\].{0,300}http",
      3.8, ["infra", "network", "http_5xx"]),

    R("infra.network.rate_limit",
      r"rate limit.*exceeded"
      r"|api rate limit"
      r"|too many requests"
      r"|http.*429"
      r"|secondary rate limit"
      r"|toomanyrequests"
      r"|you have reached your pull rate limit"
      r"|retry-after"
      r"|you have exceeded a secondary rate limit",
      3.8, ["infra", "network", "rate_limit"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — DNS / TLS / SOCKET  →  infrastructure
    # ══════════════════════════════════════════════════════════════════════
    R("infra.network.dns",
      r"temporary failure in name resolution"
      r"|name or service not known"
      r"|could not resolve host"
      r"|getaddrinfo enotfound"
      r"|eai_again"
      r"|dns lookup failed"
      r"|no such host"
      r"|dial tcp.{0,100}no such host"
      r"|failed to resolve"
      r"|name resolution failure",
      4.2, ["infra", "network", "dns"]),

    R("infra.network.tls_ssl",
      r"tls handshake timeout"
      r"|tls handshake (error|failure)"
      r"|certificate verify failed"
      r"|ssl.*error"
      r"|x509: certificate"
      r"|self.?signed certificate"
      r"|unable to get local issuer certificate"
      r"|pkix path.{0,60}failed"
      r"|ssl_connect returned"
      r"|bad handshake"
      r"|javax\.net\.ssl\.sslhandshakeexception"
      r"|tlsv1 alert protocol version"
      r"|ssl_error_syscall",
      3.8, ["infra", "network", "tls"]),

    R("infra.network.socket",
      r"connection refused"
      r"|connection reset by peer"
      r"|econnreset"
      r"|broken pipe"
      r"|epipe"
      r"|network is unreachable"
      r"|no route to host"
      r"|ehostunreach"
      r"|failed to establish a new connection"
      r"|failed to connect"
      r"|errno 111"
      r"|errno 110"
      r"|use of closed network connection"
      r"|read: connection reset"
      r"|dial unix /var/run/docker\.sock: connect: connection refused",
      3.2, ["infra", "network", "socket"]),

    R("infra.network.timeout",
      r"read timed out"
      r"|socket timeout"
      r"|connect timed out"
      r"|operation timed out"
      r"|request timed out"
      r"|deadline exceeded"
      r"|i/o timeout"
      r"|httpclient\.timeout",
      2.0, ["infra", "network", "timeout"],
      forbids_any=("test",)),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — PACKAGE REGISTRY NETWORK  →  infrastructure
    #  NETWORK failures against public registries.
    #  Key: retries present + connection error = infra, not config.
    # ══════════════════════════════════════════════════════════════════════
    R("infra.network.registry",
      r"npm err!.{0,80}(econnreset|etimedout|eai_again|enotfound|network request failed)"
      r"|request to https?://<url:registry\.npmjs\.org>.{0,200}failed, reason:"
      r"|pip._vendor.urllib3.exceptions.readtimeouterror"
      r"|readtimeouterror.*httpsconnectionpool"
      r"|retrying.{0,100}(readtimeouterror|connectionerror|httpsconnectionpool)"
      r"|could not fetch url https?://<url:pypi"
      r"|gem::remotefetcher::fetcherror"
      r"|bundler::httperror could not fetch specs"
      r"|too many connection resets.{0,80}https?"
      r"|spurious network error.{0,60}tries remaining"
      r"|could not get https?://<url:crates\.io"
      r"|failed to fetch.{0,200}(archive\.ubuntu\.com|security\.ubuntu\.com)"
      r"|e: failed to fetch.{0,200}(503|502|timed out|connection refused)"
      r"|apt.{0,60}connect to.{0,100}timed out"
      r"|hash sum mismatch"
      r"|org\.apache\.http\.nohttpresponseexception"
      r"|maven.*return code is: 50[234]"
      r"|could not transfer artifact.{0,100}(503|502|timeout|connection)"
      r"|warning: retrying.{0,100}after connection broken"
      r"|gradle.{0,80}downloading.{0,80}gradle-\S+\.zip.{0,80}(failed|timeout)",
      3.8, ["infra", "network", "registry"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — DOCKER / CONTAINER  →  infrastructure
    # ══════════════════════════════════════════════════════════════════════
    R("infra.container.runtime",
      r"error response from daemon"
      r"|cannot connect to the docker daemon"
      r"|is the docker daemon running"
      r"|failed to create shim task"
      r"|runc create failed"
      r"|docker: error response"
      r"|failed to start container"
      r"|buildkit.{0,60}failed to solve"
      r"|failed to copy: httpreadseeker"
      r"|failed to copy: cannot reuse body"
      r"|buildkitd.*failed"
      r"|failed to register layer",
      3.8, ["infra", "container", "runtime"]),

    R("infra.container.docker_socket_perm",
      r"permission denied.{0,80}docker\.sock"
      r"|docker\.sock.{0,80}permission denied"
      r"|connect: permission denied.{0,80}docker"
      r"|permission denied while trying to connect to the docker daemon"
      r"|dial unix /var/run/docker\.sock: connect: permission denied",
      4.5, ["infra", "container", "permission"]),

    R("infra.container.image_not_found",
      r"manifest unknown"
      r"|image not found"
      r"|repository .{0,120} not found"
      r"|not found: manifest"
      r"|name unknown"
      r"|errimagenotfound",
      3.5, ["infra", "container", "image"],
      note="persistent same tag → config; intermittent → infra"),

    R("infra.container.registry_server_error",
      r"failed to pull image.{0,120}(503|502|504|service unavailable|timeout)"
      r"|registry.{0,120}(503|502|504|service unavailable)"
      r"|error pulling image.{0,120}(503|502|504)"
      r"|toomanyrequests.*docker"
      r"|you have reached your pull rate limit",
      3.8, ["infra", "container", "registry"]),

    R("infra.container.image_pull_auth",
      r"pull access denied"
      r"|unauthorized: authentication required"
      r"|denied: requested access to the resource is denied"
      r"|imagepullbackoff.{0,100}(unauthorized|forbidden)"
      r"|docker.{0,80}bad credentials"
      r"|docker.{0,80}incorrect username or password",
      3.5, ["infra", "container", "auth"]),

    R("infra.container.entrypoint_or_arch",
      r"exec format error"
      r"|standard_init_linux\.go"
      r"|executable file not found in \$path"
      r"|no matching manifest for linux"
      r"|requested image.{0,80}platform"
      r"|the requested image.{0,80}platform does not match",
      3.8, ["infra", "container", "arch"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — KUBERNETES / POD  →  infrastructure
    # ══════════════════════════════════════════════════════════════════════
    R("infra.pod_start_timeout",
      r"timed out waiting for pod to start"
      r"|waiting for pod running"
      r"|pod .{0,80} failed to start"
      r"|containersnotready"
      r"|pod has unbound immediate persistentvolumeclaims"
      r"|crashloopbackoff"
      r"|imagepullbackoff"
      r"|context deadline exceeded.*pod",
      4.2, ["infra", "kubernetes", "timeout"]),

    # ══════════════════════════════════════════════════════════════════════
    #  INFRASTRUCTURE — JOB TIMEOUTS / CANCEL  →  infrastructure
    # ══════════════════════════════════════════════════════════════════════
    R("timeout.job_execution",
      r"job execution timeout"
      r"|job has exceeded the maximum execution time"
      r"|process completed with exit code 124"
      r"|process completed with exit code 125"
      r"|process completed with exit code 100"
      r"|command timed out"
      r"|the job was canceled"
      r"|build timed out",
      3.5, ["timeout"]),

    R("timeout.no_output",
      r"no output has been received"
      r"|too long with no output"
      r"|activity timeout"
      r"|log output timeout",
      3.5, ["timeout", "no_output"]),

    R("cancel.concurrency",
      r"canceling since a higher priority waiting request exists"
      r"|cancelled because a newer run was started"
      r"|concurrency group"
      r"|superseded by a newer build",
      3.5, ["cancel", "concurrency"]),

    R("cancel.manual",
      r"canceled by user"
      r"|cancelled by user"
      r"|manually canceled"
      r"|manual abort",
      3.5, ["cancel"]),

    R("cancel.provider",
      r"the operation was canceled"
      r"|workflow was cancelled"
      r"|build was canceled",
      1.8, ["cancel"]),

    # ══════════════════════════════════════════════════════════════════════
    #  DEPENDENCY RESOLUTION  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("dep.not_found",
      r"could not find artifact"
      r"|could not find a version that satisfies the requirement"
      r"|no matching distribution found for"
      r"|dependency resolution failed"
      r"|package .{0,120} does not exist"
      r"|unable to find package"
      r"|module not found: can't resolve"
      r"|could not resolve dependencies"
      r"|no candidates found for"
      r"|npm err! 404.{0,100}not in.{0,60}registry"
      r"|npm err! notarget no matching version"
      r"|npm err! etarget"
      r"|bundler::gemnotfound: could not find"
      r"|go: module .{0,120} not found"
      r"|go: .{0,120}@.{0,60}: unknown revision"
      r"|cargo.*no matching package named"
      r"|non-resolvable parent pom"
      r"|could not find.*in any channel",
      3.8, ["dependency", "not_found"]),

    R("dep.version_conflict",
      r"version conflict"
      r"|dependency convergence error"
      r"|could not resolve version conflict"
      r"|peer dep.{0,60}conflict"
      r"|resolution impossible"
      r"|because .{0,120} depends on .{0,120} depends on"
      r"|requires .{0,100} but .{0,80} is installed"
      r"|lock file.{0,60}out of date"
      r"|npm err! eresolve"
      r"|npm err! could not resolve dependency"
      r"|npm err! eunsupportedprotocol"
      r"|bundler could not find compatible versions"
      r"|cargo.*could not select a version for",
      3.5, ["dependency", "version_conflict"]),

    R("dep.checksum_or_integrity",
      r"checksum (failed|mismatch)"
      r"|integrity check failed"
      r"|hash mismatch"
      r"|sha256 mismatch"
      r"|tarball data.{0,60}corrupted"
      r"|verifying.{0,100}checksum mismatch"
      r"|go: .{0,120}checksum mismatch"
      r"|npm err!.*integrity check"
      r"|error inflating zlib stream",
      3.8, ["dependency", "integrity"]),

    R("dep.registry_unavailable",
      r"could not transfer artifact"
      r"|failed to transfer"
      r"|premature end of content-length"
      r"|org\.apache\.http\.nohttpresponseexception"
      r"|could not get.{0,80}crates\.io-index"
      r"|failed to generate package metadata"
      r"|distribution not found at: file://"
      r"|resolution will not be reattempted until",
      3.5, ["dependency", "registry"]),

    R("dep.toolchain_incompatible",
      r"requires python\s*[>=<~!]+\s*\d+\.\d+"
      r"|unsupported class file major version"
      r"|node version.{0,60}not supported"
      r"|engine .{0,100} incompatible"
      r"|requires go >="
      r"|java.{0,60}version.{0,60}not supported"
      r"|unsupported java"
      r"|java\.lang\.unsupportedclassversionerror"
      r"|rustc.{0,60}is not supported",
      3.3, ["dependency", "toolchain"]),

    R("dep.auth",
      r"(401 unauthorized|403 forbidden|unauthorized|forbidden|bad credentials).{0,200}(npm|pip|pypi|maven|gradle|nuget|gem|cargo|registry|artifact|repository)"
      r"|(npm|pip|pypi|maven|gradle|nuget|gem|cargo|registry|artifact).{0,200}(401 unauthorized|403 forbidden|unauthorized|forbidden|bad credentials)"
      r"|npm err! code e401"
      r"|npm err! code e403",
      3.8, ["dependency", "auth"]),

    # ══════════════════════════════════════════════════════════════════════
    #  CONFIGURATION  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("config.secret_missing",
      r"missing secret"
      r"|secret .{0,100} not found"
      r"|required secret"
      r"|credentials not found"
      r"|credentialnotfoundexception"
      r"|could not find credentials"
      r"|no credentials configured"
      r"|secrets\.[a-z0-9_]+ is not set"
      r"|aws_access_key_id.{0,60}not set"
      r"|aws_secret_access_key.{0,60}not set"
      r"|gcp.*credentials.*not found",
      4.2, ["config", "secret"]),

    R("config.env_missing_invalid",
      r"required environment variable .{0,100} is not set"
      r"|environment variable .{0,100} (is not set|not found|missing)"
      r"|missing environment variable"
      r"|\.env.{0,60}not found"
      r"|keyerror: '[a-z_][a-z0-9_]*'"
      r"|bash: .{0,60}: unbound variable"
      r"|: parameter not set"
      r"|: parameter null or not set",
      3.8, ["config", "env"]),

    R("config.yaml_syntax",
      r"invalid yaml"
      r"|yaml\.scanner\.scannerror"
      r"|yaml parse error"
      r"|mapping values are not allowed"
      r"|did not find expected key"
      r"|found character '\\t' that cannot start any token"
      r"|could not find expected ':'"
      r"|while parsing a block mapping",
      4.2, ["config", "yaml"]),

    R("config.json_toml_syntax",
      r"tomldecodeerror"
      r"|jsondecodeerror"
      r"|json.{0,40}parse.{0,40}error"
      r"|invalid json"
      r"|invalid toml"
      r"|syntaxerror: unexpected token.{0,80}json"
      r"|expected value: line \d+ column \d+",
      3.8, ["config", "schema"]),

    R("config.invalid_ci_schema",
      r"workflow is not valid"
      r"|invalid workflow file"
      r"|the template is not valid"
      r"|config file is invalid"
      r"|unknown keys:"
      r"|schema validation failed"
      r"|unrecognized named-value"
      r"|input required and not supplied"
      r"|jobs config should contain"
      r"|included file .{0,100} does not exist",
      3.8, ["config", "ci_schema"]),

    R("config.runtime_version",
      r"unsupported.*version"
      r"|version.*not supported"
      r"|unable to find node version"
      r"|unable to find go version"
      r"|could not find java version"
      r"|version .{0,80} not found.{0,80}platform"
      r"|no such file.{0,60}python\d"
      r"|error: unable to find node version",
      3.2, ["config", "version"]),

    R("config.file_path_missing",
      r"no such file or directory"
      r"|file not found"
      r"|cannot find.*file"
      r"|path does not exist"
      r"|working-directory .{0,100} does not exist"
      r"|cannot stat .{0,120}: no such file"
      r"|no files were found with the provided path",
      2.5, ["config", "path"], forbids_any=("artifact",)),

    R("config.context_or_scope",
      r"resource not accessible by integration"
      r"|context .{0,80} not found"
      r"|not authorized to use context"
      r"|github token.{0,60}permission"
      r"|insufficient permission"
      r"|403 forbidden.{0,100}github",
      3.5, ["config", "scope"]),

    R("config.precommit",
      r"hookid:"
      r"|pre-commit.{0,60}failed"
      r"|pre-commit hook"
      r"|files were modified by this hook"
      r"|an unexpected error has occurred.{0,60}pre-commit",
      3.5, ["config", "precommit"]),

    # ══════════════════════════════════════════════════════════════════════
    #  AUTH / PERMISSIONS  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("auth.workspace_or_file_perm",
      r"permission denied"
      r"|access denied"
      r"|eacces"
      r"|operation not permitted"
      r"|read-only file system"
      r"|mkdir.{0,60}permission denied",
      2.2, ["auth", "permission"],
      forbids_any=("scm", "dependency", "container")),

    R("auth.cloud",
      r"an error occurred \(invalidclienttokenid\)"
      r"|an error occurred \(signaturedoesnotmatch\)"
      r"|an error occurred \(accessdenied\)"
      r"|an error occurred \(expiredtokenexception\)"
      r"|invalid_grant:.{0,80}(bad request|account not found)"
      r"|aadsts\d+:"
      r"|google\.api_core\.exceptions\.permissiondenied"
      r"|unable to locate credentials"
      r"|no credentials provided"
      r"|the security token included in the request is expired",
      3.8, ["auth", "cloud"]),

    R("policy.branch_or_protection",
      r"protected branch"
      r"|branch protection"
      r"|required status check"
      r"|changes were rejected because"
      r"|cannot force-push"
      r"|deployment protected"
      r"|! \[rejected\].{0,100}protected",
      3.5, ["policy", "branch"]),

    R("policy.action_or_workflow_restricted",
      r"actions are disabled"
      r"|workflow was not approved"
      r"|not allowed to use action"
      r"|github actions is disabled"
      r"|workflow run was rejected",
      3.5, ["policy", "workflow"]),

    # ══════════════════════════════════════════════════════════════════════
    #  ARTIFACTS / CACHE  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("artifact.upload_missing",
      r"no files to upload"
      r"|no matching files"
      r"|no artifacts found that match"
      r"|path does not exist.{0,80}artifact"
      r"|upload-artifact.{0,60}no files",
      3.8, ["artifact", "missing"]),

    R("artifact.download_missing_expired",
      r"artifact.{0,60}not found"
      r"|artifact.{0,60}expired"
      r"|no artifact named"
      r"|download artifact.{0,60}404",
      3.5, ["artifact", "expired"]),

    R("artifact.permission_or_encoding",
      r"artifact.{0,60}permission denied"
      r"|artifact.{0,60}unauthorized"
      r"|artifact.{0,60}forbidden"
      r"|invalid artifact path"
      r"|artifact name.{0,60}invalid",
      3.2, ["artifact", "permission"]),

    R("cache.corrupt",
      r"cache.{0,60}corrupt"
      r"|invalid cache"
      r"|failed to extract cache"
      r"|tar.{0,60}unexpected eof"
      r"|cache archive.{0,60}invalid",
      2.5, ["cache", "corrupt"]),

    R("workspace.path_or_cleanup",
      r"cleanup.{0,60}eacces"
      r"|post job cleanup.{0,60}failed"
      r"|failed to clean workspace"
      r"|workspace cleanup failed"
      r"|directory not empty",
      2.8, ["workspace", "cleanup"]),

    # ══════════════════════════════════════════════════════════════════════
    #  DEPLOY / RELEASE  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("deploy.cluster_api",
      r"kubectl.{0,80}(forbidden|unauthorized|timed out|connection refused)"
      r"|helm.{0,80}(forbidden|timed out|failed)"
      r"|kubernetes cluster unreachable"
      r"|the server doesn't have a resource type"
      r"|error: you must be logged in to the server"
      r"|kubectl.{0,60}error from server",
      3.5, ["deploy", "kubernetes"]),

    R("deploy.resource_not_ready",
      r"rollout status.{0,100}exceeded its progress deadline"
      r"|deployment .{0,120} exceeded its progress deadline"
      r"|pods?.{0,60}not ready"
      r"|crashloopbackoff"
      r"|imagepullbackoff",
      3.5, ["deploy", "not_ready"]),

    R("deploy.package_publish_auth",
      r"publish.{0,80}(401|403|unauthorized|forbidden)"
      r"|npm publish.{0,40}403"
      r"|twine upload.{0,40}403"
      r"|nuget push.{0,40}unauthorized"
      r"|maven deploy.{0,40}401",
      3.5, ["deploy", "publish"]),

    R("deploy.approval_needed",
      r"approval required"
      r"|waiting for approval"
      r"|manual approval"
      r"|on hold"
      r"|environment protection rules",
      3.2, ["deploy", "approval"]),

    R("deploy.unauthorized_context",
      r"deployment.{0,60}unauthorized"
      r"|forbidden.{0,60}deploy"
      r"|not authorized.{0,60}deploy"
      r"|environment .{0,80} not allowed"
      r"|protected environment"
      r"|helm install.*failed"
      r"|error: installation failed",
      3.2, ["deploy", "auth"]),

    # ══════════════════════════════════════════════════════════════════════
    #  RUNTIME / SCRIPTS  →  configuration
    # ══════════════════════════════════════════════════════════════════════
    R("runtime.command_not_found",
      r"command not found"
      r"|is not recognized as an internal or external command"
      r"|executable file not found"
      r"|/bin/sh: .{0,100}: not found"
      r"|no such file or directory.{0,80}(python|node|java|mvn|gradle|make|cmake|pip)"
      r"|program .{0,80} not found in path",
      3.0, ["runtime", "path"]),

    # Always dampened — only wins if literally nothing else matched
    R("runtime.exit_code",
      r"process completed with exit code [1-9]\d*"
      r"|exited with status [1-9]\d*",
      1.0, ["runtime", "exit_code"]),
]


# ── Log cleaning ──────────────────────────────────────────────────────────────
def clean_log(text: str) -> str:
    # ANSI escape sequences
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    # GitHub Actions group markers only — preserve ##[error] lines
    text = re.sub(r"##\[(?:group|endgroup|debug|section|command)\].*", "", text)
    # ISO timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*", "", text)
    # URLs — preserve hostname for registry/service matching
    text = re.sub(r"https?://([^/\s\"']+)\S*", r"<url:\1>", text)
    # Git SHAs and hashes
    text = re.sub(r"\b[A-Fa-f0-9]{40}\b", "<SHA>", text)
    text = re.sub(r"\b[A-Fa-f0-9]{64}\b", "<HASH>", text)
    return text.strip()


# ── Multi-window extraction ───────────────────────────────────────────────────
def extract_windows(text: str) -> list[dict]:
    lines = text.splitlines()
    if not lines:
        return []

    marker_idxs = [
        i for i, line in enumerate(lines)
        if any(m in line.lower() for m in FAILURE_MARKERS)
    ]
    if not marker_idxs:
        marker_idxs = [len(lines) - 1]

    anchors: list[tuple[str, int]] = [
        ("first_failure", marker_idxs[0]),
        ("last_failure",  marker_idxs[-1]),
    ]

    for i, line in enumerate(lines):
        low = line.lower()
        if ("traceback (most recent call last)" in low
                or "short test summary info" in low
                or re.search(r"=+ failures =+", low)
                or "--- fail:" in low):
            anchors.append(("stack_or_summary", i))
            break

    seen_idxs: set[int] = set()
    windows: list[dict] = []
    for name, idx in anchors:
        if idx in seen_idxs:
            continue
        seen_idxs.add(idx)
        start   = max(0, idx - WINDOW_BEFORE)
        end     = min(len(lines), idx + WINDOW_AFTER)
        snippet = "\n".join(lines[start:end]).strip()
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[-MAX_SNIPPET_CHARS:]
        if len(snippet) >= MIN_TEXT_CHARS:
            windows.append({"name": name, "text": snippet})

    tail = "\n".join(lines[-min(len(lines), WINDOW_BEFORE + WINDOW_AFTER):]).strip()
    if len(tail) >= MIN_TEXT_CHARS and all(w["text"] != tail for w in windows):
        if len(tail) > MAX_SNIPPET_CHARS:
            tail = tail[-MAX_SNIPPET_CHARS:]
        windows.append({"name": "tail", "text": tail})

    return windows


# ── Scoring ───────────────────────────────────────────────────────────────────
WINDOW_MULT = {
    "first_failure":    1.15,
    "stack_or_summary": 1.10,
    "last_failure":     1.00,
    "tail":             0.85,
}


def score_windows(windows: list[dict],
                  ctx: dict[str, bool]) -> tuple[dict, dict]:
    scores: dict[str, float] = defaultdict(float)
    hits:   dict[str, list]  = defaultdict(list)

    for w in windows:
        mult = WINDOW_MULT.get(w["name"], 1.0)
        text = w["text"]

        for rule in RULES:
            if rule.forbids_any and any(ctx.get(k) for k in rule.forbids_any):
                continue

            m = rule.regex.search(text)
            if not m:
                continue

            weight = rule.weight * mult

            # Context boosts
            if rule.label.startswith("test.")           and ctx.get("test"):              weight *= 1.15
            if rule.label.startswith("dep.")            and ctx.get("dependency"):        weight *= 1.15
            if rule.label.startswith("infra.container") and ctx.get("container"):         weight *= 1.15
            if rule.label.startswith("scm.")            and ctx.get("scm"):               weight *= 1.15
            if rule.label.startswith("infra.")          and ctx.get("infra"):             weight *= 1.10
            if rule.label.startswith("deploy.")         and ctx.get("deploy"):            weight *= 1.20
            if rule.label.startswith("artifact.")       and ctx.get("artifact"):          weight *= 1.20
            if rule.label.startswith("static.")         and ctx.get("static_lint_step"):  weight *= 1.20
            # Dampen compile rules in pure test context
            if rule.label.startswith("compile.") and ctx.get("test") and not ctx.get("build"):
                weight *= 0.80
            # Dampen generic terminators
            if rule.label == "runtime.exit_code": weight *= 0.50
            if rule.label == "cancel.provider":   weight *= 0.60

            scores[rule.label] += weight

            if len(hits[rule.label]) < 6:
                hits[rule.label].append({
                    "pattern": rule.pattern[:120],
                    "match":   text[m.start():m.end()].replace("\n", " ")[:200],
                    "weight":  round(weight, 3),
                    "window":  w["name"],
                })

    return dict(scores), dict(hits)


# ── Confidence calibration ────────────────────────────────────────────────────
def calibrate_confidence(best_score: float, second_score: float,
                          n_hits: int, hint: float | None) -> float:
    if hint is not None:
        return hint
    margin = best_score - second_score
    if best_score >= 7.5 and margin >= 3.0 and n_hits >= 2: return 0.96
    if best_score >= 5.5 and margin >= 2.0:                  return 0.92
    if best_score >= 4.0 and margin >= 1.5:                  return 0.86
    if best_score >= 3.0 and margin >= 1.0:                  return 0.76
    if best_score >= 2.2 and margin >= 0.7:                  return 0.66
    return 0.50


# ── Main labeling ─────────────────────────────────────────────────────────────
def classify(row: dict, windows: list[dict]) -> dict:
    meta         = " ".join(str(row.get(k, "")) for k in
                            ("workflow_name", "job_name", "failing_step"))
    failing_step = str(row.get("failing_step", ""))
    ctx          = infer_context(
        windows[0]["text"] if windows else "", meta, failing_step)

    scores, hits = score_windows(windows, ctx)

    if not scores:
        return _unknown("no pattern matched")

    ranked           = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score     = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_label in {"runtime.exit_code", "cancel.provider"} and len(ranked) > 1:
        alt_label, alt_score = ranked[1]
        if alt_score >= best_score * 0.50:
            best_label, best_score = alt_label, alt_score
            second_score = ranked[0][1]

    hint = next(
        (r.confidence_hint for r in RULES
         if r.label == best_label and r.confidence_hint is not None),
        None
    )

    conf    = calibrate_confidence(
        best_score, second_score, len(hits.get(best_label, [])), hint)
    margin  = best_score - second_score
    needs_review = (
        conf < OLLAMA_THRESHOLD
        or (second_score > 0 and margin / max(best_score, 0.01) < 0.18)
    )

    family = label_family(best_label)
    bucket = to_bucket(best_label)

    return {
        "primary_label":    best_label,
        "label_family":     family,
        "secondary_labels": [l for l, _ in ranked[1:5]],
        "rule_confidence":  round(conf, 4),
        "needs_review":     needs_review,
        "rule_reason":      (f"best={best_label} score={best_score:.2f} "
                             f"margin={margin:.2f}"),
        "rule_scores":      {k: round(v, 3) for k, v in ranked[:15]},
        "rule_hits":        hits,
        "rule_context":     ctx,
        "label":            bucket,
        "rule_label":       bucket,
        "rule_confidence_legacy": round(conf, 4),
    }


def _unknown(reason: str) -> dict:
    return {
        "primary_label":    "unknown.other",
        "label_family":     "insufficient_evidence",
        "secondary_labels": [],
        "rule_confidence":  0.0,
        "needs_review":     True,
        "rule_reason":      reason,
        "rule_scores":      {},
        "rule_hits":        {},
        "rule_context":     {},
        "label":            None,
        "rule_label":       None,
        "rule_confidence_legacy": 0.0,
    }


# ── Resume ────────────────────────────────────────────────────────────────────
def load_existing_sources() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    seen: set[str] = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["source_file"])
            except Exception:
                pass
    return seen


# ── Per-row processor ─────────────────────────────────────────────────────────
def process_row(row: dict, seen: set[str]) -> tuple[dict | None, str]:
    log_path = Path(row["log_path"])
    if str(log_path) in seen:
        return None, "already_done"
    if not log_path.exists():
        return None, "missing_log"
    try:
        raw = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, "read_error"

    cleaned = clean_log(raw)
    if len(cleaned) < MIN_TEXT_CHARS:
        return None, "too_short"

    windows = extract_windows(cleaned)
    if not windows:
        return None, "no_windows"

    result = classify(row, windows)

    out_row = {
        "repo":          row.get("repo"),
        "lang":          row.get("lang"),
        "run_id":        row.get("run_id"),
        "run_number":    row.get("run_number"),
        "job_id":        row.get("job_id"),
        "workflow_name": row.get("workflow_name", ""),
        "job_name":      row.get("job_name", ""),
        "failing_step":  row.get("failing_step", ""),
        "created_at":    row.get("created_at", ""),
        "source_file":   str(log_path),
        "text":          windows[0]["text"],
        **result,
    }
    return out_row, "written"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Run 01_fetch_massive_github_logs.py first")

    seen  = load_existing_sources()
    stats: Counter = Counter()

    with INDEX_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    write_lock = threading.Lock()

    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            futures = {pool.submit(process_row, row, seen): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Extracting snippets", unit="log"):
                try:
                    out_row, status = future.result()
                except Exception:
                    stats["error"] += 1
                    continue

                stats[status] += 1
                if out_row is None:
                    continue

                with write_lock:
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

                conf   = float(out_row.get("rule_confidence", 0) or 0)
                bucket = out_row.get("label")
                family = out_row.get("label_family", "unknown")

                if bucket:
                    stats[f"bucket:{bucket}"] += 1
                stats[f"family:{family}"] += 1

                if out_row.get("needs_review"):
                    stats["needs_ollama"] += 1
                elif conf >= 0.86:
                    stats["high_confidence"] += 1
                else:
                    stats["medium_confidence"] += 1

    STATS_PATH.write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    print("\nSnippet extraction complete.\n")
    print("5-bucket distribution (training labels):")
    for b in ("compilation", "test_failure", "flaky_test",
              "infrastructure", "configuration"):
        print(f"  {b:<22} {stats.get(f'bucket:{b}', 0):>7}")
    print(f"\nTotal written    : {stats.get('written', 0)}")
    print(f"Needs Ollama     : {stats.get('needs_ollama', 0)}")
    print(f"High confidence  : {stats.get('high_confidence', 0)}")
    print(f"Medium confidence: {stats.get('medium_confidence', 0)}")
    print(f"Stats saved to   : {STATS_PATH}")


if __name__ == "__main__":
    main()
