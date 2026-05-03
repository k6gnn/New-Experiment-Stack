"""
step3_train_m13.py  — M13 Final Classifier
===========================================
Single clean training script. Merges v7 + Phase-1 v2 improvements.

What is in here:
  - Robust data loader (multiple field name fallbacks, env override)
  - 57 engineered features (counts, timing, frameworks, keywords,
    strong compile sub-features, split config sub-features,
    stage/order features, boundary helpers, ontology one-hots)
  - runtime.exit_code separated from configuration (the key Phase-1 fix)
  - Evidence-gated guardrails — stops configuration from being too greedy
  - Random Forest + GradientBoosting trained and saved separately
  - Clean-label GB default, high-recall RF option, and dual-model hybrid inference
  - Honest 80/20 stratified held-out evaluation
  - Sample weights: real rows by confidence, synthetic at 0.6

Run from ML_Training_Set root:
  python step3_train_m13.py

Override dataset path:
  set M13_DATASET=C:\\path\\to\\github_actions_training_dataset.jsonl

Outputs:
  m13_model.pkl                     # backwards-compatible default alias: clean GB
  m13_model_rf.pkl                  # high-recall compilation option
  m13_model_gb.pkl                  # cleaner default labels
  m13_model_rf_calibrated.pkl       # optional probability-calibrated RF
  m13_model_bundle.pkl              # both models + params for dual inference
  m13_label_encoder.pkl
  m13_feature_names.pkl
  m13_training_report.txt

Requirements:
  pip install pandas numpy scikit-learn joblib tqdm
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_FILE = Path("Ollama_DataGathering/data/final/github_actions_training_dataset.jsonl")

# Backwards-compatible default alias. In this version it points to the clean-label GB model,
# not to calibrated RF.
MODEL_OUT    = "m13_model.pkl"
ENCODER_OUT  = "m13_label_encoder.pkl"
FEATURES_OUT = "m13_feature_names.pkl"
REPORT_OUT   = "m13_training_report.txt"

# Separately saved production candidates.
MODEL_RF_OUT            = "m13_model_rf.pkl"              # high-recall compilation option
MODEL_GB_OUT            = "m13_model_gb.pkl"              # cleaner-label default option
MODEL_RF_CALIBRATED_OUT = "m13_model_rf_calibrated.pkl"   # optional probability-calibrated RF
MODEL_BUNDLE_OUT        = "m13_model_bundle.pkl"          # RF + GB + params for dual inference

# Extra outputs added by the upgrade.
METADATA_OUT             = "m13_metadata.json"
METRICS_OUT              = "m13_metrics.json"
FEATURE_IMPORTANCE_OUT   = "m13_feature_importance.csv"
CONFUSION_MATRIX_OUT     = "m13_confusion_matrix.csv"
ABLATION_RESULTS_OUT     = "m13_ablation_results.csv"
SYNTHETIC_SWEEP_OUT      = "m13_synthetic_weight_sweep.csv"
ERRORS_OUT               = "m13_errors.csv"
ERRORS_COMP_TEST_OUT     = "m13_errors_compilation_vs_test.csv"
ERRORS_CONFIG_COMP_OUT   = "m13_errors_configuration_vs_compilation.csv"
LOW_CONFIDENCE_OUT       = "m13_low_confidence.csv"

TARGET_LABELS = ["compilation", "test_failure", "flaky_test", "configuration", "infrastructure"]

TEST_SIZE        = 0.20
VALIDATION_SIZE  = 0.15  # carved out of train only for guardrail tuning/model selection
RANDOM_STATE     = 42
SYNTHETIC_WEIGHT = 0.60

# Added experiment controls.
PRIMARY_SELECTION_METRIC = "macro_f1_guardrailed"
ENABLE_ABLATION_STUDY    = True
ENABLE_CALIBRATION       = True
SAVE_CALIBRATED_RF       = True   # saved separately; never used as default classifier unless you choose it
CALIBRATION_METHOD       = "sigmoid"
CALIBRATION_CV           = 3
ENABLE_SYNTHETIC_SWEEP   = True
LOW_CONFIDENCE_THRESHOLD = 0.55
LOW_MARGIN_THRESHOLD     = 0.10
SIDE_STUDY_RF_TREES      = 160

# Inference policy: GB is the default clean-label model; RF is available for high-recall compilation use.
DEFAULT_MODEL_POLICY = "clean_gb"
DUAL_DEFAULT_STRATEGY = "hybrid_compile_rescue"
DUAL_RF_COMPILE_MIN_PROBA = 0.55
DUAL_REQUIRE_COMPILE_EVIDENCE = True
DUAL_COMPILE_BOUNDARY_CLASSES = ("test_failure", "configuration")

# Guardrail thresholds — defaults preserved, but now tunable on validation.
CONFIG_TO_COMPILE_MARGIN     = 0.18
CONFIG_TO_TEST_MARGIN        = 0.18
MIN_ALT_PROBA_FOR_GUARDRAIL  = 0.30
GUARDRAIL_COMPILE_MARGINS    = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25]
GUARDRAIL_TEST_MARGINS       = [0.05, 0.10, 0.15, 0.18, 0.20, 0.25]
GUARDRAIL_MIN_ALT_PROBAS     = [0.20, 0.25, 0.30, 0.35, 0.40]

# Synthetic-data impact study. "real_only" drops synthetic rows entirely.
SYNTHETIC_WEIGHT_SWEEP = ["real_only", 0.30, 0.60, 1.00]

# ── Feature names ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # ── Test counts ───────────────────────────────────────────────────────────
    "feat_tests_ran",
    "feat_tests_failed",
    "feat_num_tests_failed",
    "feat_num_tests_run",
    "feat_num_tests_skipped",
    "feat_fail_ratio",
    # ── Timing ────────────────────────────────────────────────────────────────
    "feat_build_duration",
    # ── Framework signals ─────────────────────────────────────────────────────
    "feat_fw_junit",
    "feat_fw_pytest",
    "feat_fw_gradle",
    "feat_fw_vitest",
    "feat_is_java",
    # ── Original keyword features ─────────────────────────────────────────────
    "feat_kw_compile_fail",
    "feat_kw_test_assert",
    "feat_kw_flaky",
    "feat_kw_infra_network",
    "feat_kw_infra_runner",
    "feat_kw_dep_resolution",
    "feat_kw_config_fail",
    # ── Original disambiguation features ──────────────────────────────────────
    "feat_compile_no_tests",
    "feat_compile_with_config",
    "feat_config_no_compile",
    # ── Step name features ────────────────────────────────────────────────────
    "feat_step_is_test",
    "feat_step_is_compile",
    "feat_step_is_lint",
    "feat_step_is_docker",
    "feat_step_is_setup",
    # ── Ontology one-hots (script 02 primary_label) ───────────────────────────
    "feat_primary_is_compile",
    "feat_primary_is_test",
    "feat_primary_is_flaky",
    "feat_primary_is_infra",
    "feat_primary_is_config",
    "feat_primary_is_runtime",   # Phase-1 key fix: runtime.exit_code ≠ configuration
    # ── Strong compile sub-features ───────────────────────────────────────────
    "feat_compile_jvm_strong",
    "feat_compile_ts_strong",
    "feat_compile_rust_go_strong",
    "feat_compile_native_linker",
    "feat_compile_python_syntax",
    "feat_compile_build_task",
    # ── Split config sub-features ─────────────────────────────────────────────
    "feat_config_secret_env",
    "feat_config_yaml_workflow",
    "feat_config_lint_format",
    "feat_config_auth_permission",
    "feat_config_missing_file",
    "feat_config_tool_version",
    "feat_config_dep_resolution_strong",
    # ── Stage / order features ────────────────────────────────────────────────
    "feat_first_error_before_tests",
    "feat_first_error_after_tests_started",
    "feat_error_in_compile_step",
    "feat_error_in_setup_step",
    "feat_error_in_lint_step",
    "feat_error_in_test_step",
    "feat_has_compile_task",
    "feat_has_dependency_install",
    # ── Boundary helper features ──────────────────────────────────────────────
    "feat_compile_over_config_guard",
    "feat_config_strong_only",
    "feat_dep_resolution_without_network",
]

# ── Compiled patterns ─────────────────────────────────────────────────────────

_STEP_TEST    = re.compile(r"\b(pytest|unittest|test|surefire|failsafe|jest|mocha|rspec|vitest|nunit|xunit)\b", re.I)
_STEP_COMPILE = re.compile(r"\b(compile|build|javac|tsc|rustc|gcc|g\+\+|clang|cmake|make|gradle|maven.*compil|kotlin|kotlinc)\b", re.I)
_STEP_LINT    = re.compile(r"\b(lint|format|flake8|mypy|ruff|eslint|checkstyle|spotless|black|isort|prettier|rubocop|clippy)\b", re.I)
_STEP_DOCKER  = re.compile(r"\b(docker|buildx|buildkit|container|image|push|pull)\b", re.I)
_STEP_SETUP   = re.compile(r"\b(set.?up|install|setup|bootstrap|init|configure|pip install|npm install|yarn|pnpm|bundle|composer install)\b", re.I)
_ERROR_ANCHOR = re.compile(
    r"error:|fatal:|failed|failure|exception|traceback|cannot find symbol|syntaxerror|"
    r"compilation failed|compilation error|process completed with exit code|assertionerror|"
    r"no matching distribution|npm err!|could not resolve|permission denied|invalid yaml|"
    r"no such file or directory", re.I)
_TEST_START   = re.compile(
    r"\b(pytest|collected \d+ items|tests run:|surefire|failsafe|jest|vitest|mocha|rspec|go test|cargo test|dotnet test)\b", re.I)


# ── Ontology → one-hot ────────────────────────────────────────────────────────

def primary_to_onehot(primary: str) -> tuple[int, int, int, int, int, int]:
    """Map script 02 ontology hint to 6 features.
    Key: runtime.* is NOT configuration — that was the Phase-1 root cause fix.
    """
    p = (primary or "").lower()
    return (
        int(p.startswith("compile.")),
        int(p.startswith("test.")),
        int(p.startswith("flaky.")),
        int(p.startswith("infra.") or p.startswith("timeout.") or p.startswith("cancel.")),
        int(p.startswith(("config.", "dep.", "auth.", "policy.", "scm.",
                           "artifact.", "cache.", "workspace.", "deploy.", "static."))),
        int(p.startswith("runtime.")),
    )


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*", "", text)
    text = re.sub(r"##\[(?:group|endgroup|debug|section|command)\].*", "", text)
    return text


def _bool(pattern: str, text: str) -> int:
    return int(bool(re.search(pattern, text, re.I | re.M)))


def _first_pos(rx: re.Pattern, text: str) -> int | None:
    m = rx.search(text)
    return None if m is None else m.start()


def _before(a: int | None, b: int | None) -> bool:
    return a is not None and (b is None or a < b)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(row: dict) -> list[float]:
    text         = row.get("text", "")
    lang         = str(row.get("lang", "")).lower()
    failing_step = str(row.get("failing_step", "")).lower()
    primary      = str(row.get("primary_label", ""))

    c = clean(text)

    # ── Test counts ───────────────────────────────────────────────────────────
    maven_runs  = sum(int(x) for x in re.findall(r"tests run: (\d+)", c, re.I))
    maven_fail  = sum(int(x) for x in re.findall(r"failures: (\d+)", c, re.I))
    maven_skip  = sum(int(x) for x in re.findall(r"skipped: (\d+)", c, re.I))
    pytest_fail = sum(int(x) for x in re.findall(r"(\d+) failed", c, re.I))
    pytest_pass = sum(int(x) for x in re.findall(r"(\d+) passed", c, re.I))
    pytest_skip = sum(int(x) for x in re.findall(r"(\d+) skipped", c, re.I))
    pytest_runs = pytest_fail + pytest_pass

    num_run    = maven_runs if maven_runs > 0 else pytest_runs
    num_failed = maven_fail if maven_fail > 0 else pytest_fail
    num_skip   = maven_skip if maven_skip > 0 else pytest_skip
    tests_ran  = 1 if num_run > 0 else 0
    tests_failed_flag = 1 if num_failed > 0 else 0
    fail_ratio = num_failed / num_run if num_run > 0 else 0.0

    # ── Timing ────────────────────────────────────────────────────────────────
    build_dur = 0.0
    bd = re.search(r"BUILD (?:FAILED|SUCCESSFUL) in (\d+)m (\d+)s", c, re.I)
    if bd:
        build_dur = int(bd.group(1)) * 60 + int(bd.group(2))
    else:
        bd2 = re.search(r"BUILD (?:FAILED|SUCCESSFUL) in (\d+)s", c, re.I)
        if bd2:
            build_dur = float(bd2.group(1))
        else:
            elapsed = re.findall(r"Time elapsed: ([\d.]+) s", c, re.I)
            if elapsed:
                build_dur = sum(float(x) for x in elapsed)

    # ── Framework signals ─────────────────────────────────────────────────────
    fw_junit  = _bool(r"\bjunit\b|\bsurefire\b|\bfailsafe\b", c)
    fw_pytest = _bool(r"\bpytest\b|\bpy\.test\b|\bunittest\b", c)
    fw_gradle = _bool(r"\bgradle\b|\bgradlew\b", c)
    fw_vitest = _bool(r"\bvitest\b|\bjest\b|\bmocha\b|\brspec\b", c)
    is_java   = 1 if lang == "java" else 0

    # ── Strong compile sub-features ───────────────────────────────────────────
    compile_jvm_strong = _bool(
        r"\[error\]\s+compilation error|compilation failed; see the compiler error output|"
        r"maven-compiler-plugin.{0,120}(failure|failed)|javac.{0,80}error|"
        r"\.java:\d+: error|e: .+\.kt:\(\d+,\d+\):|kotlin compilation error|"
        r"execution failed for task.{0,80}compil|> task :.{0,100}compil.{0,60}failed|"
        r"package .{0,120} does not exist|cannot find symbol|cannot resolve symbol|"
        r"unresolved reference:", c)
    compile_ts_strong = _bool(
        r"error ts\d+:|ts\d+: error ts|tsc.{0,80}(found|failed|error)|"
        r"typescript.{0,80}\d+ errors?|type error:", c)
    compile_rust_go_strong = _bool(
        r"rustc.{0,60}error\[|error\[e\d+\]:|could not compile\s+`|"
        r"go build.{0,80}failed|build constraints exclude all go files|"
        r"^fail\s+\S+\s+\[build failed\]|error: aborting due to \d+ previous errors?", c)
    compile_native_linker = _bool(
        r"undefined reference to|ld returned \d+ exit status|linker command failed|"
        r"collect2: error: ld returned|ld: library not found|cannot find -l\w+|"
        r"undefined symbols for architecture|link\.exe.*fatal error lnk|"
        r"error c\d{4}:|fatal error c\d{4}:", c)
    compile_python_syntax = _bool(
        r"syntaxerror:|indentationerror:|taberror:|unterminated string literal|"
        r"invalid syntax|parse error|unexpected eof while parsing", c)
    compile_build_task = _bool(
        r"execution failed for task.{0,100}(compile|build)|"
        r"> task :.{0,100}(compile|build).{0,60}failed|"
        r"build failed.*(javac|kotlinc|tsc|rustc|gcc|clang|msbuild|cmake)", c)

    kw_compile = int(any([compile_jvm_strong, compile_ts_strong, compile_rust_go_strong,
                           compile_native_linker, compile_python_syntax, compile_build_task])
                     or bool(re.search(
                         r"compilation error|compilation failed|cannot find symbol|"
                         r"syntaxerror:|indentationerror:|unresolved reference:|error ts\d+:|"
                         r"undefined reference to|rustc.*error\[|could not find crate|"
                         r"kotlin compilation error|error: aborting due to", c, re.I)))

    # ── Test / flaky / infra signals ──────────────────────────────────────────
    kw_assert = _bool(
        r"assertionerror|assertionfailederror|expected:<|failures!!!|failed .+\.py::|"
        r"e\s+assert |short test summary|= failures =|comparisonfailure|"
        r"expected:.*but was:|tests run.*failures: [1-9]|"
        r"assertion .left == right. failed|not equal: expected", c)

    kw_flaky = _bool(
        r"testtimeoutexception|timed out after \d+|concurrentmodificationexception|intermittent|"
        r"race condition|data race|deadlock|non.?deterministic|rerun failures|flakes: [1-9]|"
        r"\bflaky\b|flakytest|sockettimeoutexception|readtimeoutexception|"
        r"passed \d+ times.*failed \d+ times|retry.*attempt \d+|staleelementreferenceexception|"
        r"jest did not exit one second", c)

    kw_infra_network = _bool(
        r"connection refused|connection reset by peer|econnreset|"
        r"temporary failure in name resolution|name or service not known|"
        r"could not resolve host|getaddrinfo enotfound|eai_again|"
        r"tls handshake|certificate verify failed|503 service unavailable|502 bad gateway|"
        r"504 gateway timeout|unexpected http response: 5\d\d|toomanyrequests|"
        r"you have reached your pull rate limit|failed to pull image|manifest unknown|"
        r"error response from daemon|"
        r"failed to fetch.{0,80}(archive\.ubuntu|pypi|npmjs|crates\.io)|"
        r"readtimeouterror.*httpsconnectionpool|"
        r"retrying.{0,80}(readtimeouterror|connectionerror)|"
        r"no space left on device|enospc", c)

    kw_infra_runner = _bool(
        r"the runner has received a shutdown signal|runner.{0,40}lost communication|"
        r"worker process exited with code|process completed with exit code 137|"
        r"outofmemoryerror|java\.lang\.outofmemoryerror|oomkilled|"
        r"javascript heap out of memory|fatal error: runtime: out of memory|"
        r"signal: killed|updated oom_score_adj", c)

    # ── Split config sub-features ─────────────────────────────────────────────
    config_secret_env = _bool(
        r"missing secret|secret .{0,80} not found|credentials not found|"
        r"environment variable .{0,80} is not set|required env(?:ironment)? var|"
        r"input required and not supplied|missing required input|"
        r"api key.{0,80}(missing|not set)", c)
    config_yaml_workflow = _bool(
        r"yaml\.scanner\.scannerror|invalid yaml|yaml parse error|workflow is not valid|"
        r"the workflow is not valid|mapping values are not allowed|"
        r"did not find expected key|unexpected value '.{0,80}'|"
        r"a sequence was not expected", c)
    config_lint_format = _bool(
        r"black --check|would reformat|\d+ files? would be reformatted|flake8|ruff.{0,80}error|"
        r"mypy.*error|eslint.{0,80}error|prettier.{0,80}check.{0,80}failed|"
        r"pre-commit.*failed|hookid:|checkstyle.{0,80}violations|spotless check failed|"
        r"isort.{0,80}check.{0,80}failed|rubocop.{0,80}(offense|failed)", c)
    config_auth_permission = _bool(
        r"permission denied|forbidden|unauthorized|authentication failed|bad credentials|"
        r"could not read username|remote: invalid username or password|"
        r"403 forbidden|401 unauthorized", c)
    config_missing_file = _bool(
        r"no such file or directory|file not found|could not find file|"
        r"cannot find path|path does not exist|missing file|directory nonexistent", c)
    config_tool_version = _bool(
        r"unsupported node version|unsupported python version|requires python [><=]|"
        r"node version .{0,80} not supported|java version .{0,80} not supported|"
        r"unsupported engine|npm err! engine|the engine .{0,80} is incompatible", c)
    config_dep_resolution_strong = _bool(
        r"could not resolve dependencies|could not resolve all files for configuration|"
        r"could not find a version that satisfies|no matching distribution found for|"
        r"could not find artifact|npm err! eresolve|npm err! could not resolve dependency|"
        r"npm err! 404|npm err! etarget|npm err! notarget|npm err! eunsupportedprotocol|"
        r"bundler::gemnotfound|go: module .{0,120} not found|dependency convergence error|"
        r"version conflict|non-resolvable parent pom|"
        r"composer.{0,80}(could not find|conflict)", c)

    kw_dep_resolution = config_dep_resolution_strong
    kw_config = int(any([config_secret_env, config_yaml_workflow, config_lint_format,
                          config_auth_permission, config_missing_file,
                          config_tool_version, config_dep_resolution_strong]))

    # ── Step name features ────────────────────────────────────────────────────
    step_test    = int(bool(_STEP_TEST.search(failing_step)))
    step_compile = int(bool(_STEP_COMPILE.search(failing_step)))
    step_lint    = int(bool(_STEP_LINT.search(failing_step)))
    step_docker  = int(bool(_STEP_DOCKER.search(failing_step)))
    step_setup   = int(bool(_STEP_SETUP.search(failing_step)))

    # ── Stage / order features ────────────────────────────────────────────────
    first_error_pos = _first_pos(_ERROR_ANCHOR, c)
    first_test_pos  = _first_pos(_TEST_START, c)
    first_error_before_tests         = int(_before(first_error_pos, first_test_pos))
    first_error_after_tests_started  = int(
        first_error_pos is not None
        and first_test_pos is not None
        and first_error_pos > first_test_pos)

    has_compile_task = _bool(
        r"\b(javac|kotlinc|tsc|rustc|gcc|g\+\+|clang|msbuild|cmake|"
        r"maven-compiler-plugin|compilejava|compilekotlin|cargo build|"
        r"go build|npm run build)\b", c)
    has_dependency_install = _bool(
        r"\b(pip install|npm install|npm ci|yarn install|pnpm install|"
        r"bundle install|composer install|mvn dependency|gradle dependencies|"
        r"go mod download|cargo fetch)\b", c)

    error_in_compile_step = int(step_compile or (kw_compile and has_compile_task))
    error_in_setup_step   = int(step_setup or (has_dependency_install and kw_config and not kw_compile))
    error_in_lint_step    = int(step_lint or config_lint_format)
    error_in_test_step    = int(step_test or first_error_after_tests_started)

    # ── Boundary helper features ──────────────────────────────────────────────
    compile_no_tests   = int(kw_compile == 1 and tests_ran == 0)
    compile_with_config = int(kw_compile == 1 and kw_config == 1)
    config_no_compile  = int(kw_config == 1 and kw_compile == 0)

    strong_compile_count = sum([compile_jvm_strong, compile_ts_strong, compile_rust_go_strong,
                                 compile_native_linker, compile_python_syntax, compile_build_task])
    strong_config_count  = sum([config_secret_env, config_yaml_workflow, config_lint_format,
                                 config_auth_permission, config_tool_version, config_dep_resolution_strong])

    compile_over_config_guard       = int(strong_compile_count > 0 and not (step_setup or has_dependency_install))
    config_strong_only              = int(strong_config_count > 0 and strong_compile_count == 0)
    dep_resolution_without_network  = int(config_dep_resolution_strong == 1 and kw_infra_network == 0)

    # ── Ontology one-hots ─────────────────────────────────────────────────────
    p_compile, p_test, p_flaky, p_infra, p_config, p_runtime = primary_to_onehot(primary)

    return [
        float(tests_ran), float(tests_failed_flag), float(num_failed),
        float(num_run), float(num_skip), float(fail_ratio), float(build_dur),
        float(fw_junit), float(fw_pytest), float(fw_gradle), float(fw_vitest),
        float(is_java),
        float(kw_compile), float(kw_assert), float(kw_flaky),
        float(kw_infra_network), float(kw_infra_runner),
        float(kw_dep_resolution), float(kw_config),
        float(compile_no_tests), float(compile_with_config), float(config_no_compile),
        float(step_test), float(step_compile), float(step_lint),
        float(step_docker), float(step_setup),
        float(p_compile), float(p_test), float(p_flaky),
        float(p_infra), float(p_config), float(p_runtime),
        float(compile_jvm_strong), float(compile_ts_strong),
        float(compile_rust_go_strong), float(compile_native_linker),
        float(compile_python_syntax), float(compile_build_task),
        float(config_secret_env), float(config_yaml_workflow),
        float(config_lint_format), float(config_auth_permission),
        float(config_missing_file), float(config_tool_version),
        float(config_dep_resolution_strong),
        float(first_error_before_tests), float(first_error_after_tests_started),
        float(error_in_compile_step), float(error_in_setup_step),
        float(error_in_lint_step), float(error_in_test_step),
        float(has_compile_task), float(has_dependency_install),
        float(compile_over_config_guard), float(config_strong_only),
        float(dep_resolution_without_network),
    ]


# ── Guardrails ────────────────────────────────────────────────────────────────

def _idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


_COMPILE_GUARD_IDXS = [
    _idx("feat_kw_compile_fail"),
    _idx("feat_compile_jvm_strong"),
    _idx("feat_compile_ts_strong"),
    _idx("feat_compile_rust_go_strong"),
    _idx("feat_compile_native_linker"),
    _idx("feat_compile_python_syntax"),
    _idx("feat_compile_build_task"),
    _idx("feat_error_in_compile_step"),
    _idx("feat_compile_over_config_guard"),
]
_TRUE_CONFIG_GUARD_IDXS = [
    _idx("feat_config_secret_env"),
    _idx("feat_config_yaml_workflow"),
    _idx("feat_config_lint_format"),
    _idx("feat_config_auth_permission"),
    _idx("feat_config_tool_version"),
    _idx("feat_dep_resolution_without_network"),
    _idx("feat_error_in_setup_step"),
]
_TEST_GUARD_IDXS = [
    _idx("feat_tests_ran"),
    _idx("feat_tests_failed"),
    _idx("feat_kw_test_assert"),
    _idx("feat_error_in_test_step"),
]


@dataclass(frozen=True)
class GuardrailParams:
    config_to_compile_margin: float = CONFIG_TO_COMPILE_MARGIN
    config_to_test_margin: float = CONFIG_TO_TEST_MARGIN
    min_alt_proba_for_guardrail: float = MIN_ALT_PROBA_FOR_GUARDRAIL


def predict_with_guardrails(
    model,
    X: np.ndarray,
    le: LabelEncoder,
    params: GuardrailParams | None = None,
) -> np.ndarray:
    """Evidence-gated guardrail — now parameterized for validation tuning."""
    params = params or GuardrailParams()
    raw_pred = model.predict(X).copy()
    if not hasattr(model, "predict_proba"):
        return raw_pred

    proba = model.predict_proba(X)
    class_to_idx = {name: i for i, name in enumerate(le.classes_)}

    cfg_enc  = le.transform(["configuration"])[0]
    comp_enc = le.transform(["compilation"])[0]
    test_enc = le.transform(["test_failure"])[0]

    cfg_pidx  = class_to_idx["configuration"]
    comp_pidx = class_to_idx["compilation"]
    test_pidx = class_to_idx["test_failure"]

    guarded = raw_pred.copy()

    for i in range(len(raw_pred)):
        if raw_pred[i] != cfg_enc:
            continue

        p_cfg  = proba[i, cfg_pidx]
        p_comp = proba[i, comp_pidx]
        p_test = proba[i, test_pidx]

        strong_compile = X[i, _COMPILE_GUARD_IDXS].sum() > 0
        true_config    = X[i, _TRUE_CONFIG_GUARD_IDXS].sum() > 0
        test_signal    = X[i, _TEST_GUARD_IDXS].sum() > 0

        if (strong_compile and not true_config
                and p_comp >= params.min_alt_proba_for_guardrail
                and (p_cfg - p_comp) <= params.config_to_compile_margin):
            guarded[i] = comp_enc
            continue

        if (test_signal and not true_config
                and p_test >= params.min_alt_proba_for_guardrail
                and (p_cfg - p_test) <= params.config_to_test_margin):
            guarded[i] = test_enc
            continue

    return guarded


def _proba_for_label(model, X: np.ndarray, le: LabelEncoder, label: str) -> np.ndarray:
    """Return probability for a label, safely handling estimators without predict_proba."""
    if not hasattr(model, "predict_proba"):
        return np.zeros(len(X), dtype=np.float32)
    label_idx = {name: i for i, name in enumerate(le.classes_)}[label]
    return model.predict_proba(X)[:, label_idx]


def predict_with_dual_models(
    rf_model,
    gb_model,
    X: np.ndarray,
    le: LabelEncoder,
    rf_params: GuardrailParams | None = None,
    gb_params: GuardrailParams | None = None,
    strategy: str = DUAL_DEFAULT_STRATEGY,
    return_details: bool = False,
):
    """Use RF and GB together for inference.

    Strategies:
      - "gb_clean": GB + guardrails only. Best default when false positives are costly.
      - "rf_high_recall": RF + guardrails only. Best when missing compilation failures is costly.
      - "agreement_or_gb": use agreed label; otherwise fall back to GB.
      - "hybrid_compile_rescue": start from GB, but let RF rescue likely compilation cases
        when RF predicts compilation with enough probability and compile evidence.

    Returns encoded labels. If return_details=True, returns (labels, details_dict).
    """
    rf_params = rf_params or GuardrailParams()
    gb_params = gb_params or GuardrailParams()

    rf_pred = predict_with_guardrails(rf_model, X, le, rf_params)
    gb_pred = predict_with_guardrails(gb_model, X, le, gb_params)

    if strategy == "gb_clean":
        final = gb_pred.copy()
    elif strategy == "rf_high_recall":
        final = rf_pred.copy()
    elif strategy == "agreement_or_gb":
        final = np.where(rf_pred == gb_pred, rf_pred, gb_pred).copy()
    elif strategy == "hybrid_compile_rescue":
        final = gb_pred.copy()
        comp_enc = le.transform(["compilation"])[0]
        boundary_encs = set(le.transform(list(DUAL_COMPILE_BOUNDARY_CLASSES)))
        rf_p_comp = _proba_for_label(rf_model, X, le, "compilation")
        strong_compile_evidence = X[:, _COMPILE_GUARD_IDXS].sum(axis=1) > 0
        rescue_mask = (
            (rf_pred == comp_enc)
            & (gb_pred != comp_enc)
            & np.isin(gb_pred, list(boundary_encs))
            & (rf_p_comp >= DUAL_RF_COMPILE_MIN_PROBA)
        )
        if DUAL_REQUIRE_COMPILE_EVIDENCE:
            rescue_mask = rescue_mask & strong_compile_evidence
        final[rescue_mask] = comp_enc
    else:
        raise ValueError(
            f"Unknown dual-model strategy: {strategy!r}. "
            "Use gb_clean, rf_high_recall, agreement_or_gb, or hybrid_compile_rescue."
        )

    if not return_details:
        return final

    details = {
        "strategy": strategy,
        "disagreements": int((rf_pred != gb_pred).sum()),
        "used_rf_predictions": int((final == rf_pred).sum()),
        "used_gb_predictions": int((final == gb_pred).sum()),
    }
    if strategy == "hybrid_compile_rescue":
        details["compile_rescues"] = int((final != gb_pred).sum())
        details["rf_compile_min_proba"] = DUAL_RF_COMPILE_MIN_PROBA
        details["require_compile_evidence"] = DUAL_REQUIRE_COMPILE_EVIDENCE
    return final, details


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str, f=None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if f:
        f.write(line + "\n")
        f.flush()


# ── Robust data loader ────────────────────────────────────────────────────────

LABEL_KEYS = ("label", "final_label", "training_label", "target_label",
              "bucket_label", "bucket", "gold_label", "true_label",
              "actual", "ollama_label", "rule_label")

TEXT_KEYS = ("text", "snippet", "log_snippet", "failure_snippet",
             "clean_text", "failure_window", "body", "log")

LABEL_ALIASES = {
    "compile": "compilation", "build_compile": "compilation", "build": "compilation",
    "test": "test_failure", "tests": "test_failure", "testfail": "test_failure",
    "test_failure": "test_failure", "test-failure": "test_failure",
    "flaky": "flaky_test", "flakytest": "flaky_test",
    "flaky_test": "flaky_test", "flaky-test": "flaky_test",
    "infra": "infrastructure", "infrastructure": "infrastructure",
    "config": "configuration", "configuration": "configuration",
}


def _canonical_label(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_")
    if not v or v in {"none", "null", "unknown"}:
        return None
    return LABEL_ALIASES.get(v, v if v in TARGET_LABELS else None)


def _row_label(row: dict) -> str | None:
    for key in LABEL_KEYS:
        label = _canonical_label(row.get(key))
        if label in TARGET_LABELS:
            return label
    return None


def _row_text(row: dict) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def resolve_dataset_path() -> Path:
    """Find training JSONL. Override with: set M13_DATASET=path\\to\\file.jsonl"""
    candidates = []
    env_path = os.getenv("M13_DATASET")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        GITHUB_FILE,
        Path("github_actions_training_dataset.jsonl"),
        Path("data/final/github_actions_training_dataset.jsonl"),
        Path("Ollama_DataGathering/data/final/github_actions_training_dataset.jsonl"),
        Path(__file__).resolve().parent / "github_actions_training_dataset.jsonl",
        Path(__file__).resolve().parent / "Ollama_DataGathering/data/final/github_actions_training_dataset.jsonl",
    ])
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    checked = "\n    ".join(str(p) for p in candidates)
    raise FileNotFoundError("Could not find dataset. Checked:\n    " + checked)


def load_data(report_f=None, keep_rows: bool = False):
    data_path = resolve_dataset_path()
    log(f"Loading data from {data_path}...", report_f)

    rows_X, rows_y, rows_w, kept_rows = [], [], [], []
    skipped_label = skipped_text = skipped_feat = 0
    synthetic_count = real_count = 0
    invalid_label_values: Counter = Counter()
    first_row_keys = None

    with data_path.open("r", encoding="utf-8-sig") as f:
        lines = [line for line in f if line.strip()]

    for line in tqdm(lines, desc="  Extracting features", unit="row"):
        try:
            row = json.loads(line)
            if first_row_keys is None:
                first_row_keys = sorted(row.keys())[:20]

            label = _row_label(row)
            if label not in TARGET_LABELS:
                skipped_label += 1
                for key in LABEL_KEYS:
                    if key in row:
                        invalid_label_values[f"{key}={row.get(key)}"] += 1
                        break
                continue

            text = _row_text(row)
            if not text.strip():
                skipped_text += 1
                continue

            row = dict(row)
            row["label"] = label
            row["text"]  = text

            feats = extract_features(row)
            if len(feats) != len(FEATURE_NAMES):
                raise ValueError(f"feature length mismatch: {len(feats)} != {len(FEATURE_NAMES)}")

            rows_X.append(feats)
            rows_y.append(label)
            if keep_rows:
                kept_rows.append(row)

            source     = str(row.get("label_source", "")).lower()
            confidence = float(row.get("confidence") or row.get("rule_confidence")
                               or row.get("ollama_confidence") or 0.8)
            if source == "synthetic" or row.get("synthetic") is True:
                weight = SYNTHETIC_WEIGHT
                synthetic_count += 1
            else:
                weight = max(0.6, min(1.0, confidence))
                real_count += 1
            rows_w.append(weight)

        except Exception as exc:
            skipped_feat += 1
            if skipped_feat <= 3:
                log(f"  [skip] {exc}", report_f)
            continue

    y_arr = np.array(rows_y)
    log(f"  Total file rows  : {len(lines):,}", report_f)
    log(f"  Loaded           : {len(rows_X):,}", report_f)
    log(f"  Real rows        : {real_count:,}", report_f)
    log(f"  Synthetic        : {synthetic_count:,}  (weight={SYNTHETIC_WEIGHT})", report_f)
    log(f"  Skipped label    : {skipped_label:,}", report_f)
    log(f"  Skipped text     : {skipped_text:,}", report_f)
    log(f"  Skipped feat     : {skipped_feat:,}", report_f)

    if len(rows_X) == 0:
        raise RuntimeError(
            f"Loaded 0 rows.\n"
            f"Dataset: {data_path}\n"
            f"First row keys: {first_row_keys}\n"
            f"Top invalid labels: {invalid_label_values.most_common(8)}\n"
            f"Fix: set M13_DATASET env var to correct JSONL path.")

    log("  Label distribution:", report_f)
    for label in TARGET_LABELS:
        count = int((y_arr == label).sum())
        log(f"    {label:<22} {count:>6,}  ({count/len(y_arr)*100:5.1f}%)", report_f)

    out = (np.array(rows_X, dtype=np.float32), y_arr, np.array(rows_w, dtype=np.float32))
    return (*out, kept_rows) if keep_rows else out


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate_predictions(y_test_enc, y_pred, le, report_f, name: str) -> dict[str, float]:
    acc      = accuracy_score(y_test_enc, y_pred)
    f1_macro = f1_score(y_test_enc, y_pred, average="macro",    zero_division=0)
    f1_wt    = f1_score(y_test_enc, y_pred, average="weighted", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_enc, y_pred, labels=range(len(le.classes_)), zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
    }
    for i, label in enumerate(le.classes_):
        metrics[f"{label}_precision"] = float(precision[i])
        metrics[f"{label}_recall"] = float(recall[i])
        metrics[f"{label}_f1"] = float(f1[i])

    log(f"\n{name}", report_f)
    log(f"  Accuracy    : {acc:.4f}", report_f)
    log(f"  F1 macro    : {f1_macro:.4f}", report_f)
    log(f"  F1 weighted : {f1_wt:.4f}", report_f)
    log("\nPer-class report:", report_f)
    log(classification_report(y_test_enc, y_pred,
                               target_names=le.classes_,
                               zero_division=0, digits=4), report_f)
    log("Confusion matrix (rows=actual, cols=predicted):", report_f)
    cm = confusion_matrix(y_test_enc, y_pred)
    cm_df = pd.DataFrame(cm,
                          index=[f"act:{c}" for c in le.classes_],
                          columns=[f"pred:{c}" for c in le.classes_])
    log(cm_df.to_string(), report_f)
    return metrics


def _evaluate(model, X_test, y_test_enc, le, report_f, name: str,
              guardrail_params: GuardrailParams | None = None) -> dict:
    raw_pred = model.predict(X_test)
    raw_metrics = _evaluate_predictions(y_test_enc, raw_pred, le, report_f, f"{name} — raw")
    guarded_pred = predict_with_guardrails(model, X_test, le, guardrail_params)
    changed = int((guarded_pred != raw_pred).sum())
    log(f"\nGuardrail changed {changed:,} / {len(raw_pred):,} predictions", report_f)
    guarded_metrics = _evaluate_predictions(y_test_enc, guarded_pred, le, report_f, f"{name} — guardrailed")
    return {
        "raw": raw_metrics,
        "guardrailed": guarded_metrics,
        "guardrail_changed": changed,
        "guardrail_params": asdict(guardrail_params or GuardrailParams()),
    }



# ── Upgrade helpers: ablations, model selection, calibration, exports ─────────

def _is_synthetic_row(row: dict) -> bool:
    source = str(row.get("label_source", "")).lower()
    return source == "synthetic" or row.get("synthetic") is True


def _row_confidence(row: dict) -> float:
    confidence = float(row.get("confidence") or row.get("rule_confidence")
                       or row.get("ollama_confidence") or 0.8)
    return max(0.6, min(1.0, confidence))


def make_sample_weights(rows: list[dict], synthetic_weight: float = SYNTHETIC_WEIGHT) -> np.ndarray:
    return np.array([
        synthetic_weight if _is_synthetic_row(row) else _row_confidence(row)
        for row in rows
    ], dtype=np.float32)


def _safe_fit(model, X, y, sample_weight=None):
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        return model.fit(X, y)


def _safe_json_dump(obj, path: str | Path) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, GuardrailParams):
            return asdict(o)
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=default)


def _dataset_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_random_forest(n_estimators: int = 300) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=25,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )


def build_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def metric_dict(y_true_enc, y_pred_enc, le: LabelEncoder, prefix: str = "") -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=range(len(le.classes_)), zero_division=0)
    out = {
        f"{prefix}accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
        f"{prefix}macro_f1": float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)),
    }
    for i, label in enumerate(le.classes_):
        out[f"{prefix}{label}_precision"] = float(precision[i])
        out[f"{prefix}{label}_recall"] = float(recall[i])
        out[f"{prefix}{label}_f1"] = float(f1[i])
    return out


def tune_guardrail_params(model, X_val, y_val_enc, le, report_f, name: str) -> tuple[GuardrailParams, dict[str, float]]:
    best_params = GuardrailParams()
    best_pred = predict_with_guardrails(model, X_val, le, best_params)
    best_metrics = metric_dict(y_val_enc, best_pred, le)
    best_score = best_metrics["macro_f1"]

    for compile_margin in GUARDRAIL_COMPILE_MARGINS:
        for test_margin in GUARDRAIL_TEST_MARGINS:
            for min_alt in GUARDRAIL_MIN_ALT_PROBAS:
                params = GuardrailParams(float(compile_margin), float(test_margin), float(min_alt))
                pred = predict_with_guardrails(model, X_val, le, params)
                metrics = metric_dict(y_val_enc, pred, le)
                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    best_params = params
                    best_metrics = metrics

    log(f"\n{name} — tuned guardrails on validation", report_f)
    log(f"  Params      : {asdict(best_params)}", report_f)
    log(f"  Val accuracy: {best_metrics['accuracy']:.4f}", report_f)
    log(f"  Val macro F1: {best_metrics['macro_f1']:.4f}", report_f)
    return best_params, best_metrics


def _feature_subset(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idxs = [FEATURE_NAMES.index(name) for name in feature_names]
    return X[:, idxs]


def run_ablation_study(X_train, X_test, y_train_enc, y_test_enc, w_train, le, report_f) -> list[dict]:
    if not ENABLE_ABLATION_STUDY:
        return []

    primary_features = [f for f in FEATURE_NAMES if f.startswith("feat_primary_")]
    feature_sets = {
        "full": FEATURE_NAMES,
        "no_primary_label": [f for f in FEATURE_NAMES if f not in primary_features],
        "ontology_only": primary_features,
    }

    log(f"\n{'='*62}", report_f)
    log("ABLATION STUDY — primary_label / ontology dependence", report_f)
    log(f"{'='*62}", report_f)

    results = []
    for name, features in feature_sets.items():
        model = build_random_forest(n_estimators=SIDE_STUDY_RF_TREES)
        Xtr = _feature_subset(X_train, features)
        Xte = _feature_subset(X_test, features)
        _safe_fit(model, Xtr, y_train_enc, sample_weight=w_train)
        pred = model.predict(Xte)
        metrics = metric_dict(y_test_enc, pred, le)
        row = {"ablation": name, "num_features": len(features), **metrics}
        results.append(row)
        log(f"  {name:<18} features={len(features):>2}  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}", report_f)

    pd.DataFrame(results).to_csv(ABLATION_RESULTS_OUT, index=False)
    log(f"Saved ablation results: {ABLATION_RESULTS_OUT}", report_f)
    return results


def run_synthetic_weight_sweep(X_train, X_test, y_train_enc, y_test_enc, rows_train, le,
                               guardrail_params, report_f) -> list[dict]:
    if not ENABLE_SYNTHETIC_SWEEP:
        return []

    synthetic_mask = np.array([_is_synthetic_row(r) for r in rows_train], dtype=bool)

    log(f"\n{'='*62}", report_f)
    log("SYNTHETIC WEIGHT SWEEP", report_f)
    log(f"{'='*62}", report_f)

    results = []
    for setting in SYNTHETIC_WEIGHT_SWEEP:
        if setting == "real_only":
            mask = ~synthetic_mask
            Xtr = X_train[mask]
            ytr = y_train_enc[mask]
            used_rows = [r for r, keep in zip(rows_train, mask) if keep]
            wtr = make_sample_weights(used_rows, SYNTHETIC_WEIGHT)
            label = "real_only"
            synthetic_rows_used = 0
        else:
            Xtr = X_train
            ytr = y_train_enc
            wtr = make_sample_weights(rows_train, float(setting))
            label = f"synthetic_weight_{float(setting):.2f}"
            synthetic_rows_used = int(synthetic_mask.sum())

        model = build_random_forest(n_estimators=SIDE_STUDY_RF_TREES)
        _safe_fit(model, Xtr, ytr, sample_weight=wtr)
        raw_pred = model.predict(X_test)
        guarded_pred = predict_with_guardrails(model, X_test, le, guardrail_params)
        row = {
            "setting": label,
            "train_rows": int(len(Xtr)),
            "synthetic_rows_used": synthetic_rows_used,
            **metric_dict(y_test_enc, raw_pred, le, prefix="raw_"),
            **metric_dict(y_test_enc, guarded_pred, le, prefix="guardrailed_"),
        }
        results.append(row)
        log(f"  {label:<24} rows={len(Xtr):>6,}  raw_macro_f1={row['raw_macro_f1']:.4f}  guarded_macro_f1={row['guardrailed_macro_f1']:.4f}", report_f)

    pd.DataFrame(results).to_csv(SYNTHETIC_SWEEP_OUT, index=False)
    log(f"Saved synthetic sweep: {SYNTHETIC_SWEEP_OUT}", report_f)
    return results


def _multiclass_brier_score(y_true_enc: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_onehot = np.zeros((len(y_true_enc), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true_enc)), y_true_enc] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def calibration_report(model, X_test, y_test_enc, le, report_f, name: str) -> dict:
    if not hasattr(model, "predict_proba"):
        return {}
    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    out = {
        "log_loss": float(log_loss(y_test_enc, proba, labels=list(range(len(le.classes_))))),
        "multiclass_brier": _multiclass_brier_score(y_test_enc, proba, len(le.classes_)),
    }
    for i, label in enumerate(le.classes_):
        out[f"brier_{label}"] = float(brier_score_loss((y_test_enc == i).astype(int), proba[:, i]))

    conf = proba.max(axis=1)
    bucket_df = pd.DataFrame({"confidence": conf, "correct": (pred == y_test_enc).astype(int)})
    bucket_df["bucket"] = pd.cut(bucket_df["confidence"], [0.0, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
    buckets = bucket_df.groupby("bucket", observed=True).agg(
        rows=("correct", "size"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean"),
    ).reset_index()
    log(f"\n{name} — calibration", report_f)
    log(f"  Log loss         : {out['log_loss']:.4f}", report_f)
    log(f"  Multiclass Brier : {out['multiclass_brier']:.4f}", report_f)
    log(buckets.to_string(index=False), report_f)
    return out


def make_calibrated_model(base_model):
    try:
        return CalibratedClassifierCV(estimator=base_model, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


def row_preview(row: dict, max_chars: int = 600) -> str:
    text = clean(str(row.get("text", ""))).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def active_features_for_row(x_row: np.ndarray, limit: int = 20) -> str:
    active = []
    for name, value in zip(FEATURE_NAMES, x_row):
        if abs(float(value)) > 1e-9:
            active.append(f"{name}={float(value):.3g}")
    return "; ".join(active[:limit])


def prediction_dataframe(model, X, y_enc, rows, le, guardrail_params) -> pd.DataFrame:
    raw_pred = model.predict(X)
    guarded_pred = predict_with_guardrails(model, X, le, guardrail_params)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else np.zeros((len(X), len(le.classes_)))
    sorted_proba = np.sort(proba, axis=1)
    margin_top2 = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] >= 2 else sorted_proba[:, -1]
    true_labels = le.inverse_transform(y_enc)
    raw_labels = le.inverse_transform(raw_pred)
    guarded_labels = le.inverse_transform(guarded_pred)

    rows_out = []
    for i, row in enumerate(rows):
        item = {
            "true_label": true_labels[i],
            "raw_pred": raw_labels[i],
            "guarded_pred": guarded_labels[i],
            "correct": bool(guarded_pred[i] == y_enc[i]),
            "confidence": float(proba[i].max()) if len(proba) else 0.0,
            "margin_top2": float(margin_top2[i]) if len(proba) else 0.0,
            "repo": row.get("repo") or row.get("repository") or row.get("project") or "",
            "workflow": row.get("workflow") or row.get("workflow_name") or "",
            "job": row.get("job") or row.get("job_name") or "",
            "failing_step": row.get("failing_step", ""),
            "primary_label": row.get("primary_label", ""),
            "label_source": row.get("label_source", ""),
            "confidence_source": row.get("confidence") or row.get("rule_confidence") or row.get("ollama_confidence") or "",
            "synthetic": _is_synthetic_row(row),
            "active_features": active_features_for_row(X[i]),
            "text_preview": row_preview(row),
        }
        for j, label in enumerate(le.classes_):
            item[f"p_{label}"] = float(proba[i, j])
        rows_out.append(item)
    return pd.DataFrame(rows_out)


def export_error_reports(model, X_test, y_test_enc, rows_test, le, guardrail_params, report_f) -> dict:
    df = prediction_dataframe(model, X_test, y_test_enc, rows_test, le, guardrail_params)
    errors = df[~df["correct"]].copy()
    errors.to_csv(ERRORS_OUT, index=False)

    comp_test = errors[
        ((errors["true_label"] == "test_failure") & (errors["guarded_pred"] == "compilation")) |
        ((errors["true_label"] == "compilation") & (errors["guarded_pred"] == "test_failure"))
    ].copy()
    comp_test.to_csv(ERRORS_COMP_TEST_OUT, index=False)

    config_comp = errors[
        ((errors["true_label"] == "configuration") & (errors["guarded_pred"] == "compilation")) |
        ((errors["true_label"] == "compilation") & (errors["guarded_pred"] == "configuration"))
    ].copy()
    config_comp.to_csv(ERRORS_CONFIG_COMP_OUT, index=False)

    low_conf = df[(df["confidence"] < LOW_CONFIDENCE_THRESHOLD) | (df["margin_top2"] < LOW_MARGIN_THRESHOLD)].copy()
    low_conf.to_csv(LOW_CONFIDENCE_OUT, index=False)

    counts = {
        "all_test_rows": int(len(df)),
        "errors": int(len(errors)),
        "compilation_vs_test_errors": int(len(comp_test)),
        "configuration_vs_compilation_errors": int(len(config_comp)),
        "low_confidence_or_low_margin": int(len(low_conf)),
    }
    log("\nExported error analysis:", report_f)
    for name, count in counts.items():
        log(f"  {name:<34} {count:,}", report_f)
    return counts


def save_confusion_matrix_csv(y_true_enc, y_pred_enc, le) -> None:
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    pd.DataFrame(
        cm,
        index=[f"act:{c}" for c in le.classes_],
        columns=[f"pred:{c}" for c in le.classes_],
    ).to_csv(CONFUSION_MATRIX_OUT)


def evaluate_dual_strategies(rf_model, gb_model, X_test, y_test_enc, le,
                             rf_params, gb_params, report_f) -> dict:
    """Evaluate the supported dual-model inference policies on held-out data."""
    log(f"\n{'='*62}", report_f)
    log("DUAL-MODEL INFERENCE POLICIES", report_f)
    log(f"{'='*62}", report_f)
    results = {}
    for strategy in ["gb_clean", "rf_high_recall", "agreement_or_gb", "hybrid_compile_rescue"]:
        pred, details = predict_with_dual_models(
            rf_model, gb_model, X_test, le, rf_params, gb_params,
            strategy=strategy, return_details=True)
        metrics = metric_dict(y_test_enc, pred, le)
        results[strategy] = {**metrics, "details": details}
        log(
            f"  {strategy:<24} accuracy={metrics['accuracy']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  "
            f"comp_precision={metrics['compilation_precision']:.4f}  "
            f"comp_recall={metrics['compilation_recall']:.4f}  "
            f"disagreements={details.get('disagreements', 0):,}  "
            f"rescues={details.get('compile_rescues', 0):,}"
        )
    return results


def export_feature_importance(model, report_f) -> list[dict]:
    if not hasattr(model, "feature_importances_"):
        log("\nSelected model has no feature_importances_; skipping feature importance export.", report_f)
        return []
    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    df.to_csv(FEATURE_IMPORTANCE_OUT, index=False)
    log("\nTop 20 feature importances (selected base model):", report_f)
    for _, row in df.head(20).iterrows():
        feat = row["feature"]
        imp = row["importance"]
        tag = (" [KEYWORD]" if "kw" in feat
               else " [STAGE]" if any(x in feat for x in ["step", "error_in", "first_error", "has_"])
               else " [ONTOLOGY]" if "primary" in feat
               else " [COMPILE]" if "compile_" in feat
               else " [CONFIG]" if "config_" in feat or "dep_resolution" in feat
               else "")
        log(f"  {feat:<38} {imp:.4f}  {'█' * int(imp * 300)}{tag}", report_f)
    log(f"Saved feature importances: {FEATURE_IMPORTANCE_OUT}", report_f)
    return df.head(20).to_dict(orient="records")

# ── Training ──────────────────────────────────────────────────────────────────

def train(report_f) -> None:
    data_path = resolve_dataset_path()
    X, y, w, rows = load_data(report_f, keep_rows=True)

    log(f"\nSplitting: 80% train / 20% held-out test (stratified)...", report_f)
    X_train, X_test, y_train, y_test, w_train, w_test, rows_train, rows_test = train_test_split(
        X, y, w, rows, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    log(f"  Train : {len(X_train):,}  Test : {len(X_test):,}", report_f)
    log("  Test label distribution:", report_f)
    for label in TARGET_LABELS:
        log(f"    {label:<22} {int((y_test == label).sum()):>5,}", report_f)

    le = LabelEncoder()
    le.fit(TARGET_LABELS)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    X_fit, X_val, y_fit_enc, y_val_enc, w_fit, w_val, rows_fit, rows_val = train_test_split(
        X_train, y_train_enc, w_train, rows_train,
        test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_train_enc)
    log(f"\nValidation split for model selection / guardrail tuning:", report_f)
    log(f"  Fit : {len(X_fit):,}  Val : {len(X_val):,}", report_f)

    log(f"\nFeatures : {len(FEATURE_NAMES)}", report_f)
    log("  Test counts + timing + frameworks : 13", report_f)
    log("  Original keyword + disambiguation : 10", report_f)
    log("  Step name                         : 5", report_f)
    log("  Ontology one-hots (incl runtime)  : 6", report_f)
    log("  Strong compile sub-features       : 6", report_f)
    log("  Split config sub-features         : 7", report_f)
    log("  Stage / order                     : 8", report_f)
    log("  Boundary helpers                  : 3", report_f)

    metrics_bundle = {
        "dataset_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "validation_rows": int(len(X_val)),
        "target_labels": list(le.classes_),
    }

    # 1) Ablation: does the model still work without primary_label?
    metrics_bundle["ablation"] = run_ablation_study(
        X_train, X_test, y_train_enc, y_test_enc, w_train, le, report_f)

    # 3 + 7) Compare RF/GB and tune guardrails on validation data.
    log(f"\n{'='*62}", report_f)
    log("MODEL SELECTION + GUARDRAIL TUNING", report_f)
    log(f"{'='*62}", report_f)

    log(f"\nRandom Forest 5-fold CV on train set ({len(X_train):,} rows)...", report_f)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        build_random_forest(300), X_train, y_train_enc, cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        n_jobs=1, return_train_score=True)
    cv_summary = {}
    log("\nRandom Forest — Cross-validation:", report_f)
    for metric in ["test_accuracy", "test_f1_macro", "test_f1_weighted",
                   "train_accuracy", "train_f1_macro"]:
        scores = cv_results[metric]
        cv_summary[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }
        log(f"  {metric:<25} {scores.mean():.4f} ± {scores.std():.4f}  "
            f"[{scores.min():.4f} – {scores.max():.4f}]", report_f)

    candidates = {
        "random_forest": build_random_forest(300),
        "gradient_boosting": build_gradient_boosting(),
    }
    validation_summary = {"random_forest_cv": cv_summary}
    tuned_params = {}

    for name, base in candidates.items():
        log(f"\nTraining {name} on fit split for validation selection...", report_f)
        model = clone(base)
        _safe_fit(model, X_fit, y_fit_enc, sample_weight=w_fit)
        raw_pred = model.predict(X_val)
        raw_metrics = metric_dict(y_val_enc, raw_pred, le)
        params, guarded_metrics = tune_guardrail_params(model, X_val, y_val_enc, le, report_f, name)
        guarded_pred = predict_with_guardrails(model, X_val, le, params)
        validation_summary[name] = {
            "raw": raw_metrics,
            "guardrailed": guarded_metrics,
            "guardrail_params": asdict(params),
            "guardrail_changed": int((guarded_pred != raw_pred).sum()),
        }
        tuned_params[name] = params
        log(f"  {name:<18} raw_macro_f1={raw_metrics['macro_f1']:.4f}  "
            f"guarded_macro_f1={guarded_metrics['macro_f1']:.4f}", report_f)

    def selection_score(model_name: str) -> float:
        if PRIMARY_SELECTION_METRIC == "accuracy_guardrailed":
            return validation_summary[model_name]["guardrailed"]["accuracy"]
        if PRIMARY_SELECTION_METRIC == "compilation_precision_guardrailed":
            return validation_summary[model_name]["guardrailed"]["compilation_precision"]
        return validation_summary[model_name]["guardrailed"]["macro_f1"]

    selected_name = max(candidates, key=selection_score)
    selected_guardrail_params = tuned_params[selected_name]
    log(f"\nSelected primary model by {PRIMARY_SELECTION_METRIC}: {selected_name}", report_f)
    log(f"Selected guardrail params: {asdict(selected_guardrail_params)}", report_f)

    # 9) Synthetic data impact study.
    metrics_bundle["synthetic_weight_sweep"] = run_synthetic_weight_sweep(
        X_train, X_test, y_train_enc, y_test_enc, rows_train, le,
        tuned_params.get("random_forest", GuardrailParams()), report_f)

    # Fit final models on full train and evaluate on held-out test.
    log(f"\n{'='*62}", report_f)
    log(f"HELD-OUT TEST SET EVALUATION ({len(X_test):,} rows)", report_f)
    log(f"{'='*62}", report_f)

    final_models = {}
    heldout_results = {}
    for name, base in candidates.items():
        log(f"\nFitting final {name} on full train set...", report_f)
        model = clone(base)
        _safe_fit(model, X_train, y_train_enc, sample_weight=w_train)
        final_models[name] = model
        heldout_results[name] = _evaluate(model, X_test, y_test_enc, le, report_f, name, tuned_params[name])

    # Keep both final models. GB is the default clean-label model; RF is the high-recall compilation option.
    default_model_name = "gradient_boosting"
    default_model = final_models[default_model_name]
    default_guardrail_params = tuned_params[default_model_name]
    high_recall_model_name = "random_forest"
    high_recall_model = final_models[high_recall_model_name]
    high_recall_guardrail_params = tuned_params[high_recall_model_name]

    log(f"\nDefault clean-label model: {default_model_name} (saved as {MODEL_OUT} and {MODEL_GB_OUT})", report_f)
    log(f"High-recall compilation model: {high_recall_model_name} (saved as {MODEL_RF_OUT})", report_f)

    # Evaluate ways to use RF + GB together.
    metrics_bundle["dual_model_strategies"] = evaluate_dual_strategies(
        high_recall_model, default_model, X_test, y_test_enc, le,
        high_recall_guardrail_params, default_guardrail_params, report_f)

    calibration_results = {}
    calibrated_rf = None

    # 8) Probability calibration for RF is saved separately, never as the default classifier.
    if ENABLE_CALIBRATION and SAVE_CALIBRATED_RF:
        log(f"\n{'='*62}", report_f)
        log("CALIBRATION — random_forest saved separately", report_f)
        log(f"{'='*62}", report_f)
        calibrated_rf = make_calibrated_model(clone(candidates["random_forest"]))
        _safe_fit(calibrated_rf, X_train, y_train_enc, sample_weight=w_train)
        calibration_results["random_forest_uncalibrated"] = calibration_report(
            high_recall_model, X_test, y_test_enc, le, report_f, "random_forest uncalibrated")
        calibration_results["random_forest_calibrated"] = calibration_report(
            calibrated_rf, X_test, y_test_enc, le, report_f, "random_forest calibrated")
        heldout_results["random_forest_calibrated"] = _evaluate(
            calibrated_rf, X_test, y_test_enc, le, report_f,
            "random_forest calibrated", high_recall_guardrail_params)
        final_models["random_forest_calibrated"] = calibrated_rf
        log(f"Calibrated RF will be saved as {MODEL_RF_CALIBRATED_OUT}, not as the default model.", report_f)

    # 6) Export error/boundary examples for the default clean-label GB model.
    error_counts = export_error_reports(
        default_model, X_test, y_test_enc, rows_test, le, default_guardrail_params, report_f)

    selected_pred = predict_with_guardrails(default_model, X_test, le, default_guardrail_params)
    save_confusion_matrix_csv(y_test_enc, selected_pred, le)
    log(f"Saved confusion matrix: {CONFUSION_MATRIX_OUT}", report_f)

    feature_top20 = export_feature_importance(default_model, report_f)

    metrics_bundle.update({
        "validation": validation_summary,
        "heldout": heldout_results,
        "calibration": calibration_results,
        "validation_selected_model": selected_name,
        "validation_selected_guardrail_params": asdict(selected_guardrail_params),
        "default_model": default_model_name,
        "default_model_policy": DEFAULT_MODEL_POLICY,
        "default_guardrail_params": asdict(default_guardrail_params),
        "high_recall_model": high_recall_model_name,
        "high_recall_guardrail_params": asdict(high_recall_guardrail_params),
        "error_exports": error_counts,
        "feature_importances_top20": feature_top20,
    })

    metadata = {
        "model_name": "M13 Final Classifier — dual-model upgraded",
        "default_model": default_model_name,
        "default_model_policy": DEFAULT_MODEL_POLICY,
        "default_model_file": MODEL_OUT,
        "gb_model_file": MODEL_GB_OUT,
        "rf_model_file": MODEL_RF_OUT,
        "rf_calibrated_model_file": MODEL_RF_CALIBRATED_OUT if calibrated_rf is not None else None,
        "bundle_file": MODEL_BUNDLE_OUT,
        "high_recall_model": high_recall_model_name,
        "validation_selected_model": selected_name,
        "selected_by": PRIMARY_SELECTION_METRIC,
        "target_labels": TARGET_LABELS,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "synthetic_weight_default": SYNTHETIC_WEIGHT,
        "default_guardrail_params": asdict(default_guardrail_params),
        "high_recall_guardrail_params": asdict(high_recall_guardrail_params),
        "validation_selected_guardrail_params": asdict(selected_guardrail_params),
        "dual_default_strategy": DUAL_DEFAULT_STRATEGY,
        "dual_rf_compile_min_proba": DUAL_RF_COMPILE_MIN_PROBA,
        "dual_require_compile_evidence": DUAL_REQUIRE_COMPILE_EVIDENCE,
        "calibration_enabled": ENABLE_CALIBRATION,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_saved_as_default": False,
        "dataset_path": str(data_path),
        "dataset_sha256": _dataset_hash(data_path),
        "created_at": datetime.now().isoformat(),
        "outputs": {
            "model_default_alias_clean_gb": MODEL_OUT,
            "model_gb": MODEL_GB_OUT,
            "model_rf": MODEL_RF_OUT,
            "model_rf_calibrated": MODEL_RF_CALIBRATED_OUT if calibrated_rf is not None else None,
            "model_bundle": MODEL_BUNDLE_OUT,
            "label_encoder": ENCODER_OUT,
            "feature_names": FEATURES_OUT,
            "report": REPORT_OUT,
            "metrics": METRICS_OUT,
            "metadata": METADATA_OUT,
            "feature_importance": FEATURE_IMPORTANCE_OUT,
            "confusion_matrix": CONFUSION_MATRIX_OUT,
            "ablation": ABLATION_RESULTS_OUT,
            "synthetic_sweep": SYNTHETIC_SWEEP_OUT,
            "errors": ERRORS_OUT,
            "errors_compilation_vs_test": ERRORS_COMP_TEST_OUT,
            "errors_configuration_vs_compilation": ERRORS_CONFIG_COMP_OUT,
            "low_confidence": LOW_CONFIDENCE_OUT,
        },
    }

    # Save default alias + separate models.
    joblib.dump(default_model, MODEL_OUT)       # backwards-compatible alias: clean GB
    joblib.dump(default_model, MODEL_GB_OUT)
    joblib.dump(high_recall_model, MODEL_RF_OUT)
    if calibrated_rf is not None:
        joblib.dump(calibrated_rf, MODEL_RF_CALIBRATED_OUT)

    bundle = {
        "models": {
            "gradient_boosting_clean": default_model,
            "random_forest_high_recall": high_recall_model,
            "random_forest_calibrated": calibrated_rf,
        },
        "label_encoder": le,
        "feature_names": FEATURE_NAMES,
        "guardrail_params": {
            "gradient_boosting": default_guardrail_params,
            "random_forest": high_recall_guardrail_params,
        },
        "default_policy": DEFAULT_MODEL_POLICY,
        "dual_default_strategy": DUAL_DEFAULT_STRATEGY,
        "metadata": metadata,
    }
    joblib.dump(bundle, MODEL_BUNDLE_OUT)
    joblib.dump(le, ENCODER_OUT)
    joblib.dump(FEATURE_NAMES, FEATURES_OUT)
    _safe_json_dump(metrics_bundle, METRICS_OUT)
    _safe_json_dump(metadata, METADATA_OUT)

    log(f"\nSaved default clean model alias: {MODEL_OUT}", report_f)
    log(f"Saved separate models: {MODEL_GB_OUT}, {MODEL_RF_OUT}" + (f", {MODEL_RF_CALIBRATED_OUT}" if calibrated_rf is not None else ""), report_f)
    log(f"Saved dual-model bundle: {MODEL_BUNDLE_OUT}", report_f)
    log(f"Saved encoder/features: {ENCODER_OUT}, {FEATURES_OUT}", report_f)
    log(f"Saved metadata: {METADATA_OUT}", report_f)
    log(f"Saved metrics: {METRICS_OUT}", report_f)
    log("Inference options: GB clean labels, RF high-recall compilation, or predict_with_dual_models(..., strategy='hybrid_compile_rescue').", report_f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  M13 Final Classifier — dual-model upgraded")
    print(f"  {len(FEATURE_NAMES)} features | 80/20 split | RF + GB + tuned guardrails + dual inference")
    print(f"  Synthetic weight: {SYNTHETIC_WEIGHT}")
    print("=" * 62)

    with open(REPORT_OUT, "w", encoding="utf-8") as report_f:
        log("M13 Training Report — Dual-Model Upgraded", report_f)
        log(f"Started  : {datetime.now().isoformat()}", report_f)
        log(f"Features : {len(FEATURE_NAMES)}", report_f)
        log("=" * 62, report_f)
        train(report_f)
        log(f"Finished : {datetime.now().isoformat()}", report_f)

    print(f"\nDone. Report saved to {REPORT_OUT}")


if __name__ == "__main__":
    main()
