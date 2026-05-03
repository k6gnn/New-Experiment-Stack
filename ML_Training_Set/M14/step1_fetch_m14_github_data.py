"""
Step 1: Fetch GitHub Actions run history for M14 v2
====================================================
Builds a GitHub Actions dataset for a proactive failure predictor.

Input:
  - repos.json: list of repos, either [{"owner":"apache","repo":"commons-lang"}, ...]
                or ["apache/commons-lang", ...]
  - .env with GITHUB_TOKEN=...

Output:
  - data/m14_github_runs.csv
  - data/m14_fetch_report.txt
  - data/cache/runs/*.json
  - data/cache/commits/*.json

Run:
  python step1_fetch_m14_github_data.py

Notes:
  This script only uses information available before a pipeline run starts:
  previous run history + commit/change metadata for the current SHA.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import requests

# ── Config ────────────────────────────────────────────────────────────────────
REPOS_FILE = Path("repos.json")
DATA_DIR = Path("data")
RUNS_OUT = DATA_DIR / "m14_github_runs.csv"
REPORT_OUT = DATA_DIR / "m14_fetch_report.txt"
RUN_CACHE_DIR = DATA_DIR / "cache" / "runs"
COMMIT_CACHE_DIR = DATA_DIR / "cache" / "commits"

MAX_RUNS_PER_REPO = 400          # increase to 700/1000 if you want a bigger dataset
PER_PAGE = 100
WORKERS = 12                     # 10-12 is reasonable with token rotation
REQUEST_DELAY_SEC = 0.04         # keep pacing to avoid secondary rate limits
INCLUDE_CONCLUSIONS = {"success", "failure", "timed_out"}

API = "https://api.github.com"

_request_lock = Lock()
_last_request_at = 0.0

# Token rotation state.
# IMPORTANT: multiple PATs from the SAME GitHub user usually share the same primary
# rate limit. This rotation is useful when you have several authorized tokens from
# different users/accounts, or just want cleaner failover between tokens.
_token_lock = Lock()
_token_index = 0
_token_blocked_until: dict[str, float] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_tokens() -> list[str]:
    """Load one or more GitHub tokens from environment variables.

    Supported formats in .env:
      GITHUB_TOKENS=token1,token2,token3

    Or:
      GITHUB_TOKEN_1=token1
      GITHUB_TOKEN_2=token2
      GITHUB_TOKEN_3=token3

    Backward compatible:
      GITHUB_TOKEN=token1
      GH_TOKEN=token1
    """
    raw = os.environ.get("GITHUB_TOKENS", "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    for name in ("GITHUB_TOKEN_1", "GITHUB_TOKEN_2", "GITHUB_TOKEN_3", "GITHUB_TOKEN", "GH_TOKEN"):
        t = os.environ.get(name)
        if t and t.strip():
            tokens.append(t.strip())

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if not unique:
        raise SystemExit("ERROR: set GITHUB_TOKENS or GITHUB_TOKEN in environment or .env")
    return unique


def choose_token(tokens: list[str]) -> str:
    """Round-robin across tokens, skipping tokens blocked by rate-limit reset."""
    global _token_index
    now = time.time()
    with _token_lock:
        for _ in range(len(tokens)):
            t = tokens[_token_index % len(tokens)]
            _token_index += 1
            if _token_blocked_until.get(t, 0) <= now:
                return t

        # All tokens are blocked. Sleep until the earliest reset.
        earliest = min(_token_blocked_until.get(t, now) for t in tokens)
        wait = max(5, int(earliest - now) + 2)
    print(f"All tokens rate-limited. Sleeping {wait}s...")
    time.sleep(wait)
    return choose_token(tokens)


def api_get(url: str, params: dict[str, Any] | None = None) -> Any | None:
    global _last_request_at
    tokens = load_tokens()

    for attempt in range(6):
        t = choose_token(tokens)
        headers = {
            "Authorization": f"Bearer {t}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "m14-thesis-research",
        }

        # Keep global pacing to avoid GitHub secondary rate limits.
        with _request_lock:
            elapsed = time.time() - _last_request_at
            if elapsed < REQUEST_DELAY_SEC:
                time.sleep(REQUEST_DELAY_SEC - elapsed)
            _last_request_at = time.time()

        r = requests.get(url, headers=headers, params=params, timeout=40)
        if r.status_code == 200:
            return r.json()

        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            remaining = r.headers.get("X-RateLimit-Remaining")
            if reset and reset.isdigit() and remaining == "0":
                blocked_until = int(reset) + 5
                with _token_lock:
                    _token_blocked_until[t] = blocked_until
                wait = max(5, blocked_until - int(time.time()))
                print(f"Token rate-limited until reset; switching token if available. Reset in ~{wait}s.")
                continue

            # Secondary rate limit or abuse detection. Token rotation usually does not help.
            wait = 20 * (attempt + 1)
            print(f"Secondary/API limit suspected. Sleeping {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code in (404, 409, 422):
            return None

        time.sleep(2 * (attempt + 1))

    return None


def normalise_repos(raw: Any) -> list[tuple[str, str]]:
    repos: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, str) and "/" in item:
            owner, repo = item.split("/", 1)
        elif isinstance(item, dict):
            if "full_name" in item:
                owner, repo = item["full_name"].split("/", 1)
            else:
                owner, repo = item["owner"], item["repo"]
        else:
            continue
        repos.append((owner.strip(), repo.strip()))
    return repos


def cache_key(owner: str, repo: str, sha: str) -> Path:
    safe = f"{owner}__{repo}__{sha}.json".replace("/", "__")
    return COMMIT_CACHE_DIR / safe


def fetch_runs(owner: str, repo: str) -> list[dict[str, Any]]:
    RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = RUN_CACHE_DIR / f"{owner}__{repo}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    all_runs: list[dict[str, Any]] = []
    page = 1
    while len(all_runs) < MAX_RUNS_PER_REPO:
        url = f"{API}/repos/{owner}/{repo}/actions/runs"
        data = api_get(url, {"per_page": PER_PAGE, "page": page})
        if not data or not data.get("workflow_runs"):
            break
        for run in data["workflow_runs"]:
            conclusion = run.get("conclusion")
            status = run.get("status")
            if status != "completed" or conclusion not in INCLUDE_CONCLUSIONS:
                continue
            all_runs.append(run)
            if len(all_runs) >= MAX_RUNS_PER_REPO:
                break
        if len(data["workflow_runs"]) < PER_PAGE:
            break
        page += 1

    cache.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    return all_runs


def fetch_commit(owner: str, repo: str, sha: str) -> dict[str, Any] | None:
    if not sha:
        return None
    COMMIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = cache_key(owner, repo, sha)
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    data = api_get(f"{API}/repos/{owner}/{repo}/commits/{sha}")
    if data:
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


SRC_RE = re.compile(r"(src/|app/|lib/|server/|service/|main/|\.java$|\.py$|\.kt$|\.js$|\.ts$)", re.I)
TEST_RE = re.compile(r"(test|tests|spec|__tests__|src/test|pytest|junit|surefire)", re.I)
BUILD_RE = re.compile(r"(pom\.xml|build\.gradle|settings\.gradle|gradle\.properties|Makefile|CMakeLists\.txt|setup\.py|pyproject\.toml|tox\.ini)", re.I)
DEP_RE = re.compile(r"(pom\.xml|build\.gradle|requirements.*\.txt|poetry\.lock|Pipfile|Pipfile\.lock|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|gradle\.lockfile)", re.I)
CI_RE = re.compile(r"(\.github/workflows/|Jenkinsfile|\.gitlab-ci\.yml|circleci|azure-pipelines)", re.I)
DOC_RE = re.compile(r"(README|CHANGELOG|LICENSE|\.md$|docs/|\.rst$|\.txt$)", re.I)


def extract_change_features(commit: dict[str, Any] | None) -> dict[str, Any]:
    if not commit:
        return {
            "files_changed_count": 0, "lines_added": 0, "lines_deleted": 0,
            "src_files_changed": 0, "test_files_changed": 0, "build_files_changed": 0,
            "ci_config_changed": 0, "dependency_files_changed": 0,
            "docs_only_change": 0, "has_large_change": 0,
        }
    stats = commit.get("stats") or {}
    files = commit.get("files") or []
    names = [f.get("filename", "") for f in files]
    additions = int(stats.get("additions") or sum(int(f.get("additions", 0)) for f in files))
    deletions = int(stats.get("deletions") or sum(int(f.get("deletions", 0)) for f in files))
    total = additions + deletions
    nfiles = len(names)

    src = sum(1 for n in names if SRC_RE.search(n) and not TEST_RE.search(n))
    tests = sum(1 for n in names if TEST_RE.search(n))
    build = sum(1 for n in names if BUILD_RE.search(n))
    ci = sum(1 for n in names if CI_RE.search(n))
    deps = sum(1 for n in names if DEP_RE.search(n))
    docs = sum(1 for n in names if DOC_RE.search(n))

    return {
        "files_changed_count": nfiles,
        "lines_added": additions,
        "lines_deleted": deletions,
        "src_files_changed": src,
        "test_files_changed": tests,
        "build_files_changed": build,
        "ci_config_changed": ci,
        "dependency_files_changed": deps,
        "docs_only_change": int(nfiles > 0 and docs == nfiles),
        "has_large_change": int(nfiles >= 20 or total >= 500),
    }


def process_repo(owner: str, repo: str) -> list[dict[str, Any]]:
    runs = fetch_runs(owner, repo)
    rows: list[dict[str, Any]] = []
    for run in runs:
        sha = run.get("head_sha") or ""
        commit = fetch_commit(owner, repo, sha)
        change = extract_change_features(commit)
        conclusion = run.get("conclusion")
        rows.append({
            "repository": f"{owner}/{repo}",
            "owner": owner,
            "repo": repo,
            "run_id": run.get("id"),
            "run_number": run.get("run_number"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "event": run.get("event"),
            "workflow_name": run.get("name"),
            "head_branch": run.get("head_branch"),
            "head_sha": sha,
            "conclusion": conclusion,
            "build_failed": int(conclusion in {"failure", "timed_out"}),
            **change,
        })
    return rows


def main() -> None:
    load_env()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not REPOS_FILE.exists():
        raise SystemExit("ERROR: repos.json not found")

    repos = normalise_repos(json.loads(REPOS_FILE.read_text(encoding="utf-8")))
    print(f"Repos: {len(repos)} | max runs/repo: {MAX_RUNS_PER_REPO} | workers: {WORKERS}")

    all_rows: list[dict[str, Any]] = []
    repo_stats = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_repo, owner, repo): (owner, repo) for owner, repo in repos}
        for fut in as_completed(futs):
            owner, repo = futs[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                c = Counter(r["conclusion"] for r in rows)
                repo_stats.append((f"{owner}/{repo}", len(rows), dict(c)))
                print(f"✓ {owner}/{repo:<40} rows={len(rows):>4} {dict(c)}")
            except Exception as e:
                print(f"✗ {owner}/{repo}: {e}")

    if not all_rows:
        raise SystemExit("No rows collected")

    # chronological order is important for sequence construction
    all_rows.sort(key=lambda r: (r["repository"], r.get("created_at") or ""))

    fieldnames = list(all_rows[0].keys())
    with RUNS_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    with REPORT_OUT.open("w", encoding="utf-8") as f:
        f.write(f"M14 GitHub run fetch report\nStarted/finished: {datetime.now().isoformat()}\n\n")
        f.write(f"Repos: {len(repos)}\nRows: {len(all_rows)}\n")
        f.write(f"Overall failure rate: {sum(r['build_failed'] for r in all_rows)/len(all_rows):.4f}\n\n")
        f.write("Per repo:\n")
        for repo, n, c in sorted(repo_stats):
            f.write(f"  {repo:<45} {n:>5} {c}\n")

    print(f"\nSaved: {RUNS_OUT}")
    print(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
