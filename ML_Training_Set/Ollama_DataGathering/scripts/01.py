import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Dual-token setup ──────────────────────────────────────────────────────────
# Add GITHUB_TOKEN and optionally GITHUB_TOKEN_2 to your .env
# Two tokens = 10,000 requests/hour instead of 5,000
_TOKENS = [t for t in [
    os.getenv("GITHUB_TOKEN", ""),
    os.getenv("GITHUB_TOKEN_2", ""),
] if t]

if not _TOKENS:
    print("WARNING: No GITHUB_TOKEN found. Unauthenticated limit is only 60/hour!")

_token_index = 0
_token_lock  = threading.Lock()

def next_token() -> str:
    """Round-robin across available tokens."""
    global _token_index
    if not _TOKENS:
        return ""
    with _token_lock:
        tok = _TOKENS[_token_index % len(_TOKENS)]
        _token_index += 1
    return tok


REPOS_PATH  = Path("repos.json")
RAW_LOG_DIR = Path("data/raw_logs")
RECORDS_DIR = Path("data/records")
INDEX_PATH  = RECORDS_DIR / "jobs_index.jsonl"

RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# ── Performance settings ──────────────────────────────────────────────────────
RUNS_PER_REPO          = 300
MAX_REPO_WORKERS       = 16
MAX_LOG_WORKERS        = 6
# With 2 tokens: 2 × 5000/hour = 10000/hour → safe at 2.7 req/s
# With 1 token:  5000/hour → safe at 1.38 req/s
# Script auto-selects based on how many tokens are loaded.
_BASE_RATE             = 1.38   # per token per second
GLOBAL_REQUESTS_PER_SECOND = _BASE_RATE * len(_TOKENS) if _TOKENS else _BASE_RATE
LOG_CAP_CHARS          = 700_000  # head 100k + tail 600k

SKIP_WORKFLOW_PATTERNS = [
    r"codeql", r"security", r"scorecard", r"dependabot", r"renovate",
    r"release", r"publish", r"deploy", r"docker.?push", r"github.?pages",
    r"pages.?deploy", r"coverage.?report", r"codecov", r"stale",
    r"labeler", r"benchmark", r"notify",
]

# ── Token-bucket rate limiter ─────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, per_second: float):
        self.lock     = threading.Lock()
        self.interval = 1.0 / per_second
        self.last     = 0.0

    def acquire(self):
        with self.lock:
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

rate_limiter = RateLimiter(GLOBAL_REQUESTS_PER_SECOND)

# ── Rate limit hit counter — one clean line per hit, no spam ─────────────────
_rate_limit_hits = 0
_rate_limit_lock = threading.Lock()

def _handle_rate_limit(wait_seconds: int):
    global _rate_limit_hits
    with _rate_limit_lock:
        _rate_limit_hits += 1
        tqdm.write(f"  [rate limit #{_rate_limit_hits}] pausing {wait_seconds}s — will resume automatically")
    time.sleep(wait_seconds)


# ── Per-thread session (one session per thread, token rotates per request) ────
_session_local = threading.local()

def get_session() -> requests.Session:
    """One persistent session per thread for keep-alive. Token is set per-request."""
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=0,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        _session_local.session = s
    return _session_local.session

def _auth_headers() -> dict:
    tok = next_token()
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def github_get_json(url: str):
    for attempt in range(4):
        rate_limiter.acquire()
        try:
            r = get_session().get(url, headers=_auth_headers(), timeout=20)

            if r.status_code in (403, 429):
                _handle_rate_limit(int(r.headers.get("Retry-After", "60")))
                continue
            if r.status_code in (404, 410):
                return None
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException:
            time.sleep(2 ** attempt)

    return None


def github_get_text(url: str) -> str:
    for attempt in range(4):
        rate_limiter.acquire()
        try:
            r = get_session().get(url, headers=_auth_headers(), timeout=30, allow_redirects=True)

            if r.status_code in (404, 410):
                return ""
            if r.status_code in (403, 429):
                _handle_rate_limit(int(r.headers.get("Retry-After", "60")))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            r.raise_for_status()
            text = r.text

            if len(text) > LOG_CAP_CHARS:
                head = text[:100_000]
                tail = text[-(LOG_CAP_CHARS - 100_000):]
                return head + "\n\n...[middle truncated]...\n\n" + tail

            return text

        except requests.RequestException:
            time.sleep(2 ** attempt)

    return ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def should_skip_workflow(run: dict) -> bool:
    combined = (str(run.get("name", "")) + " " + str(run.get("path", ""))).lower()
    return any(re.search(pat, combined, re.I) for pat in SKIP_WORKFLOW_PATTERNS)

def safe_repo(owner: str, repo: str) -> str:
    return f"{owner}__{repo}"

def get_failed_runs(owner: str, repo: str):
    all_runs, page = [], 1
    while len(all_runs) < RUNS_PER_REPO:
        data = github_get_json(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
            f"?status=failure&per_page=100&page={page}"
        )
        if not data:
            break
        runs = data.get("workflow_runs", [])
        if not runs:
            break
        for run in runs:
            if not should_skip_workflow(run):
                all_runs.append(run)
            if len(all_runs) >= RUNS_PER_REPO:
                break
        if len(runs) < 100:
            break
        page += 1
    return all_runs[:RUNS_PER_REPO]

def get_jobs(owner: str, repo: str, run_id: int):
    jobs, page = [], 1
    while True:
        data = github_get_json(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
            f"?per_page=100&page={page}"
        )
        if not data:
            break
        batch = data.get("jobs", [])
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs

def failed_step_name(job: dict) -> str:
    for step in job.get("steps", []):
        if step.get("conclusion") == "failure":
            return step.get("name", "")
    return job.get("name", "")


# ── Resume: load already-indexed job IDs to prevent duplicates ───────────────
def load_indexed_job_ids() -> set:
    if not INDEX_PATH.exists():
        return set()
    seen = set()
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["job_id"])
            except Exception:
                pass
    return seen


index_lock = threading.Lock()

def append_jsonl(path: Path, row: dict, lock: threading.Lock):
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Per-job download ──────────────────────────────────────────────────────────
def _download_job(args):
    owner, repo, repo_log_dir, run, job, indexed_ids = args
    run_id    = run["id"]
    job_id    = job["id"]
    log_path  = repo_log_dir / f"run_{run_id}_job_{job_id}.log"

    # Skip if already in the index (dedup — protects previous effort)
    if job_id in indexed_ids:
        return None, "already_indexed"

    if log_path.exists() and log_path.stat().st_size > 0:
        cached = True
    else:
        log_text = github_get_text(
            f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        )
        if not log_text.strip():
            return None, "empty"
        log_path.write_text(log_text, encoding="utf-8", errors="ignore")
        cached = False

    record = {
        "repo":          f"{owner}/{repo}",
        "owner":         owner,
        "repo_name":     repo,
        "lang":          run.get("_lang", ""),
        "run_id":        run_id,
        "run_number":    run.get("run_number"),
        "job_id":        job_id,
        "workflow_name": run.get("name", ""),
        "job_name":      job.get("name", ""),
        "failing_step":  failed_step_name(job),
        "created_at":    run.get("created_at", ""),
        "log_path":      str(log_path),
    }
    return record, "cached" if cached else "downloaded"


# ── Per-repo fetcher ──────────────────────────────────────────────────────────
def fetch_repo(repo_info: dict, indexed_ids: set):
    owner    = repo_info["owner"]
    repo     = repo_info["repo"]
    lang     = repo_info["lang"]
    repo_key = safe_repo(owner, repo)

    # Resume: skip repos that already fully completed
    stats_path = RECORDS_DIR / f"{repo_key}_fetch_stats.json"
    if stats_path.exists():
        tqdm.write(f"  [skip] {owner}/{repo} — already completed")
        return json.loads(stats_path.read_text(encoding="utf-8"))

    repo_log_dir      = RAW_LOG_DIR / repo_key
    repo_log_dir.mkdir(parents=True, exist_ok=True)
    repo_records_path = RECORDS_DIR / f"{repo_key}_records.jsonl"

    tqdm.write(f"  [start] {owner}/{repo}")

    runs = get_failed_runs(owner, repo)
    for r in runs:
        r["_lang"] = lang

    stats = {
        "repo":            f"{owner}/{repo}",
        "runs_seen":       len(runs),
        "jobs_downloaded": 0,
        "jobs_cached":     0,
        "already_indexed": 0,
        "empty_logs":      0,
        "no_failed_jobs":  0,
    }

    # Step 1 — list all failed jobs across runs
    job_tasks = []
    for run in tqdm(runs, desc=f"{owner}/{repo} | listing runs", leave=False, unit="run"):
        jobs   = get_jobs(owner, repo, run["id"])
        failed = [j for j in jobs if j.get("conclusion") == "failure"]
        if not failed:
            stats["no_failed_jobs"] += 1
            continue
        for job in failed:
            job_tasks.append((owner, repo, repo_log_dir, run, job, indexed_ids))

    # Step 2 — download logs in parallel
    with ThreadPoolExecutor(max_workers=MAX_LOG_WORKERS) as pool:
        futures = {pool.submit(_download_job, t): t for t in job_tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"{owner}/{repo} | downloading logs",
                           leave=False, unit="log"):
            try:
                record, status = future.result()
            except Exception as e:
                tqdm.write(f"  [error] {owner}/{repo}: {e}")
                continue

            if status == "already_indexed":
                stats["already_indexed"] += 1
                continue
            elif status == "empty" or record is None:
                stats["empty_logs"] += 1
                continue
            elif status == "cached":
                stats["jobs_cached"] += 1
            else:
                stats["jobs_downloaded"] += 1

            with index_lock:
                indexed_ids.add(record["job_id"])

            append_jsonl(repo_records_path, record, index_lock)
            append_jsonl(INDEX_PATH, record, index_lock)

    # Writing stats marks this repo as done for future resumes
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    tqdm.write(
        f"  [done] {owner}/{repo}: "
        f"new={stats['jobs_downloaded']} "
        f"cached={stats['jobs_cached']} "
        f"skipped={stats['already_indexed']} "
        f"empty={stats['empty_logs']}"
    )
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not REPOS_PATH.exists():
        raise FileNotFoundError("Missing repos.json")

    repos = json.loads(REPOS_PATH.read_text(encoding="utf-8"))

    tqdm.write("Loading existing index to prevent duplicates...")
    indexed_ids = load_indexed_job_ids()
    tqdm.write(f"  Already indexed: {len(indexed_ids)} jobs — these will be skipped")

    tokens_loaded = len(_TOKENS)
    effective_rate = GLOBAL_REQUESTS_PER_SECOND

    print("=" * 70)
    print("Massive GitHub Actions log fetcher")
    print(f"Repos           : {len(repos)}")
    print(f"Runs per repo   : {RUNS_PER_REPO}")
    print(f"Repo workers    : {MAX_REPO_WORKERS}  |  Log workers: {MAX_LOG_WORKERS} per repo")
    print(f"Tokens loaded   : {tokens_loaded}  ({tokens_loaded} × 5000/hr = {tokens_loaded * 5000}/hr capacity)")
    print(f"Rate limit      : {effective_rate:.2f} req/s  (~{int(effective_rate * 3600)}/hr)")
    print(f"Already indexed : {len(indexed_ids)} jobs (will be skipped)")
    print("=" * 70)

    all_stats = []
    with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as pool:
        futures = [pool.submit(fetch_repo, r, indexed_ids) for r in repos]
        for future in tqdm(as_completed(futures), total=len(repos),
                           desc="Overall progress", unit="repo"):
            try:
                all_stats.append(future.result())
            except Exception as e:
                tqdm.write(f"  [repo failed] {e}")

    summary_path = RECORDS_DIR / "fetch_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")

    total_new     = sum(s.get("jobs_downloaded", 0) for s in all_stats)
    total_cached  = sum(s.get("jobs_cached", 0)     for s in all_stats)
    total_skipped = sum(s.get("already_indexed", 0) for s in all_stats)
    total_empty   = sum(s.get("empty_logs", 0)      for s in all_stats)

    print("\n" + "=" * 70)
    print("FINISHED")
    print(f"  New logs downloaded : {total_new}")
    print(f"  Loaded from cache   : {total_cached}")
    print(f"  Skipped (duplicate) : {total_skipped}")
    print(f"  Empty / expired     : {total_empty}")
    print(f"  Index saved to      : {INDEX_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
