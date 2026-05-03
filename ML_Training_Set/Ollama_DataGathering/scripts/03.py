#!/usr/bin/env python3
"""
03_label_uncertain_with_ollama.py  (v4 — final complete edition)

Reads snippets_with_rules.jsonl produced by script 02.
Sends rows where rule_confidence < RULE_ACCEPT_THRESHOLD to a local Ollama
model for labeling. Saves results to ollama_labeled_uncertain.jsonl.

Features:
  - Parallel workers (tune OLLAMA_WORKERS for your GPU)
  - Full resume — restart anytime, already-labeled rows are skipped
  - Graceful Ctrl+C — in-flight requests finish, nothing is half-written
  - Strict output: label is always one of 5 buckets or None
  - Rich prompt with ontology-aligned decision rules and disambiguation hints
  - Retry on Ollama transient errors with exponential backoff
  - Primary label hint from script 02 passed to Ollama as context
  - num_predict cap to prevent runaway generation
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL    = "http://localhost:11434/api/generate"

# Tune for your GPU. RTX 4060 8GB + qwen2.5:7b → start at 3.
# Watch nvidia-smi: if GPU util < 90%, raise by 1.
OLLAMA_WORKERS = 3

INPUT_PATH = Path("data/snippets/snippets_with_rules.jsonl")
OUT_PATH   = Path("data/llm_labeled/ollama_labeled_uncertain.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Rows with rule_confidence >= this are already confident — skip Ollama
RULE_ACCEPT_THRESHOLD     = 0.82
MAX_TEXT_CHARS_FOR_OLLAMA = 10_000

# Max tokens Ollama generates — label + confidence + short reason fits in 60
OLLAMA_NUM_PREDICT = 80

# Retry config for transient Ollama errors
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0   # seconds, doubles each retry

# ── Valid labels ──────────────────────────────────────────────────────────────
VALID_LABELS = {
    "compilation", "test_failure", "flaky_test",
    "infrastructure", "configuration",
}

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert GitHub Actions CI/CD failure classifier.

Classify the log snippet below into exactly one of these labels:

  compilation    — source code failed to compile, type-check, link, or build.
                   Includes: javac/kotlinc/tsc/rustc/gcc errors, linker errors,
                   SyntaxError, IndentationError, unresolved symbols, build task
                   failures BEFORE any tests ran.

  test_failure   — tests ran and failed deterministically.
                   Includes: assertion errors (AssertionError, expected/actual),
                   JUnit/pytest/Jest/RSpec/Go test failures, snapshot mismatches,
                   test exceptions. NOT flaky — same failure every run.

  flaky_test     — test failure is non-deterministic / intermittent.
                   Includes: TestTimedOutException, ConcurrentModificationException,
                   data race, deadlock, rerun markers (RERUN, Flakes: N), timing
                   issues, Selenium stale element, unhandled promise rejection.
                   Choose this over test_failure if timing/concurrency signals exist.

  infrastructure — external environment failure, not the code's fault.
                   Includes: runner shutdown/lost communication, OOM (exit 137),
                   disk full (ENOSPC), network errors (DNS, TLS, 502/503/504),
                   Docker pull failures, GitHub Actions cache/artifact HTTP errors,
                   rate limiting (429), registry timeouts, pod start failures.

  configuration  — setup or configuration is wrong, not the code logic itself.
                   Includes: missing secrets/env vars, bad YAML, wrong tool
                   version, missing files, permission denied, linting failures
                   (flake8/mypy/eslint/black), pre-commit hooks, dependency
                   resolution failures (pip no version, npm 404), auth failures.

━━━ DECISION RULES (apply in order) ━━━
1. Runner shutdown / lost communication / OOM (exit 137) → infrastructure
2. Explicit retry/rerun markers OR concurrency/timing signals → flaky_test
3. Test ran and produced expected/actual mismatch → test_failure
4. Source could not compile BEFORE any test ran → compilation
5. Network/registry/Docker/cache HTTP errors → infrastructure
6. Missing secret/env var/file/YAML/permission/lint → configuration

━━━ DISAMBIGUATION ━━━
- pip "no version that satisfies" with NO network retries → configuration
- pip ReadTimeoutError with retries → infrastructure
- docker "manifest unknown" intermittent → infrastructure; persistent → configuration
- exit code 1 alone → look at what's above it, not the exit code itself
- "COMPILATION ERROR" inside a test task → still compilation
- flake8/mypy/ruff errors → configuration (static analysis), NOT compilation

━━━ METADATA ━━━
repo          : {repo}
language      : {lang}
workflow      : {workflow_name}
job           : {job_name}
failing_step  : {failing_step}
rule_hint     : {rule_hint}
rule_conf     : {rule_conf}

━━━ LOG SNIPPET ━━━
{text}

━━━ OUTPUT ━━━
Return ONLY valid JSON, no explanation outside the JSON:
{{"label": "compilation|test_failure|flaky_test|infrastructure|configuration", "confidence": 0.0, "reason": "max 12 words"}}
"""

# ── Shutdown handling ─────────────────────────────────────────────────────────
_shutdown = threading.Event()

def _handle_signal(sig, frame):
    tqdm.write(
        "\n[interrupted] Waiting for in-flight requests to finish "
        "before saving. Ctrl+C again to force quit.")
    _shutdown.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

_write_lock = threading.Lock()


# ── Resume ────────────────────────────────────────────────────────────────────
def already_labeled_keys() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    seen: set[str] = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(str(json.loads(line)["source_file"]))
            except Exception:
                pass
    return seen


# ── Ollama communication ──────────────────────────────────────────────────────
def parse_ollama_response(response_text: str) -> dict:
    """Extract the inner JSON from Ollama's wrapper response."""
    outer = json.loads(response_text)
    inner = outer.get("response", "").strip()
    try:
        return json.loads(inner)
    except json.JSONDecodeError:
        # Try to extract JSON object if model added surrounding text
        start = inner.find("{")
        end   = inner.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(inner[start:end])
        raise ValueError(f"Could not parse Ollama JSON: {inner[:300]}")


def ask_ollama(row: dict) -> dict:
    """Send one row to Ollama. Retries on transient errors."""
    text = row.get("text", "")
    if len(text) > MAX_TEXT_CHARS_FOR_OLLAMA:
        # Take the last N chars — errors are at the end
        text = text[-MAX_TEXT_CHARS_FOR_OLLAMA:]

    # Pass script 02's best guess as a hint so Ollama has context
    rule_hint = row.get("primary_label") or row.get("rule_label") or "unknown"
    rule_conf = f"{float(row.get('rule_confidence') or 0):.2f}"

    prompt = PROMPT_TEMPLATE.format(
        repo=row.get("repo", ""),
        lang=row.get("lang", ""),
        workflow_name=row.get("workflow_name", ""),
        job_name=row.get("job_name", ""),
        failing_step=row.get("failing_step", ""),
        rule_hint=rule_hint,
        rule_conf=rule_conf,
        text=text,
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature":   0,
            "num_predict":   OLLAMA_NUM_PREDICT,
            "repeat_penalty": 1.1,
        },
    }

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()
            return parse_ollama_response(r.text)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                tqdm.write(f"  [ollama retry {attempt+1}/{MAX_RETRIES}] {exc} — waiting {wait:.0f}s")
                time.sleep(wait)

    raise RuntimeError(f"Ollama failed after {MAX_RETRIES} attempts: {last_exc}")


# ── Per-row worker ────────────────────────────────────────────────────────────
def process_row(row: dict, out_file) -> str:
    """Label one row and write result. Returns status string."""
    if _shutdown.is_set():
        return "skipped_shutdown"

    try:
        result     = ask_ollama(row)
        label      = result.get("label")
        confidence = float(result.get("confidence") or 0)
        reason     = str(result.get("reason", ""))[:200]

        if label not in VALID_LABELS:
            label      = None
            confidence = 0.0
            reason     = f"invalid label returned: {result}"

        out_row = {
            **row,
            "ollama_label":      label,
            "ollama_confidence": round(confidence, 4),
            "ollama_reason":     reason,
            "ollama_model":      OLLAMA_MODEL,
        }
        status = "labeled"

    except Exception as exc:
        out_row = {
            **row,
            "ollama_label":      None,
            "ollama_confidence": 0.0,
            "ollama_reason":     f"ollama_error: {exc}",
            "ollama_model":      OLLAMA_MODEL,
        }
        status = "error"

    with _write_lock:
        out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_file.flush()

    return status


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run 02_extract_snippets_and_rules.py first.")

    tqdm.write("Loading already-labeled keys for resume...")
    seen = already_labeled_keys()
    tqdm.write(f"  Already done: {len(seen):,} — will be skipped")

    rows_to_label: list[dict] = []
    skipped_confident = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if str(row.get("source_file", "")) in seen:
                continue

            conf = float(row.get("rule_confidence") or 0)
            if conf >= RULE_ACCEPT_THRESHOLD:
                skipped_confident += 1
                continue

            rows_to_label.append(row)

    print("=" * 62)
    print(f"  Ollama model    : {OLLAMA_MODEL}")
    print(f"  Workers         : {OLLAMA_WORKERS}  (parallel Ollama requests)")
    print(f"  num_predict cap : {OLLAMA_NUM_PREDICT} tokens")
    print(f"  Confidence gate : {RULE_ACCEPT_THRESHOLD}")
    print(f"  To label        : {len(rows_to_label):,}")
    print(f"  Already done    : {len(seen):,}")
    print(f"  Skipped (conf≥{RULE_ACCEPT_THRESHOLD}) : {skipped_confident:,}")
    print(f"  Output          : {OUT_PATH}")
    print("  Ctrl+C to pause — safe to resume anytime by rerunning")
    print("=" * 62)

    if not rows_to_label:
        print("Nothing to label. All rows are either done or confident.")
        return

    stats: dict[str, int] = {
        "labeled": 0, "error": 0, "skipped_shutdown": 0}
    label_counts: dict[str, int] = {}

    with OUT_PATH.open("a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=OLLAMA_WORKERS) as pool:
            futures = {
                pool.submit(process_row, row, out_file): row
                for row in rows_to_label
            }

            with tqdm(total=len(rows_to_label),
                      desc="Ollama labeling", unit="snippet") as pbar:
                for future in as_completed(futures):
                    if _shutdown.is_set():
                        for f in futures:
                            f.cancel()

                    try:
                        status = future.result()
                    except Exception as exc:
                        tqdm.write(f"  [worker error] {exc}")
                        status = "error"

                    stats[status] = stats.get(status, 0) + 1
                    pbar.update(1)
                    pbar.set_postfix(
                        labeled=stats["labeled"],
                        errors=stats["error"],
                        refresh=False,
                    )

    # ── Final summary ─────────────────────────────────────────────────────────
    # Count label distribution from output file
    if OUT_PATH.exists():
        with OUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    lbl = json.loads(line).get("ollama_label")
                    if lbl:
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
                except Exception:
                    pass

    print("\n" + "=" * 62)
    status_word = "FINISHED" if not _shutdown.is_set() else "INTERRUPTED — safe to resume"
    print(f"  {status_word}")
    print(f"  Labeled    : {stats['labeled']:,}")
    print(f"  Errors     : {stats['error']:,}")
    if stats.get("skipped_shutdown"):
        print(f"  Skipped    : {stats['skipped_shutdown']:,}  (next run will continue)")

    if label_counts:
        print("\n  Label distribution (all output so far):")
        for lbl in ("compilation", "test_failure", "flaky_test",
                    "infrastructure", "configuration"):
            print(f"    {lbl:<22} {label_counts.get(lbl, 0):>7}")
        invalid = {k: v for k, v in label_counts.items() if k not in VALID_LABELS}
        if invalid:
            print(f"    [invalid labels]      {sum(invalid.values()):>7}")

    print(f"\n  Output saved to: {OUT_PATH}")
    print("=" * 62)


if __name__ == "__main__":
    main()
