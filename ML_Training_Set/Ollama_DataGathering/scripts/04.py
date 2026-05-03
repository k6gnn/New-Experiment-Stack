import json
from pathlib import Path
from collections import Counter

RULES_PATH  = Path("data/snippets/snippets_with_rules.jsonl")
OLLAMA_PATH = Path("data/llm_labeled/ollama_labeled_uncertain.jsonl")

TRAIN_PATH   = Path("data/final/github_actions_training_dataset.jsonl")
REVIEW_PATH  = Path("data/final/needs_manual_review.jsonl")
SUMMARY_PATH = Path("data/final/dataset_summary.json")

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {"compilation", "test_failure", "flaky_test", "infrastructure", "configuration"}

RULE_ACCEPT_THRESHOLD   = 0.82
OLLAMA_ACCEPT_THRESHOLD = 0.80

# When rule and Ollama disagree, penalise accepted confidence by this factor
DISAGREEMENT_PENALTY = 0.90


def load_ollama_by_source() -> dict:
    data = {}
    if not OLLAMA_PATH.exists():
        return data
    with OLLAMA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                data[str(row["source_file"])] = row
            except Exception:
                pass
    return data


def make_training_row(row: dict, label: str, confidence: float,
                      source: str, reason: str) -> dict:
    return {
        # Core training fields
        "text":         row["text"],
        "label":        label,
        "confidence":   round(confidence, 4),
        "label_source": source,
        "reason":       reason,

        # Fine-grained ontology label preserved from script 02
        "primary_label": row.get("primary_label"),

        # Metadata
        "repo":          row.get("repo"),
        "lang":          row.get("lang"),
        "run_id":        row.get("run_id"),
        "run_number":    row.get("run_number"),
        "job_id":        row.get("job_id"),
        "workflow_name": row.get("workflow_name"),
        "job_name":      row.get("job_name"),
        "failing_step":  row.get("failing_step"),
        "created_at":    row.get("created_at"),
        "source_file":   row.get("source_file"),

        # Rule signal
        "rule_label":      row.get("rule_label"),
        "rule_confidence": row.get("rule_confidence"),
        "rule_reason":     row.get("rule_reason"),

        # Ollama signal
        "ollama_label":      row.get("ollama_label"),
        "ollama_confidence": row.get("ollama_confidence"),
        "ollama_reason":     row.get("ollama_reason"),
        "ollama_model":      row.get("ollama_model"),
    }


def main():
    if not RULES_PATH.exists():
        raise FileNotFoundError("Run 02_extract_snippets_and_rules.py first")

    ollama_by_source = load_ollama_by_source()
    print(f"Ollama-labeled rows loaded : {len(ollama_by_source):,}")

    counts        = Counter()
    source_counts = Counter()
    repo_counts   = Counter()
    lang_counts   = Counter()
    accepted = review = skipped_dupe = disagreements = 0

    seen_job_ids: set = set()

    with RULES_PATH.open("r", encoding="utf-8") as inp, \
         TRAIN_PATH.open("w", encoding="utf-8")  as train_out, \
         REVIEW_PATH.open("w", encoding="utf-8") as review_out:

        for line in inp:
            try:
                base = json.loads(line)
            except Exception:
                continue

            # ── Deduplication by job_id ───────────────────────────────────────
            job_id = base.get("job_id")
            if job_id and job_id in seen_job_ids:
                skipped_dupe += 1
                continue
            if job_id:
                seen_job_ids.add(job_id)

            source_file = str(base["source_file"])
            rule_label  = base.get("rule_label")
            rule_conf   = float(base.get("rule_confidence") or 0)

            final_label      = None
            final_confidence = 0.0
            label_source     = None
            reason           = ""

            # 1. Strong rule label → accept immediately
            if rule_label in VALID_LABELS and rule_conf >= RULE_ACCEPT_THRESHOLD:
                final_label      = rule_label
                final_confidence = rule_conf
                label_source     = "rule"
                reason           = base.get("rule_reason", "")

            # 2. Uncertain → check Ollama result
            else:
                ollama = ollama_by_source.get(source_file)
                if ollama:
                    base.update({
                        "ollama_label":      ollama.get("ollama_label"),
                        "ollama_confidence": ollama.get("ollama_confidence"),
                        "ollama_reason":     ollama.get("ollama_reason"),
                        "ollama_model":      ollama.get("ollama_model"),
                    })
                    ollama_label = ollama.get("ollama_label")
                    ollama_conf  = float(ollama.get("ollama_confidence") or 0)

                    if ollama_label in VALID_LABELS and ollama_conf >= OLLAMA_ACCEPT_THRESHOLD:
                        final_label      = ollama_label
                        final_confidence = ollama_conf
                        label_source     = "ollama"
                        reason           = ollama.get("ollama_reason", "")

                        # ── Agreement check ───────────────────────────────────
                        # Rule had a valid opinion that differs from Ollama →
                        # penalise confidence so training treats this as less certain
                        if (rule_label in VALID_LABELS
                                and rule_label != ollama_label
                                and rule_conf >= 0.50):
                            final_confidence *= DISAGREEMENT_PENALTY
                            disagreements    += 1
                            reason = (f"[disagreement: rule={rule_label} "
                                      f"ollama={ollama_label}] {reason}")

            # 3. Accept or flag for review
            if final_label in VALID_LABELS:
                train_row = make_training_row(
                    base, final_label, final_confidence, label_source, reason
                )
                train_out.write(json.dumps(train_row, ensure_ascii=False) + "\n")
                accepted += 1
                counts[final_label]         += 1
                source_counts[label_source] += 1
                repo_counts[base.get("repo", "")] += 1
                lang_counts[base.get("lang", "")]  += 1
            else:
                base["review_reason"] = "low_confidence_or_missing_ollama"
                review_out.write(json.dumps(base, ensure_ascii=False) + "\n")
                review += 1

    summary = {
        "accepted_training_examples": accepted,
        "needs_manual_review":        review,
        "skipped_duplicates":         skipped_dupe,
        "rule_ollama_disagreements":  disagreements,
        "label_distribution":         dict(counts),
        "label_source_distribution":  dict(source_counts),
        "language_distribution":      dict(lang_counts),
        "top_repos":                  dict(repo_counts.most_common(30)),
        "train_path":                 str(TRAIN_PATH),
        "review_path":                str(REVIEW_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
