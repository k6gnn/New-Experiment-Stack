import json
import csv
from pathlib import Path

# Change this if your M13 folder is somewhere else
M13_DATASET = Path(r"C:\Users\Kanan\Desktop\New Experiment Stack\ML_Training_Set\Ollama_DataGathering\data\final\github_actions_training_dataset.jsonl")

OUT = Path("data/m13_run_labels.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {
    "compilation",
    "test_failure",
    "flaky_test",
    "configuration",
    "infrastructure",
}

def pick(row, names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return ""

rows_out = []
skipped = 0

with M13_DATASET.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except Exception:
            skipped += 1
            continue

        repository = pick(row, [
            "repository",
            "repo",
            "repository_name",
            "project",
            "gh_project_name",
        ])

        run_id = pick(row, [
            "run_id",
            "github_run_id",
            "workflow_run_id",
            "runId",
            "id",
        ])

        failure_type = pick(row, [
            "failure_type",
            "label",
            "target",
            "final_label",
            "m13_label",
        ])

        failure_type = str(failure_type).strip()

        if not repository or not run_id or failure_type not in VALID_LABELS:
            skipped += 1
            continue

        rows_out.append({
            "repository": str(repository).strip(),
            "run_id": str(run_id).strip(),
            "failure_type": failure_type,
        })

# Remove duplicates
unique = {}
for r in rows_out:
    key = (r["repository"], r["run_id"])
    unique[key] = r

rows_out = list(unique.values())

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["repository", "run_id", "failure_type"])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Written: {OUT}")
print(f"Rows: {len(rows_out)}")
print(f"Skipped: {skipped}")

from collections import Counter
print("Label distribution:")
for label, count in Counter(r["failure_type"] for r in rows_out).most_common():
    print(f"  {label:<18} {count}")