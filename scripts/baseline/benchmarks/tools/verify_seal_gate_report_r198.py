#!/usr/bin/env python3
"""Read-only r198 verification; never starts paid or encrypted execution."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "docs/baseline/seal-artifact-gate-closure-r198.json"
REPORT_SHA256 = "4184819d6abf677f36d82e6996255ee2441bec52949bf57560b4f6587696e6e2"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if sha(REPORT) != REPORT_SHA256:
        raise SystemExit("Report changed; preserve r198 and create a new binding")
    data = json.loads(REPORT.read_text())
    checked = 0
    for section in ("parents", "source_hashes", "guarded_compiler_runtime_files_unchanged"):
        for name, expected in data[section].items():
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file() or sha(path) != expected:
                raise SystemExit("Missing/changed evidence or source: " + str(path))
            checked += 1
    print(json.dumps({
        "verified_hash_entries": checked, "task_counts": data["task_counts"],
        "unresolved": data["unresolved"], "recorded_generations": data["generations"],
        "recorded_http_attempts": data["http_attempts"], "new_paid_calls": 0,
        "new_encrypted_execution": False,
    }, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
