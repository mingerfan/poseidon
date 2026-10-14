#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
results = json.loads((root / "loads.json").read_text())
old = json.loads((root.parent / "runtime-binary-qwen-188-20261011/binary-load.json").read_text())
comparison = json.loads((root / "field-comparison.json").read_text())
assert comparison["all_fields_equal"] and comparison["hash_seconds"] == 0
assert len(results) == 9
for row in results:
    assert not row["source_hashing"] and row["hash_seconds"] == 0 and row["blob_payloads_read"] == 0
    assert row["values"] == old["values"] and row["instructions"] == old["instructions"]
    assert row["keys"] == old["keys"] and row["capabilities"] == old["capabilities"]
    assert row["bytes"] == row["source_bytes"] == old["bytes"]


def aggregate(rows):
    metrics = {}
    for key in ("load_seconds", "verify_seconds", "read_seconds", "scan_allocate_seconds", "decode_seconds",
                "peak_rss_bytes", "verify_peak_rss_bytes"):
        values = [r[key] for r in rows if key in r]
        if values:
            metrics[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    totals = [r["load_seconds"] + r["verify_seconds"] for r in rows]
    metrics["load_verify_seconds"] = {"median": statistics.median(totals), "min": min(totals), "max": max(totals)}
    return metrics


stream = next(r for r in results if r["mode"] == "plan-binary")
summary = {
    "old_binary_with_sha": aggregate([old]),
    "stream_no_sha": aggregate([stream]),
    "verifier_speedup_stream": old["verify_seconds"] / stream["verify_seconds"],
    "parallel": {str(t): aggregate([r for r in results if r["mode"] == "plan-binary-parallel" and r["threads"] == t])
                 for t in (1, 2, 4, 8)},
    "all_fields_equal": True, "source_hashing": False, "blob_payloads_read": 0,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
