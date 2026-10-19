#!/usr/bin/env python3
"""Combine independently verified host results; retain superseded pilot evidence."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--server", action="store_true")
args = parser.parse_args()
out = Path(__file__).resolve().parent
initial = json.loads((out / "local-initial-comparison/comparison.json").read_text())
repeated = json.loads((out / "local-mlp-repeated-analysis/comparison.json").read_text())
sources = [initial, repeated]
rows = [r for r in initial["comparisons"] if r["group"] != "local-mlp"] + repeated["comparisons"]
if args.server:
    server = json.loads((out / "server-analysis/comparison.json").read_text())
    sources.append(server)
    rows += server["comparisons"]
group_order = ["local-mlp", "local-qproj", "local-pressure", "server-mlp1", "server-mlp2", "server-pressure"]
name_order = ["legacy-eager", "unfixed-eager", "unfixed-waterline", "current-eager", "current-waterline",
              "unfixed-auto-w8", "current-auto-w8"]
rows.sort(key=lambda r: (group_order.index(r["group"]), name_order.index(r["name"])))
result = dict(scope="Historical Runtime headers with the same current GPU backend on each host; not a full historical repository rebuild. Three fresh-process samples per configuration, except the final RTX 4060 MLP comparison uses a separate seven-repeat run. The initial three-repeat MLP samples are retained as pilot evidence and excluded from final MLP medians. Eager online excludes upfront preparation; Waterline online includes runtime preparation. All timings include final synchronization/drain. Keys and input encryption are excluded.",
              verified_samples_total=sum(s["verified_samples"] for s in sources),
              timing_samples_used=sum(r["repeats"] for r in rows),
              excluded_pilot_samples=15, comparisons=rows,
              identities=[i for s in sources for i in s["identities"]])
lookup = {(r["group"], r["name"]): r["online_ms"]["median"] for r in rows}
result["ratios"] = {}
for group in dict.fromkeys(r["group"] for r in rows):
    original = lookup[group, "legacy-eager"]
    result["ratios"][group] = dict(
        current_eager_over_legacy_eager=lookup[group, "current-eager"] / original,
        unfixed_eager_over_legacy_eager=lookup[group, "unfixed-eager"] / original,
        current_waterline_over_legacy_eager=lookup[group, "current-waterline"] / original,
        current_waterline_over_current_eager=lookup[group, "current-waterline"] / lookup[group, "current-eager"],
        waterline_speedup=lookup[group, "unfixed-waterline"] / lookup[group, "current-waterline"])
assert result["verified_samples_total"] == result["timing_samples_used"] + result["excluded_pilot_samples"]
(out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
verified = []
analyses = ["local-initial-comparison", "local-mlp-repeated-analysis"]
if args.server:
    analyses.append("server-analysis")
for name in analyses:
    evidence = json.loads((out / name / "verification.json").read_text())
    for sample in evidence["samples"]:
        verified.append(dict(sample, analysis=name,
            used_in_final_medians=not (name == "local-initial-comparison" and sample["group"] == "local-mlp")))
assert len(verified) == result["verified_samples_total"]
assert sum(s["used_in_final_medians"] for s in verified) == result["timing_samples_used"]
(out / "verification.json").write_text(json.dumps(dict(
    verified_unprofiled_samples=len(verified), samples=verified,
    profiled_validation_samples=3, profile_evidence="nsys-mlp.json"), indent=2) + "\n")
lines = ["Seconds, median (min-max). RTX 4060 MLP n=7, others n=3.", "workload | configuration | online | forward"]
for row in rows:
    value = lambda metric: f"{row[metric]['median']/1000:.6f} ({row[metric]['min']/1000:.6f}-{row[metric]['max']/1000:.6f})"
    lines.append(f"{row['group']} | {row['name']} | {value('online_ms')} | {value('forward_ms')}")
(out / "measurements.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
