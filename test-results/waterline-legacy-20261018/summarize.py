#!/usr/bin/env python3
"""Check identities, numerics and resource bounds before reporting medians."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, required=True)
parser.add_argument("--additional-runs", type=Path, action="append", default=[])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
jobs = [(runs, job) for runs in [args.runs] + args.additional_runs
        for job in json.loads((runs / "jobs.json").read_text())]
rows, verified, inputs = [], [], []
identities = {}
for runs, job in jobs:
    suite = Path(job["suite"])
    definition = json.loads((suite / "suite.json").read_text())
    case = next(c for c in definition["cases"] if c["name"] == job["case"])
    plan_path = suite / case["plan"]
    plan = json.loads(plan_path.read_text())
    operations = [op for phase in ("initialization", "execution", "finalization") for op in plan[phase]]
    compute = [{k: v for k, v in op.items() if k != "ordinal"} for op in operations if op["kind"] == "compute"]
    key = job["group"]
    bundle = suite / "bundle"
    bundle_files = [(str(p.relative_to(bundle)), sha(p)) for p in sorted(bundle.rglob("*")) if p.is_file()]
    assert bundle_files
    identity = {"compute_sha256": digest(compute), "spec_sha256": sha(suite / "operator-spec.json"),
                "fixture_sha256": sha(suite / "fixture.json"), "weights_sha256": digest(bundle_files)}
    if key in identities:
        assert identity == identities[key], (job, identity, identities[key])
    identities[key] = identity
    assert job["variant"] != "legacy" or "waterline" not in plan
    inputs.append(dict(group=key, name=job["name"], plan_path=str(plan_path),
                       plan_sha256=sha(plan_path), **identity))
    reports, samples = [], []
    for repeat in range(args.repeats):
        folder = runs / f"{key}-r{repeat}-{job['name']}"
        summary = json.loads((folder / "summary.json").read_text())
        assert len(summary["samples"]) == 1
        sample = summary["samples"][0]
        assert sample["passed"] and sample["exit_code"] == 0 and sample["report"]["passed"], folder
        assert summary["selected_case_names"] == [job["case"]]
        assert summary["cuda"] == job["cuda"] and summary["pool_mib"] == job["pool_mib"]
        report = sample["report"]
        assert report["runtime_timing"]["compute_calls"] == len(compute)
        if case["runner"] == "qwen":
            assert report["comparison"]["slots_checked"] == 8192
            assert report["comparison"]["max_abs_error"] < 1e-6
            assert report["pinned_transfer_live_after_plan_bytes"] == 0
            assert report["pinned_transfer_live_after_validation_bytes"] == 0
            for device in report["device_measurements"]:
                if device["managed_pool"]:
                    assert device["allocated_peak_bytes"] <= device["pool_capacity_bytes"]
            if "waterline" in plan:
                count = sum(op["kind"] == "prepare_plaintext" for op in operations)
                prepared = report["preparation_measurements"]["cpu_prepare_calls"] + sum(d["gpu_encode_calls"] for d in report["device_measurements"])
                assert prepared == count
                assert report["runtime_timing"]["bundle_read_calls"] == count
        else:
            assert report["python_comparison"]["max_abs"] < 1e-4
            assert report["mock_comparison"]["max_abs"] < 1e-4
        log = (folder / ("r0-" + job["case"]) / "run.log").read_text()
        if "waterline" in plan:
            config = plan["waterline"]
            peaks = re.findall(r"waterline GPU(\d+) front_peak=(\d+) plain_peak=(\d+) remaining_front=(\d+) remaining_plain=(\d+)", log)
            assert len(peaks) == definition["devices"]
            for device, front, plain, remaining_front, remaining_plain in peaks:
                d = int(device)
                assert int(remaining_plain) == 0
                assert int(plain) <= int(config["device_fractions"][d] * config["device_capacity_bytes"][d])
                assert int(front) <= config["device_capacity_bytes"][d] - config["device_reserve_bytes"][d]
            shared = re.search(r"waterline shared host_peak=(\d+) pinned_peak=(\d+) raw_peak=(\d+)", log)
            assert shared
            assert all(int(value) <= config[budget] for value, budget in zip(shared.groups(),
                       ("host_budget_bytes", "pinned_budget_bytes", "raw_budget_bytes")))
        verified.append(dict(group=key, name=job["name"], repeat=repeat, passed=True))
        reports.append(report)
        samples.append(sample)
    for field in ("plan_sha256", "plan_source_sha256", "operator_spec_sha256", "fixture_sha256", "model_sha256"):
        if field in reports[0]:
            assert len({r[field] for r in reports}) == 1
    rows.append(dict(group=key, name=job["name"], variant=job["variant"], case=job["case"],
                     repeats=len(samples), **{metric: dict(
                         median=statistics.median(s[metric] for s in samples),
                         min=min(s[metric] for s in samples), max=max(s[metric] for s in samples),
                         samples=[s[metric] for s in samples])
                         for metric in ("online_ms", "forward_ms", "process_cpu_seconds")}))

result = dict(scope=f"Exact historical Runtime header and shared current GPU backend; not a whole-repository historical rebuild. Core configurations rotated; supplemental unfixed Waterline configurations measured in a separate rotating batch. Fresh process; identical compute instructions, spec, fixture and raw weights. Each configuration has {args.repeats} repeats. Online includes final synchronization/drain, excludes eager preparation; forward includes preparation. Key generation and input encryption are excluded.",
              verified_samples=len(verified), comparisons=rows, identities=inputs)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
(args.output / "verification.json").write_text(json.dumps(dict(samples=verified, identities=identities), indent=2) + "\n")
lines = [f"Seconds, medians (min-max); {args.repeats} repeats each.", "workload | configuration | online | forward"]
for row in rows:
    value = lambda metric: f"{row[metric]['median']/1000:.6f} ({row[metric]['min']/1000:.6f}-{row[metric]['max']/1000:.6f})"
    lines.append(f"{row['group']} | {row['name']} | {value('online_ms')} | {value('forward_ms')}")
(args.output / "measurements.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
