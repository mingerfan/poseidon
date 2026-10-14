#!/usr/bin/env python3
"""Local encrypted lookahead sweep: unchanged foreground, GPU Prepare only.

Makes new JSON plan copies whose only changed field is lookahead_groups.
Uses existing independently calculated all-slot fixtures and the real runner.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def unchanged_plan(plan):
    copy = json.loads(json.dumps(plan))
    copy["waterline"].pop("lookahead_groups")
    return json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-directory", type=Path, default=ROOT / "build-waterline-sync/scale40-prescale-composite-regression-v6/scale40")
    p.add_argument("--runner", type=Path, default=ROOT / "build-runtime-gpu-api-release/bin/poseidon_runtime_gpu_qwen_e2e")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    args.source_directory = args.source_directory.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = list(args.source_directory.glob("*.runtime-plan.json"))
    if len(sources) != 1:
        raise RuntimeError("exactly one source JSON plan required")
    original = json.loads(sources[0].read_text())
    fixture = args.source_directory / "fixture.json"
    if "waterline" in json.loads(fixture.read_text()):
        raise RuntimeError("fixture overrides plan Waterline configuration")
    spec = args.source_directory / "operator-spec.json"
    if not spec.exists():
        spec = args.source_directory.parent / "profiles/operator-spec.json"
    bundle = Path(str(sources[0]).removesuffix(".runtime-plan.json") + ".bundle")
    canonical = unchanged_plan(original)
    plans = {}
    for window in (8, 32, 64):
        copy = json.loads(json.dumps(original))
        copy["waterline"]["lookahead_groups"] = window
        if unchanged_plan(copy) != canonical:
            raise RuntimeError("lookahead copy changed the foreground or other configuration")
        plans[window] = args.output / f"lookahead-{window}.runtime-plan.json"
        save(plans[window], copy)
    all_ops = original["initialization"] + original["execution"] + original["finalization"]
    op_by_ordinal = {int(op["ordinal"]): op for op in all_ops}
    summary = dict(scope="local one-GPU encrypted lookahead sweep; no full-model or multi-GPU performance claim",
        source_plan=str(sources[0]), source_plan_sha256=digest(sources[0]), fixture_sha256=digest(fixture),
        operator_spec_sha256=digest(spec), runner_sha256=digest(args.runner),
        script_sha256=digest(Path(__file__)), only_mutation="waterline.lookahead_groups",
        unchanged_plan_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        backend="gpu", cuda_visible_devices="0", repeats=args.repeats,
        plan_instruction_counts=dict(Counter(o["kind"] for o in all_ops)),
        samples=[], traced_samples=[], traced_runs_excluded_from_timing_summary=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def execute(window, label, traced=False):
        folder = args.output / label
        folder.mkdir()
        command = [str(args.runner), str(plans[window]), str(spec), str(fixture),
            str(folder / "report.json"), str(folder / "decoded.json"), "--bundle", str(bundle), "--backend", "gpu"]
        if traced:
            command += ["--trace", str(folder / "trace")]
        save(folder / "command.json", command)
        started = time.monotonic()
        with (folder / "run.log").open("w") as log:
            run = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        process_wall_ms = (time.monotonic() - started) * 1000
        report = json.loads((folder / "report.json").read_text())
        if run.returncode or not report["passed"]:
            raise RuntimeError(f"encrypted execution failed: {folder / 'run.log'}")
        if report["waterline"]["lookahead_groups"] != window or report["waterline"]["backend"] != "gpu":
            raise RuntimeError("runner configuration differs from requested sweep")
        if report["preparation_measurements"]["cpu_prepare_calls"]:
            raise RuntimeError("GPU-only run unexpectedly prepared plaintext on CPU")
        sample = dict(window=window, label=label, passed=True, process_wall_ms=process_wall_ms,
            runtime_ms=report["runtime_ms"], runtime_timing=report["runtime_timing"],
            comparison=report["comparison"], preparation_measurements=report["preparation_measurements"],
            device_measurements=report["device_measurements"])
        if traced:
            trace = folder / "trace.rank0.csv"
            with trace.open() as stream:
                rows = list(csv.DictReader(stream))
            admits = Counter(op_by_ordinal[int(row["ordinal"])]["kind"] for row in rows if row["event"] == "admit")
            burst = peak_burst = 0
            for row in rows:
                if row["event"] == "admit":
                    burst += 1
                    peak_burst = max(peak_burst, burst)
                else:
                    burst = 0
            sample["trace"] = dict(events=dict(Counter(row["event"] for row in rows)),
                admitted_instruction_kinds=dict(admits), max_consecutive_admit_events=peak_burst,
                caution="admit bursts are trace ordering, not a measured active-group peak or GPU overlap")
        print(f"PASS window={window} label={label} runtime_ms={report['runtime_ms']:.3f}", flush=True)
        return sample

    # Rotate order to reduce drift from always measuring one window first.
    for repeat in range(args.repeats):
        order = (8, 32, 64)
        shift = repeat % 3
        order = order[shift:] + order[:shift]
        for window in order:
            summary["samples"].append(execute(window, f"repeat-{repeat}-w{window}"))
            save(args.output / "summary.json", summary)
    for window in (8, 32, 64):
        summary["traced_samples"].append(execute(window, f"trace-w{window}", True))
    summary["statistics"] = {}
    for window in (8, 32, 64):
        rows = [s for s in summary["samples"] if s["window"] == window]
        numbers = [s["runtime_ms"] for s in rows]
        summary["statistics"][str(window)] = dict(runtime_ms_median=statistics.median(numbers),
            runtime_ms_min=min(numbers), runtime_ms_max=max(numbers),
            online_ms_median=statistics.median(s["runtime_timing"]["online_execution_ms"] for s in rows),
            host_boot_ms_median=statistics.median(s["runtime_timing"]["host_boot_ms"] for s in rows),
            consumer_wait_ms_median=statistics.median(sum(d["consumer_wait_ms"] for d in s["device_measurements"]) for s in rows),
            gpu_encode_ms_median=statistics.median(sum(d["gpu_encode_ms"] for d in s["device_measurements"]) for s in rows))
    if digest(args.runner) != summary["runner_sha256"]:
        raise RuntimeError("runner changed during sweep")
    summary["passed"] = True
    save(args.output / "summary.json", summary)
    print(json.dumps(summary["statistics"]), flush=True)


if __name__ == "__main__":
    main()
