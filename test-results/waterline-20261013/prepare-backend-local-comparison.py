#!/usr/bin/env python3
"""Compare window 1/8 and CPU/GPU/Auto on one existing lazy40 small plan.

Window 1 is a one-group Waterline approximation to demand preparation. It is
neither the old eager executor nor a guarantee of absolutely no prefetch.
The source plan, fixture, bundle, math, placement, and foreground order stay
fixed. Every process checks the independent fixture, and Python separately
checks all 49,152 decoded complex slots at the fixed acceptance tolerances.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ATOL, RTOL = 0.001, 0.0001
EXPECTED_SLOTS = 49152


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical_plan(plan):
    value = json.loads(json.dumps(plan))
    value["waterline"].pop("lookahead_groups")
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def independent_check(decoded, fixture, fixture_hash):
    if decoded["fixture_sha256"] != fixture_hash:
        raise RuntimeError("decoded fixture hash differs from the frozen input")
    rows, expected = decoded["outputs"], fixture["expected_outputs"]
    slots = fixture["context"]["poly_degree"] // 2
    if len(rows) != len(expected) or decoded["slots_per_output"] != slots:
        raise RuntimeError("decoded output shape differs from the fixture")
    result = dict(slots_checked=0, nonfinite_slots=0, bad_real_slots=0,
                  bad_imaginary_slots=0, zero_reference_slots=0,
                  max_abs_error=0.0, max_imaginary=0.0,
                  max_abs_on_zero_reference=0.0)
    square_error = 0.0
    for actual, reference in zip(rows, expected):
        if actual["name"] != reference["name"] or actual["value_id"] != reference["value_id"]:
            raise RuntimeError("decoded output identity/order differs from the fixture")
        values = reference["values"]
        if len(values) != slots or actual["prefix_size"] != slots:
            raise RuntimeError("this comparison requires a full-slot independent fixture")
        if len(actual["real"]) != slots or len(actual["imaginary"]) != slots:
            raise RuntimeError("decoded output omitted physical slots")
        for real, imaginary, expected_value in zip(actual["real"], actual["imaginary"], values):
            result["slots_checked"] += 1
            if not all(math.isfinite(v) for v in (real, imaginary, expected_value)):
                result["nonfinite_slots"] += 1
                continue
            error = abs(real - expected_value)
            result["bad_real_slots"] += error > ATOL + RTOL * abs(expected_value)
            result["bad_imaginary_slots"] += abs(imaginary) > ATOL
            result["max_abs_error"] = max(result["max_abs_error"], error)
            result["max_imaginary"] = max(result["max_imaginary"], abs(imaginary))
            if expected_value == 0:
                result["zero_reference_slots"] += 1
                result["max_abs_on_zero_reference"] = max(result["max_abs_on_zero_reference"], error)
            square_error += error * error
    if result["slots_checked"] != EXPECTED_SLOTS:
        raise RuntimeError("independent check did not cover all 49,152 complex slots")
    result["rms_real_error"] = math.sqrt(square_error / EXPECTED_SLOTS)
    result["passed"] = not any(result[k] for k in
                               ("nonfinite_slots", "bad_real_slots", "bad_imaginary_slots"))
    return result


def metrics(report):
    timing, prepare = report["runtime_timing"], report["preparation_measurements"]
    devices = report["device_measurements"]
    return dict(
        forward_runtime_ms=report["runtime_ms"],
        setup_ms=timing["setup_ms"], initialization_ms=timing["initialization_ms"],
        online_execution_ms=timing["online_execution_ms"],
        consumer_wait_ms=sum(d["consumer_wait_ms"] for d in devices),
        consumer_wait_calls=sum(d["consumer_wait_calls"] for d in devices),
        cpu_prepare_calls=prepare["cpu_prepare_calls"], cpu_encode_ms=prepare["cpu_encode_ms"],
        gpu_prepare_calls=sum(d["gpu_encode_calls"] for d in devices),
        gpu_encode_ms=sum(d["gpu_encode_ms"] for d in devices),
        cpu_plain_h2d_bytes=prepare["cpu_plain_h2d_bytes"],
        transport_h2d_bytes=prepare["transport_h2d_bytes"],
        transport_h2d_ms=prepare["transport_h2d_ms"],
        gpu_raw_h2d_bytes=sum(d["gpu_raw_h2d_bytes"] for d in devices),
        gpu_raw_h2d_ms=sum(d["gpu_raw_h2d_ms"] for d in devices),
        host_boot_calls=timing["boot_calls"], host_boot_ms=timing["host_boot_ms"],
        bundle_read_calls=timing["bundle_read_calls"], bundle_read_bytes=timing["bundle_read_bytes"],
        bundle_read_ms=timing["bundle_read_ms"], raw_peak_bytes=timing["raw_peak_bytes"],
        host_max_rss_kib=report["host_max_rss_kib"],
        allocated_peak_bytes=devices[0]["allocated_peak_bytes"],
        allocated_live_after_plan_bytes=devices[0]["allocated_live_bytes"],
        pool_reserved_bytes=devices[0]["pool_reserved_bytes"],
        pool_capacity_bytes=devices[0]["pool_capacity_bytes"],
        pinned_peak_after_plan_bytes=report["pinned_transfer_peak_after_plan_bytes"],
        pinned_live_after_plan_bytes=report["pinned_transfer_live_after_plan_bytes"],
        pinned_live_after_validation_bytes=report["pinned_transfer_live_after_validation_bytes"],
        device_allocators=[{k: d[k] for k in ("logical_device", "allocated_live_bytes",
            "allocated_peak_bytes", "pool_reserved_bytes", "pool_capacity_bytes")} for d in devices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True,
                        help="Existing lazy40 production-primitives graph, fixture and bundle")
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local build artifact directory")
    parser.add_argument("--summary", type=Path, required=True, help="New compact evidence JSON")
    parser.add_argument("--log", type=Path, required=True, help="New compact execution log")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runner-scope", required=True, help="Exact build/version scope of the supplied runner")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for key in ("source_directory", "runner", "output", "summary", "log"):
        setattr(args, key, getattr(args, key).resolve())
    for path in (args.output, args.summary, args.log):
        if path.exists():
            raise FileExistsError(f"new output required: {path}")
    sources = list(args.source_directory.glob("*.runtime-plan.json"))
    if len(sources) != 1:
        raise RuntimeError("exactly one source JSON plan is required")
    source = sources[0]
    fixture_path = args.source_directory / "fixture.json"
    spec = args.source_directory / "operator-spec.json"
    bundle = Path(str(source).removesuffix(".runtime-plan.json") + ".bundle")
    original, fixture = json.loads(source.read_text()), json.loads(fixture_path.read_text())
    if "waterline" in fixture:
        raise RuntimeError("fixture must not override the plan's Waterline configuration")
    if fixture["context"]["default_scale_log2"] != 40 or fixture["tolerances"] != {"atol": ATOL, "rtol": RTOL}:
        raise RuntimeError("the fixed lazy40 context and acceptance tolerances are required")
    if fixture["model"].get("complete_qwen") or not fixture["model"].get("synthetic_weights"):
        raise RuntimeError("this study accepts only the existing synthetic small graph")
    if sum(len(row["values"]) for row in fixture["expected_outputs"]) != EXPECTED_SLOTS:
        raise RuntimeError("the independent fixture must contain all 49,152 physical slots")
    phases = ("initialization", "execution", "finalization")
    operations = [op for phase in phases for op in original[phase]]
    prepares = sum(op["kind"] == "prepare_plaintext" for op in operations)
    if len(original["target"]["device_counts"]) != 1 or original["target"]["device_counts"][0] != 1:
        raise RuntimeError("only the existing single-device graph is allowed")
    if len(operations) > 10000 or prepares != 47:
        raise RuntimeError("unexpected graph size or Prepare count; no large model may run")
    inputs = [source, fixture_path, spec, args.runner, Path(__file__).resolve()]
    inputs.extend(sorted(p for p in bundle.rglob("*") if p.is_file()))
    build = args.runner.parent.parent
    for name in ("CMakeCache.txt", "libposeidon_shared.so"):
        path = build / name
        if path.exists():
            inputs.append(path)
    for name in ("flags.make", "link.txt"):
        path = build / "src/poseidon/runtime_api/CMakeFiles/poseidon_runtime_gpu_qwen_e2e.dir" / name
        if path.exists():
            inputs.append(path)
    bindings = {str(path): digest(path) for path in inputs}
    args.output.mkdir(parents=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    unchanged = canonical_plan(original)
    plans = {}
    for window in (1, 8):
        plan = json.loads(json.dumps(original))
        plan["waterline"]["lookahead_groups"] = window
        if canonical_plan(plan) != unchanged:
            raise RuntimeError("a plan change altered foreground order or other configuration")
        plans[window] = args.output / f"lookahead-{window}.runtime-plan.json"
        save(plans[window], plan)
    summary = dict(
        passed=False, scope="Local one-GPU synthetic lazy40 production-primitives comparison; not full Qwen or multi-GPU performance",
        window1_scope="One-group Waterline window approximating demand preparation; not old eager and not absolute zero prefetch",
        fixed_graph_scope="One existing compiled/placed plan; all foreground operations and order fixed; only lookahead_groups and runtime Prepare backend vary",
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        root_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        runner_scope=args.runner_scope, source_bindings=bindings,
        build_binding_scope="Runner and available CMake/shared-library artifacts are hashed; supplied runner-scope states the static build version",
        source_plan=str(source), artifact_directory=str(args.output),
        unchanged_plan_sha256="sha256:" + hashlib.sha256(unchanged).hexdigest(),
        variant_plan_sha256={str(w): digest(p) for w, p in plans.items()},
        tolerances=dict(atol=ATOL, rtol=RTOL), expected_complex_slots=EXPECTED_SLOTS,
        backend_modes=["cpu", "gpu", "auto"], lookahead_groups=[1, 8], repeats=args.repeats,
        instruction_counts=dict(Counter(op["kind"] for op in operations)),
        watermark=original["waterline"], fixture_model=fixture["model"],
        timing_scope="Separate processes, no trace/profiler. runtime_ms is runtime.run wall time; setup/init/online are also separate. Process wall includes startup and validation.",
        metric_scope="Consumer waits and GPU events are sums, not wall-time fractions. CPU plaintext H2D is included in transport bytes; GPU raw H2D is separate. Allocated peak is managed allocator live peak, reserved/cap are separate; RSS includes keys/context/graph outside the Host budget.",
        samples=[], statistics={})
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    summary["environment"] = {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for key in ("POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB", "POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB"):
        if key in environment:
            summary["environment"][key] = environment[key]
    save(args.summary, summary)
    with args.log.open("x") as log:
        def note(text):
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()
        modes = ((1, "cpu"), (8, "cpu"), (1, "gpu"), (8, "gpu"), (1, "auto"), (8, "auto"))
        for repeat in range(args.repeats):
            shift = repeat % len(modes)
            for window, backend in modes[shift:] + modes[:shift]:
                folder = args.output / f"repeat-{repeat}-w{window}-{backend}"
                folder.mkdir()
                command = [str(args.runner), str(plans[window]), str(spec), str(fixture_path),
                           str(folder / "report.json"), str(folder / "decoded.json"),
                           "--bundle", str(bundle), "--backend", backend]
                save(folder / "command.json", command)
                started = time.monotonic()
                with (folder / "run.log").open("w") as run_log:
                    run = subprocess.run(command, env=environment, stdout=run_log, stderr=subprocess.STDOUT)
                wall = (time.monotonic() - started) * 1000
                report = json.loads((folder / "report.json").read_text())
                if run.returncode or not report.get("passed"):
                    note(f"FAIL repeat={repeat} window={window} backend={backend} exit={run.returncode} artifacts={folder}")
                    raise RuntimeError("runner execution or numerical check failed")
                if report["waterline"]["lookahead_groups"] != window or report["waterline"]["backend"] != backend:
                    raise RuntimeError("runner configuration differs from the comparison")
                if report["device_worker_count"] != 1 or report["tolerances"] != {"atol": ATOL, "rtol": RTOL}:
                    raise RuntimeError("runner device count or tolerance changed")
                comparison = independent_check(json.loads((folder / "decoded.json").read_text()), fixture, bindings[str(fixture_path)])
                measured = metrics(report)
                if not comparison["passed"] or report["comparison"]["slots_checked"] != EXPECTED_SLOTS:
                    raise RuntimeError("independent all-slot check failed")
                if measured["cpu_prepare_calls"] + measured["gpu_prepare_calls"] != prepares:
                    raise RuntimeError("Prepare count changed")
                if (backend == "gpu" and measured["cpu_prepare_calls"]) or (backend == "cpu" and measured["gpu_prepare_calls"]):
                    raise RuntimeError("fixed backend silently used the other backend")
                if measured["host_boot_calls"] != 5 or report["runtime_timing"]["compute_calls"] != 178:
                    raise RuntimeError("the fixed foreground graph changed")
                if measured["pinned_live_after_validation_bytes"]:
                    raise RuntimeError("run retained pinned storage")
                summary["samples"].append(dict(repeat=repeat, window=window, backend=backend,
                    passed=True, process_wall_ms=wall, metrics=measured,
                    independent_comparison=comparison, runner_comparison=report["comparison"],
                    report_sha256=digest(folder / "report.json"), decoded_sha256=digest(folder / "decoded.json")))
                summary.setdefault("devices", report["devices"])
                save(args.summary, summary)
                note(f"PASS repeat={repeat} window={window} backend={backend} forward_ms={measured['forward_runtime_ms']:.3f} wait_ms={measured['consumer_wait_ms']:.3f} cpu_prepare={measured['cpu_prepare_calls']} gpu_prepare={measured['gpu_prepare_calls']} slots={comparison['slots_checked']} max_abs={comparison['max_abs_error']:.9g}")
        for window, backend in modes:
            rows = [s for s in summary["samples"] if s["window"] == window and s["backend"] == backend]
            stats = {}
            for name, value in rows[0]["metrics"].items():
                if isinstance(value, (int, float)):
                    numbers = [s["metrics"][name] for s in rows]
                    stats[name] = dict(median=statistics.median(numbers), minimum=min(numbers), maximum=max(numbers))
            summary["statistics"][f"w{window}-{backend}"] = stats
        for path, expected in bindings.items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"shared build/input changed during comparison: {path}")
        summary["all_build_inputs_unchanged"] = True
        summary["passed"] = True
        summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary["numeric_max_abs_error"] = max(s["independent_comparison"]["max_abs_error"] for s in summary["samples"])
        save(args.summary, summary)
        note("ALL_CONFIGURATIONS_PASSED " + json.dumps({key: value["forward_runtime_ms"] for key, value in summary["statistics"].items()}))


if __name__ == "__main__":
    main()
