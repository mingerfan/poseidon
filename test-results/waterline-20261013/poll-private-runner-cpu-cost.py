#!/usr/bin/env python3
"""One extra encrypted run per private polling variant, with process CPU cost."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-output", type=Path, required=True)
    args = p.parse_args()
    base = args.sweep_output.resolve()
    source = json.loads((base / "summary.json").read_text())
    output = base / "cpu-cost"
    output.mkdir(exist_ok=False)
    result = dict(scope="Whole child process CPU cost, including context/key generation, input encryption, forward and validation; not forward-only CPU time",
        source_summary_sha256=digest(base / "summary.json"), script_sha256=digest(Path(__file__)), samples=[])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for variant in ("100us", "10us", "yield"):
        folder = output / variant
        folder.mkdir()
        command = json.loads((base / f"repeat-0-{variant}/command.json").read_text())
        if digest(Path(command[0])) != source["variants"][variant]["binary_sha256"]:
            raise RuntimeError("private variant binary changed")
        command[4], command[5] = str(folder / "report.json"), str(folder / "decoded.json")
        (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        with (folder / "run.log").open("w") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        wall = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        user, system = after.ru_utime-before.ru_utime, after.ru_stime-before.ru_stime
        report = json.loads((folder / "report.json").read_text())
        if run.returncode or not report["passed"] or report["comparison"]["slots_checked"] != 8192:
            raise RuntimeError("CPU-cost numerical regression")
        result["samples"].append(dict(variant=variant, user_cpu_seconds=user, system_cpu_seconds=system,
            total_cpu_seconds=user+system, process_wall_seconds=wall, average_process_cpu_cores=(user+system)/wall,
            forward_runtime_ms=report["runtime_ms"], passed=True, comparison=report["comparison"]))
        print(f"PASS {variant} user={user:.6f}s system={system:.6f}s wall={wall:.6f}s forward={report['runtime_ms']:.3f}ms", flush=True)
    for path, expected in source["source_bindings"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError("shared experiment input changed")
    result["passed"] = True
    (base / "cpu-cost.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
