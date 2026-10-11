#!/usr/bin/env python3
"""Serial same-GPU interference tests; compare paired operator baselines."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--burst", type=int, default=4)
    parser.add_argument("--raw-h2d", type=int, choices=(0, 1), default=1)
    parser.add_argument("--demand-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.repeat < 1 or not 1 <= args.burst <= 64:
        parser.error("invalid measurement counts")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(itertools.product((8, 32), (1, 8)))
    random.Random(20261011).shuffle(cases)
    reports, telemetry = [], []
    for run in range(args.runs):
        for limbs, batch in cases if run % 2 == 0 else reversed(cases):
            name = f"q{limbs}-b{batch}-run{run+1}"
            path = args.output / f"{name}.json"
            if path.exists():
                raise FileExistsError(path)
            print(name, flush=True)
            telemetry.append({"case": name, "before": subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,pstate,power.draw,clocks.sm,clocks.mem,memory.used",
                "--format=csv,noheader"], text=True).strip()})
            command = [str(args.binary.resolve()), str(limbs), str(batch), str(args.repeat),
                       str(args.burst), str(args.raw_h2d), str(path.resolve())]
            with (args.output / f"{name}.log").open("w") as log:
                subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1",
                    POSEIDON_GPU_ENCODE_DEMAND_ONLY="1" if args.demand_only else "0"),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            reports.append(json.loads(path.read_text()))
    summary = {"gpu": reports[0]["gpu"], "degree": 65536, "p_limbs": 4,
               "runs": args.runs, "repeat": args.repeat, "burst": args.burst,
               "raw_h2d": bool(args.raw_h2d), "case_order": telemetry, "rows": []}
    summary["demand_only"] = args.demand_only
    for limbs, batch in itertools.product((8, 32), (1, 8)):
        group = [p for p in reports if p["q_limbs"] == limbs and p["batch"] == batch]
        for op in sorted({r["operator"] for p in group for r in p["rows"]}):
            rows = [r for p in group for r in p["rows"] if r["operator"] == op]
            row = {"q_limbs": limbs, "batch": batch, "operator": op,
                   "baseline_ms": statistics.median(r["baseline_median_ms"] for r in rows),
                   "baseline_mean_ms": statistics.median(r["baseline_mean_ms"] for r in rows),
                   "backgrounds": {}}
            for name in sorted({name for r in rows for name in r["backgrounds"]}):
                backgrounds = [r["backgrounds"][name] for r in rows]
                row["backgrounds"][name] = {
                    "operator_ms": statistics.median(b["operator"]["median_ms"] for b in backgrounds),
                    "operator_slowdown": statistics.median(b["operator_slowdown"] for b in backgrounds),
                    "operator_mean_slowdown": statistics.median(b["operator_mean_slowdown"] for b in backgrounds),
                    "encode_ms_per_plaintext": statistics.median(b["encode_batch_latency"]["median_ms_per_plaintext"] for b in backgrounds),
                    "encoded_plaintexts_per_operator": statistics.median(b["encoded_plaintexts_per_operator"] for b in backgrounds),
                    "encode_plaintexts_per_second_in_window": statistics.median(b["encode_plaintexts_per_second_in_window"] for b in backgrounds),
                    "exact_match": all(b["operator_and_encode_exact_match"] for b in backgrounds),
                }
            summary["rows"].append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
