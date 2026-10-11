#!/usr/bin/env python3
"""Run GPU Encode cases serially; save full samples and a compact summary."""
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--pool-mb", type=int, default=64)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.repeat < 1 or not 0 <= args.pool_mb <= 4096:
        parser.error("invalid measurement counts")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", POSEIDON_GPU_ENCODE_POOL_MB=str(args.pool_mb))
    cases = list(itertools.product((8, 32), (1, 8, 32)))
    random.Random(20261011).shuffle(cases)
    reports = []
    telemetry = []
    for run in range(args.runs):
        for limbs, batch in (cases if run % 2 == 0 else list(reversed(cases))):
            name = f"q{limbs}-b{batch}-run{run + 1}"
            report = args.output / f"{name}.json"
            if report.exists():
                raise FileExistsError(report)
            print(name, flush=True)
            telemetry.append({"case": name, "before": subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,pstate,power.draw,clocks.sm,clocks.mem,memory.used",
                "--format=csv,noheader"], text=True).strip()})
            command = [str(args.binary.resolve()), str(limbs), str(batch), "40",
                       str(args.warmup), str(args.repeat), str(report.resolve())]
            env["POSEIDON_GPU_ENCODE_ORDER_SEED"] = str(20261011 + run * 10000 + limbs * 100 + batch)
            with (args.output / f"{name}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=300)
            reports.append(json.loads(report.read_text()))
    summary = {"gpu": reports[0]["gpu"], "degree": 65536, "scale_log2": 40,
               "initial_pool_mb": args.pool_mb,
               "runs": args.runs, "warmup": args.warmup, "repeat": args.repeat,
               "case_order": telemetry, "rows": []}
    for limbs, batch in itertools.product((8, 32), (1, 8, 32)):
        group = [p for p in reports if p["q_limbs"] == limbs and p["batch"] == batch]
        row = {"q_limbs": limbs, "batch": batch,
               "cpu_encode_ms_per_plaintext": statistics.median(
                   p["cpu_encode_only"]["median_ms"] / batch for p in group)}
        backend_names = sorted({m["backend"] for p in group for m in p["modes"]})
        for backend, submission in itertools.product(backend_names, ("individual", "batched")):
            modes = [m for p in group for m in p["modes"]
                     if m["backend"] == backend and m["submission"] == submission]
            if not modes:
                row[f"{backend}_{submission}"] = {"skipped": True}
                continue
            row[f"{backend}_{submission}"] = {
                "device_ms_per_plaintext": statistics.median(m["device_ms_per_plaintext"] for m in modes),
                "raw_h2d_encode_ms_per_plaintext": statistics.median(
                    m["with_raw_h2d"]["total"]["median_ms"] / batch for m in modes),
                "ntt_ms_per_plaintext": statistics.median(
                    m["device_only"]["ntt"]["median_ms"] / batch for m in modes),
                "prepare_fft_rns_ms_per_plaintext": statistics.median(
                    m["device_only"]["prepare_fft_rns"]["median_ms"] / batch for m in modes),
                "max_input_error": max(m["correctness"]["max_input_error"] for m in modes),
                "cpu_ntt_residue_differences": max(m["correctness"]["cpu_ntt_residue_differences"] for m in modes),
            }
        row["cuda_batch_speedup"] = (row["cuda_ntt_individual"]["device_ms_per_plaintext"] /
                                     row["cuda_ntt_batched"]["device_ms_per_plaintext"])
        summary["rows"].append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
