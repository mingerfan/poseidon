#!/usr/bin/env python3
"""Rotate exact benchmark configurations, with a fresh process per sample."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=Path, required=True)
parser.add_argument("--jobs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("waterline_benchmark", args.benchmark.resolve())
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
jobs = json.loads(args.jobs.read_text())
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
for group in dict.fromkeys(j["group"] for j in jobs):
    configurations = [j for j in jobs if j["group"] == group]
    for repeat in range(args.repeats):
        shift = repeat % len(configurations)
        for job in configurations[shift:] + configurations[:shift]:
            name = f"{group}-r{repeat}-{job['name']}"
            print(name, flush=True)
            benchmark.run(SimpleNamespace(
                suite=Path(job["suite"]), output=args.output / name,
                build=Path(job["build"]), cuda=job["cuda"],
                pool_mib=job["pool_mib"], repeats=1,
                cases=[job["case"]], timeout=300))
