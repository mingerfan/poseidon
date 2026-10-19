#!/usr/bin/env python3
"""Rotate exact benchmark configurations, with a fresh process per sample."""
import argparse
import importlib.util
import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=Path, required=True)
parser.add_argument("--jobs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("waterline_benchmark", args.benchmark.resolve())
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
jobs = json.loads(args.jobs.read_text())
args.output.mkdir(parents=True, exist_ok=args.resume)
if (args.output/"jobs.json").exists():
    previous=(args.output/"jobs.json").read_bytes()
    history=args.output/"jobs-history";history.mkdir(exist_ok=True)
    (history/(hashlib.sha256(previous).hexdigest()+".json")).write_bytes(previous)
(args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
for group in dict.fromkeys(j["group"] for j in jobs):
    configurations = [j for j in jobs if j["group"] == group]
    for repeat in range(args.repeats):
        shift = repeat % len(configurations)
        for job in configurations[shift:] + configurations[:shift]:
            name = f"{group}-r{repeat}-{job['name']}"
            folder=args.output/name
            if args.resume and folder.exists():
                summary=json.loads((folder/"summary.json").read_text())
                suite=json.loads((Path(job["suite"])/"suite.json").read_text())
                case=next(c for c in suite["cases"] if c["name"]==job["case"])
                runner=case["runner"]
                expected=benchmark.digest(Path(job["build"])/"bin"/f"poseidon_runtime_gpu_{runner}_e2e")
                if summary["runner_hashes"][runner]==expected and not summary["samples"][0]["passed"]:
                    raise RuntimeError("unchanged candidate already failed; inspect it before any further run: "+str(folder))
                if (len(summary["samples"])==1 and summary["samples"][0]["passed"] and
                    summary["selected_case_names"]==[job["case"]] and summary["cuda"]==job["cuda"] and
                    summary["pool_mib"]==job["pool_mib"] and summary["runner_hashes"][runner]==expected):
                    print("Keep verified "+name,flush=True);continue
                archive=args.output.parent/("pre-reuse-"+args.output.name)
                archive.mkdir(exist_ok=True)
                archived=archive/(name+'-'+summary['runner_hashes'][runner].removeprefix('sha256:')[:12])
                assert not archived.exists()
                shutil.move(str(folder),archived)
                print("Archived superseded "+name,flush=True)
            print(name, flush=True)
            benchmark.run(SimpleNamespace(
                suite=Path(job["suite"]), output=args.output / name,
                build=Path(job["build"]), cuda=job["cuda"],
                pool_mib=job["pool_mib"], repeats=1,
                cases=[job["case"]], timeout=300))
