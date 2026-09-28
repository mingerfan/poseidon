"""Recompute references and inspect real artifacts for a completed baseline.

Read-only with respect to execution evidence. No new candidate execution, API,
installation or build. Uses the existing independent unified-candidate auditor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import DEFAULT,dump,strict_file
from semantic_benchmark_execution import runtime_sources
from audit_semantic_benchmark import audit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,required=True);p.add_argument("--plaintext",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--resume",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform audit directory required")
    if not a.inside:
        command=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(command),seconds=930)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    import numpy,torch
    from platform_config import require_python_packages
    require_python_packages(torch,numpy);torch.set_num_threads(1)
    from audit_unified_candidate import verify_candidate,verify_files
    from seal_cpu_golden import BUILD
    sources=runtime_sources()
    terminal=strict_file(a.batch/"report.json",4*1024**2)
    outer_plan=strict_file(a.batch/"plan.json",8*1024**2)
    if not terminal["all_shards_processed"] or terminal["binding"]!=outer_plan["binding"]:
        raise ValueError("Baseline not fully processed")
    if outer_plan["source_hashes"]!=sources:raise ValueError("Different runtime sources")
    folders=[a.batch/("shard-"+str(s["shard"]).zfill(3)) for s in terminal["stages"]]
    score=audit(DEFAULT,a.plaintext,folders)
    selected=[m for m in score["models"] if m["baseline"]=="passed"]
    binaries={name:sha(BUILD/"lib"/file) for name,file in
              (("runtime_sha256","libSEAL_HEVM.so"),("frontend_sha256","libHecateFrontend.so"))}
    plan=dict(schema=1,baseline_report_sha256=sha(a.batch/"report.json"),
        runtime_sources=sources,selected={m["id"]:m["model_sha256"] for m in selected},
        binaries=binaries,auditor_sha256=sha(Path(__file__)),max_wall_seconds=900,max_result_mib=128,
        corpus_status_counts=score["counts"],paid_calls=0,new_encrypted_executions=0)
    plan["binding"]=digest(plan);out=a.output
    if out.exists():
        if not a.resume or strict_file(out/"plan.json",8*1024**2)!=plan:p.error("Exact audit resume required")
    else:out.mkdir(parents=True);dump(out/"plan.json",plan)
    hashes=strict_file(out/"checkpoint.json",2*1024**2) if (out/"checkpoint.json").exists() else {}
    progress=strict_file(out/"progress.json",4096) if (out/"progress.json").exists() else {"seconds":0}
    prior=progress["seconds"];start=time.monotonic();values=0;maximum=0.;records=0
    for m in selected:
        if runtime_sources()!=sources:raise ValueError("Runtime source drift")
        name=m["id"];dest=out/(name+".json")
        if len(m["evidence"])!=1:raise ValueError("Ambiguous baseline evidence")
        evidence=Path(m["evidence"][0])
        if dest.exists():
            if sha(dest)!=hashes.get(dest.name):raise ValueError("Uncheckpointed audit changed")
            checked=strict_file(dest,8*1024**2);verify_files(evidence,checked["files"])
        else:
            if prior+time.monotonic()-start>=900:break
            if sum(f.stat().st_size for f in out.iterdir() if f.is_file())>=128*1024**2:break
            report=strict_file(evidence/"report.json",8*1024**2)
            if any(report[k]!=v for k,v in binaries.items()):raise ValueError("SDK binary differs from observed execution")
            checked=verify_candidate(evidence)
            if checked["model_sha256"]!=m["model_sha256"]:raise ValueError("Wrong executed model")
            dump(dest,checked);hashes[dest.name]=sha(dest);dump(out/"checkpoint.json",hashes)
            dump(out/"progress.json",dict(seconds=prior+time.monotonic()-start))
        values+=checked["comparison"]["compared_values"]
        maximum=max(maximum,checked["comparison"]["max_absolute_error"]);records+=1
        if records%48==0:print(json.dumps(dict(audited=records,planned=len(selected))),flush=True)
    if runtime_sources()!=sources:raise ValueError("Runtime source drift")
    report=dict(binding=plan["binding"],planned_passed_cases=len(selected),independently_verified=records,
        remaining=len(selected)-records,source_denominator=1200,source_counts=score["counts"],
        compared_values=values,max_absolute_error=maximum,seconds=prior+time.monotonic()-start,
        new_encrypted_executions=0,paid_calls=0,
        scope="Actual decrypted arrays against reconstructed NumPy/math and CPU float64 Torch references; source, binary, keys, artifacts and trace checks",
        record_hashes=hashes)
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="record_hashes"}))
    return int(records!=len(selected))
if __name__=="__main__":raise SystemExit(main())
