"""Independent artifact/trace/numerical audit for the frozen directed tasks.

Accepts disjoint completed baseline/native/public coordinators, preserves all
failure/blocked/not-run states, and audits only actual successful artifacts.
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
from benchmark_runner import DEFAULT,dump,strict_file,load
from semantic_benchmark_execution import runtime_sources
from audit_semantic_benchmark import audit,verify_directed_coverage

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",action="append",type=Path,required=True)
    p.add_argument("--plaintext",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Platform audit directory required")
    if not a.inside:
        cmd=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]
        return enter_nix('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(cmd),seconds=930)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    import numpy,torch
    from platform_config import require_python_packages
    require_python_packages(torch,numpy);torch.set_num_threads(1)
    from audit_unified_candidate import verify_candidate,verify_files
    from seal_cpu_golden import BUILD
    from unified_graph_exercises import spec as native_spec
    from unified_public_exercises import spec as public_spec
    sources=runtime_sources();folders=[];bindings={};profiles=set()
    for batch in a.batch:
        terminal=strict_file(batch/"report.json",4*1024**2)
        plan=strict_file(batch/"plan.json",8*1024**2)
        if not terminal["all_shards_processed"] or terminal["binding"]!=plan["binding"]:
            raise ValueError("Incomplete coordinator")
        if plan["source_hashes"]!=sources:raise ValueError("Historical runtime cannot be rebound")
        key=(plan["mode"],plan["construction_profile"])
        if key in profiles:raise ValueError("Duplicate coordinator profile")
        profiles.add(key);bindings[str(batch)]=sha(batch/"report.json")
        folders.extend(batch/("shard-"+str(s["shard"]).zfill(3)) for s in terminal["stages"])
    if not {("directed","native"),("directed","public-v1")}<=profiles:
        raise ValueError("Both directed profiles required for complete denominator")
    score=audit(DEFAULT,a.plaintext,folders)
    rows,_=load(DEFAULT);models={r["model"]["id"]:r for r in rows}
    tasks={t["id"]:t for t in strict_file(DEFAULT/"coverage.json",4*1024**2)["directed_tasks"]}
    selected=[t for t in score["directed_tasks"] if t["state"]=="passed"]
    binaries={k:sha(BUILD/"lib"/n) for k,n in
              (("runtime_sha256","libSEAL_HEVM.so"),("frontend_sha256","libHecateFrontend.so"))}
    plan=dict(schema=1,coordinator_reports=bindings,runtime_sources=sources,binaries=binaries,
        task_hashes={t["id"]:t["task_sha256"] for t in tasks.values()},
        selected=[t["id"] for t in selected],auditor_sha256=sha(Path(__file__)),
        max_wall_seconds=900,max_result_mib=128,paid_calls=0,new_encrypted_executions=0)
    plan["binding"]=digest(plan);out=a.output
    if out.exists():
        if not a.resume or strict_file(out/"plan.json",8*1024**2)!=plan:p.error("Exact audit resume required")
    else:
        out.mkdir(parents=True);dump(out/"plan.json",plan)
        (out/"runner.py").write_bytes(Path(__file__).read_bytes())
    dump(out/"score.json",score)
    hashes=strict_file(out/"checkpoint.json",2*1024**2) if (out/"checkpoint.json").exists() else {}
    progress=strict_file(out/"progress.json",4096) if (out/"progress.json").exists() else {"seconds":0}
    prior=progress["seconds"];start=time.monotonic();values=0;maximum=0.;records=0;contexts={}
    for item in selected:
        if runtime_sources()!=sources:raise ValueError("Runtime source drift")
        name=item["id"];task=tasks[name];dest=out/(name+".json")
        if len(item["evidence"])!=1:raise ValueError("Ambiguous directed evidence")
        evidence=Path(item["evidence"][0])
        if dest.exists():
            if sha(dest)!=hashes.get(dest.name):raise ValueError("Uncheckpointed audit changed")
            checked=strict_file(dest,8*1024**2);verify_files(evidence,checked["files"])
        else:
            if prior+time.monotonic()-start>=900:break
            if sum(f.stat().st_size for f in out.iterdir() if f.is_file())>=128*1024**2:break
            report=strict_file(evidence/"report.json",8*1024**2)
            if any(report[k]!=v for k,v in binaries.items()):raise ValueError("SDK binary identity changed")
            checked=verify_candidate(evidence)
            if checked["model_sha256"]!=models[task["model_id"]]["model_sha256"]:
                raise ValueError("Executed directed model mismatch")
            expected=(public_spec if task["profile"]=="hecate-unified-public-v1" else native_spec)(task["exercise"])
            verify_directed_coverage(checked["coverage"],task,expected)
            if checked["exercise"]!=task["exercise"]:raise ValueError("Different construction exercise")
            checked.update(task_id=name,task_sha256=task["task_sha256"],requirement=task["requirement"],
                           acceptance_kind=task["acceptance_kind"])
            dump(dest,checked);hashes[dest.name]=sha(dest);dump(out/"checkpoint.json",hashes)
            dump(out/"progress.json",dict(seconds=prior+time.monotonic()-start))
        values+=checked["comparison"]["compared_values"]
        maximum=max(maximum,checked["comparison"]["max_absolute_error"]);records+=1
        contexts.setdefault(task["requirement"],set()).add(models[task["model_id"]]["topology"])
        if records%48==0:print(json.dumps(dict(audited=records,planned=len(selected))),flush=True)
    if runtime_sources()!=sources:raise ValueError("Runtime source drift")
    report=dict(binding=plan["binding"],planned_passed_tasks=len(selected),independently_verified=records,
        remaining=len(selected)-records,directed_denominator=len(tasks),
        directed_counts=score["directed_counts"],numeric_counts=score["directed_numeric_counts"],
        mixed_counts=score["directed_mixed_counts"],structural_counts=score["directed_structural_counts"],
        requirements_three_verified_contexts=sorted(k for k,v in contexts.items() if len(v)>=3),
        context_counts={k:len(v) for k,v in contexts.items()},compared_values=values,max_absolute_error=maximum,
        seconds=prior+time.monotonic()-start,new_encrypted_executions=0,paid_calls=0,
        scope="Independent reference/decrypted-array audit plus actual construction trace and finite contribution witnesses; structural children score separately",
        record_hashes=hashes)
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("record_hashes","context_counts","requirements_three_verified_contexts")}))
    return int(records!=len(selected))
if __name__=="__main__":raise SystemExit(main())
