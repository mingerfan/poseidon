"""Bounded offline baseline coordinator. No paid mode, new dependencies or builds."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from benchmark_graph import digest
from benchmark_runner import DEFAULT,ROOT,load,dump,strict_file
from semantic_benchmark_execution import runtime_sources,summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--unified-profile",choices=("native","public-v1"),default="native")
    p.add_argument("--mode",choices=("baseline","directed"),default="baseline")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    p.add_argument("--output",type=Path)
    p.add_argument("--max-wall-seconds",type=int,default=7200)
    p.add_argument("--max-result-mib",type=int,default=1024)
    a=p.parse_args()
    if not 1<=a.max_wall_seconds<=43200 or not 1<=a.max_result_mib<=40960:p.error("Budget bounds")
    rows,index=load(DEFAULT);sources=runtime_sources()
    tasks=len(rows) if a.mode=="baseline" else len(strict_file(DEFAULT/"coverage.json",4*1024**2)["directed_tasks"])
    body=dict(index=index,source_hashes=sources,max_wall_seconds=a.max_wall_seconds,
              max_result_mib=a.max_result_mib,shards=(tasks+47)//48,mode=a.mode,construction_profile=a.unified_profile,planned_tasks=tasks,concurrency=1)
    plan=dict(body,binding=digest(body),paid_calls=0)
    if not a.execute:
        print(json.dumps(dict(binding=plan["binding"],models=len(rows),mode=a.mode,planned_tasks=tasks,shards=plan["shards"],concurrency=1,
             max_wall_seconds=a.max_wall_seconds,max_result_mib=a.max_result_mib,paid_calls=0),indent=2));return 0
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output is None or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("Output must be in platform results")
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=a.max_wall_seconds+30)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:p.error("Locked Nix Python required")
    out=a.output
    if out.exists():
        if not a.resume or strict_file(out/"plan.json",8*1024**2)!=plan:p.error("Frozen coordinator differs")
    else:out.mkdir(parents=True);dump(out/"plan.json",plan)
    ledger=out/"stages.json";stages=strict_file(ledger,1024**2) if ledger.exists() else []
    begin=time.monotonic();prior=sum(x["seconds"] for x in stages);used=sum(x.get("evidence_bytes",0) for x in stages)
    for n in range(plan["shards"]):
        if runtime_sources()!=sources:raise ValueError("Source changed; stopped at shard boundary")
        dest=out/("shard-"+str(n).zfill(3))
        if n<len(stages):
            checked=summary(dest)
            if checked["binding"]!=stages[n]["binding"]:raise ValueError("Completed shard binding changed")
            if stages[n]["not_run"]==0:continue
        old=stages[n] if n<len(stages) else None
        remaining=int(a.max_wall_seconds-prior-(time.monotonic()-begin))
        space=int(a.max_result_mib-used/1024**2)
        if remaining<1 or space<1:break
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/benchmark.py"),a.mode,"--inside",
                 "--execute","--unified-profile",a.unified_profile,"--shard-index",str(n),"--output",str(dest),
                 "--max-wall-seconds",str(remaining),"--max-result-mib",str(space)]
        if dest.exists():
            prior_plan=strict_file(dest/"plan.json",8*1024**2)
            command[command.index("--max-wall-seconds")+1]=str(prior_plan["max_wall_seconds"])
            command[command.index("--max-result-mib")+1]=str(prior_plan["max_result_mib"])
            command+=["--resume"]
        start=time.monotonic();timed_out=False
        with (out/("shard-"+str(n).zfill(3)+".log")).open("a") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        if not (dest/"plan.json").exists():raise ValueError("Shard failed before plan; inspect preserved log")
        checked=summary(dest)
        stage=dict(shard=n,binding=checked["binding"],seconds=time.monotonic()-start,exit_code=code,
                   evidence_bytes=checked["evidence_bytes"],statuses=checked["statuses"],not_run=checked["not_run_selected"])
        if old:
            stage["seconds"]+=old["seconds"];stages[n]=stage
        else:stages.append(stage)
        dump(ledger,stages);used+=stage["evidence_bytes"]-(old["evidence_bytes"] if old else 0)
        print(json.dumps(stage),flush=True)
        if timed_out or checked["not_run_selected"]:break
    complete=len(stages)==plan["shards"] and all(s["not_run"]==0 for s in stages)
    report=dict(binding=plan["binding"],all_shards_processed=complete,stages=stages,
                seconds=prior+time.monotonic()-begin,evidence_bytes=used,paid_calls=0,
                failures_are_not_passes=True)
    dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="stages"},indent=2))
    return int(not complete)

if __name__=="__main__":raise SystemExit(main())
