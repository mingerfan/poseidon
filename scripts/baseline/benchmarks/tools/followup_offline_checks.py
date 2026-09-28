"""One bounded offline continuation after a currently verified baseline process.

No timer service or recurring automation. It never restarts the baseline,
installs dependencies, calls a provider, or changes runtime sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from semantic_benchmark_execution import runtime_sources
from workspace_paths import RESULTS

def process_identity(pid):
    root=Path("/proc")/str(pid)
    try:
        stat=(root/"stat").read_text()
        # comm can contain spaces; fields after the final ')' start with state.
        start=stat[stat.rfind(")")+2:].split()[19]
        command=(root/"cmdline").read_bytes()
    except FileNotFoundError:return None
    return dict(pid=pid,start_ticks=start,command_sha256=hashlib.sha256(command).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--pid",type=int,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):
        p.error("Fresh platform results output required")
    identity=process_identity(a.pid)
    if identity is None:p.error("Baseline process not live; inspect terminal state directly")
    command=(Path("/proc")/str(a.pid)/"cmdline").read_bytes()
    if b"run_benchmark_shards.py" not in command or str(a.baseline).encode() not in command:
        p.error("PID does not identify the specified baseline")
    sources=runtime_sources();source_sha=digest(sources)
    plan=strict_file(a.baseline/"plan.json",8*1024**2)
    if plan["source_hashes"]!=sources:raise ValueError("Baseline frozen source drift")
    suite=BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
    scripts=[
      ("artifact_audit","audit_benchmark_artifact_snapshot.py",[
        "--batch",str(a.baseline),"--plaintext",str(RESULTS/"upstream-chunks-r29-plaintext"),
        "--output",str(RESULTS/"benchmark-r40-final-artifact-audit")],120),
      ("dual_reference","verify_model_semantic_contexts.py",[
        "--suite",str(suite),"--bundle",str(BASE/"benchmarks/model-contexts-draft-v1/tasks.json"),
        "--execute","--output",str(RESULTS/"model-contexts-dual-reference")],200),
      ("trace_gates","verify_construction_trace_counterexamples.py",[
        "--suite",str(suite),"--execute","--output",str(RESULTS/"construction-trace-counterexamples")],170),
      ("helper_gates_000","verify_helper_call_counterexamples.py",[
        "--suite",str(suite),"--shard-index","0","--execute","--output",str(RESULTS/"helper-call-negatives-000")],230),
      ("helper_gates_001","verify_helper_call_counterexamples.py",[
        "--suite",str(suite),"--shard-index","1","--execute","--output",str(RESULTS/"helper-call-negatives-001")],230)]
    runner_hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for _,name,_,_ in scripts}
    for _,_,args,_ in scripts:
        if Path(args[args.index("--output")+1]).exists():raise ValueError("Follow-up evidence already exists")
    a.output.mkdir(parents=True)
    dump(a.output/"plan.json",dict(process=identity,baseline_binding=plan["binding"],
        runtime_source_sha256=source_sha,runner_hashes=runner_hashes,maximum_wait_seconds=3600,
        stages=[dict(id=n,script=f,args=args,timeout_seconds=t) for n,f,args,t in scripts],paid_calls=0))
    (a.output/"runner.py").write_bytes(Path(__file__).read_bytes())
    begin=time.monotonic()
    while process_identity(a.pid)==identity:
        if time.monotonic()-begin>=3600:raise RuntimeError("Observed baseline wait budget ended; no restart")
        time.sleep(2)
    terminal=strict_file(a.baseline/"report.json",4*1024**2)
    if terminal["binding"]!=plan["binding"] or not terminal["all_shards_processed"]:
        raise ValueError("Baseline is not fully processed; preserve and assess before follow-up")
    if runtime_sources()!=sources:raise ValueError("Runtime changed at transition")
    stages=[]
    for name,script,args,limit in scripts:
        if runtime_sources()!=sources or hashlib.sha256((HERE/script).read_bytes()).hexdigest()!=runner_hashes[script]:
            raise ValueError("Bound follow-up source drift")
        command=[sys.executable,"-B",str(HERE/script),*args]
        started=time.monotonic()
        with (a.output/(name+".log")).open("w") as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=limit)
        dest=Path(args[args.index("--output")+1])
        report_path=dest/"report.json"
        row=dict(id=name,exit_code=result.returncode,seconds=time.monotonic()-started,
                 report=str(report_path),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path.is_file() else None)
        stages.append(row);dump(a.output/"stages.json",stages);print(json.dumps(row),flush=True)
        if not report_path.is_file():
            raise RuntimeError("Follow-up infrastructure failure; see retained log")
    dump(a.output/"report.json",dict(stages=stages,paid_calls=0,runtime_source_sha256=source_sha,
        all_stages_completed=True,all_stages_passed=all(s["exit_code"]==0 for s in stages),
        new_encrypted_execution=False))
    return int(any(s["exit_code"] for s in stages))
if __name__=="__main__":raise SystemExit(main())
