"""Serial completion of the frozen r29 offline acceptance tasks.

Existing coordinators enforce their budgets. No live/provider mode, installation,
source update or automatic retry. A resource/infrastructure stop ends this chain.
"""
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

EXPECTED="9e8981a59439b096230afc1dbcbc9af21182d2371f8fec7422c41da2b82b1781"
def main():
    if len(sys.argv)!=1:raise ValueError("This frozen offline chain has no configurable expansion")
    if os.environ.get("POSEIDON_PLATFORM")!="aarch64-linux":raise ValueError("Explicit ARM platform required")
    source=runtime_sources()
    if digest(source)!=EXPECTED:raise ValueError("Frozen r29 source required")
    for folder in ("benchmark-r40-followup","benchmark-r41-independent-audit"):
        r=strict_file(RESULTS/folder/"report.json",2*1024**2)
        if folder.endswith("followup") and not r["all_stages_passed"]:raise ValueError("Short gates incomplete")
        if folder.endswith("independent-audit") and r["remaining"]!=0:raise ValueError("Independent audit incomplete")
    output=RESULTS/"benchmark-r41-remaining-chain"
    if output.exists():raise ValueError("Existing chain preserved; inspect before continuing")
    suite=BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
    bundle=BASE/"benchmarks/model-contexts-draft-v1/tasks.json"
    stages=[]
    for profile,name in (("native","native-directed"),("public-v1","public-directed")):
        path=RESULTS/("benchmark-r41-"+name)
        cmd=[sys.executable,"-B",str(BASE/"run_benchmark_shards.py"),"--execute","--mode","directed",
             "--unified-profile",profile,"--output",str(path),"--max-wall-seconds","7200","--max-result-mib","1024"]
        stages.append((name,cmd,path,7350))
    for index in range(2):
        path=RESULTS/("model-contexts-baseline-"+str(index).zfill(3))
        cmd=[sys.executable,"-B",str(HERE/"run_model_semantic_contexts.py"),"--suite",str(suite),
             "--bundle",str(bundle),"--plaintext",str(RESULTS/"model-contexts-dual-reference"),
             "--shard-index",str(index),"--execute","--output",str(path)]
        stages.append(("supplement-"+str(index),cmd,path,1950))
    for _,_,path,_ in stages:
        if path.exists():raise ValueError("Stage evidence exists; no automatic overwrite or repeat")
    output.mkdir(parents=True)
    dump(output/"plan.json",dict(runtime_source_sha256=EXPECTED,concurrency=1,paid_calls=0,
        chain_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stages=[dict(id=n,command=c,output=str(p),outer_timeout=t) for n,c,p,t in stages]))
    (output/"runner.py").write_bytes(Path(__file__).read_bytes())
    completed=[]
    for name,command,path,limit in stages:
        if runtime_sources()!=source:raise ValueError("Source changed at safe boundary")
        start=time.monotonic()
        with (output/(name+".log")).open("w") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGINT)
                try:code=child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
                raise RuntimeError("Outer deadline; preserve interrupted stage without retry")
        report=strict_file(path/"report.json",4*1024**2)
        record=dict(id=name,exit_code=code,seconds=time.monotonic()-start,
            report=str(path/"report.json"),report_sha256=hashlib.sha256((path/"report.json").read_bytes()).hexdigest())
        completed.append(record);dump(output/"stages.json",completed);print(json.dumps(record),flush=True)
        if name.endswith("directed"):
            if not report["all_shards_processed"]:raise ValueError("Directed stage resource/infrastructure boundary")
        elif report["not_run_selected"]:
            raise ValueError("Supplement stage resource/infrastructure boundary")
        # Ordinary model failures remain in each report and do not suppress an
        # independent subsequent track. They never count as passing cases.
    dump(output/"report.json",dict(runtime_source_sha256=EXPECTED,stages=completed,
        all_stages_processed=True,paid_calls=0,new_models_added_to_original_corpus=0))
    return 0
if __name__=="__main__":raise SystemExit(main())
