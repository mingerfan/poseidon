"""Trace-only diagnostic of pinned API calls, with exact old IR/CST comparison.

This does not create new encrypted pass records or infer output contribution from
call presence. Historical parent witnesses remain separately bound.
"""
import argparse,hashlib,json,os,shlex,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
from semantic_benchmark_execution import runtime_sources
WORKER = r'''
import sys,json,hashlib
from pathlib import Path
from collections import Counter
payload=json.loads(Path("/payload.json").read_text())
table={}
for row in payload["api_inventory"]:
    path=row["sandbox_source"]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=row["source_sha256"]:
        raise ValueError("Diagnostic source drift")
    table[(path,row["line"])]=row["symbol"]
counts=Counter()
def observe(frame,event,arg):
    if event!="call":return
    code=frame.f_code;symbol=table.get((code.co_filename,code.co_firstlineno))
    if symbol is None:return
    facts={}
    for name in ("other","self","x","A"):
        if name in frame.f_locals:facts[name+"_type"]=type(frame.f_locals[name]).__name__
    for name in ("opcode","offset","m","p"):
        value=frame.f_locals.get(name)
        if type(value) is int and abs(value)<=65536:facts[name]=value
    key=(code.co_filename,symbol,json.dumps(facts,sort_keys=True))
    counts[key]+=1
    if len(counts)>4096:raise ValueError("Diagnostic event budget")
import candidate_trace
sys.setprofile(observe)
try:candidate_trace.main()
finally:sys.setprofile(None)
Path("/out/api-calls.json").write_text(json.dumps([
 dict(source=k[0],symbol=k[1],facts=json.loads(k[2]),count=v)
 for k,v in sorted(counts.items())],sort_keys=True))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(p):
 r=strict_file(p,8*1024**2)
 if digest({k:v for k,v in r.items() if k!="binding"})!=r["binding"]:raise ValueError("Audit binding")
 return r
def compare_artifacts(out, old):
 """Compare raw frontend artifacts, then independently reconstruct executed CST."""
 from upstream_adapters.constants import canonicalize
 same={"candidate_trace.mlir":sha(out/"candidate_trace.mlir")==sha(old/"candidate_trace.mlir")}
 name="_hecate_golden.cst";raw_name=name+".upstream-original"
 lineage={"method":"identity","raw_file":name}
 if (old/raw_name).exists():
  metadata=strict_file(old/"constant-layout.json",1024**2)[name]
  if metadata.get("original_file")!=raw_name:raise ValueError("Unexpected raw CST provenance")
  reconstructed,record=canonicalize((old/raw_name).read_bytes(),metadata["period"])
  if dict(record,original_file=raw_name)!=metadata:raise ValueError("CST layout metadata drift")
  if reconstructed!=(old/name).read_bytes():raise ValueError("Executed CST differs from lossless bridge")
  same["raw_cst"]=sha(out/name)==sha(old/raw_name)
  fresh,unused=canonicalize((out/name).read_bytes(),metadata["period"])
  same["executed_cst"]=fresh==(old/name).read_bytes()
  lineage=dict(metadata,layout_record_sha256=sha(old/"constant-layout.json"),
               bridge_source_sha256=sha(BASE/"upstream_adapters/constants.py"))
 else:
  if (old/"constant-layout.json").exists():
   metadata=strict_file(old/"constant-layout.json",1024**2)
   if name in metadata:raise ValueError("Missing original CST evidence")
  same["raw_cst"]=sha(out/name)==sha(old/name)
  same["executed_cst"]=same["raw_cst"]
 return same,lineage
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--limit",type=int,default=48);p.add_argument("--offset",type=int,default=0)
 p.add_argument("--execute",action="store_true");p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 if not 1<=a.limit<=48 or a.offset<0:p.error("Diagnostic shard bounds")
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV
 audit_path=ROOT/"docs/baseline/stage2-helper-mapping-r49.json";audit=bound(audit_path)
 selected={}
 for partition in audit["helper_partitions"]:
  if partition["state"]!="three_contexts_verified":continue
  seen=set()
  for task in partition["task_ids"]:
   record=audit["records"][task]
   if record["topology"] in seen:continue
   seen.add(record["topology"]);selected[task]=record
   if len(seen)==3:break
 items=list(selected.items())[a.offset:a.offset+a.limit]
 if not items:p.error("Empty diagnostic shard")
 if not a.execute:print(json.dumps(dict(planned=len(items),total=len(selected),ids=[k for k,v in items],actual_compile=False,new_encrypted_executions=0,paid_calls=0)));return 0
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform output")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=960)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 import candidate_sandbox as sandbox
 ledger=strict_file(BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json",4*1024**2)
 inventory=[]
 for r in ledger["upstream_api"]:
  path=r["source"]
  mount=("/frontend/hecate/expr.py" if path.endswith("/expr.py") else "/upstream-poly/poly/"+Path(path).name)
  inventory.append(dict(sandbox_source=mount,source_sha256=r["source_sha256"],symbol=Path(path).stem+"."+r["symbol"],line=r["line"]))
 sources=runtime_sources();start=time.monotonic();a.output.mkdir(parents=True)
 plan=dict(sources=sources,runner_sha256=sha(Path(__file__)),worker_sha256=digest(WORKER),
           parent_audit_sha256=sha(audit_path),tasks=dict(items),max_wall_seconds=900,max_result_mib=128,
           api_inventory=inventory,paid_calls=0,new_encrypted_executions=0)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan);rows=[]
 for task,record in items:
  if runtime_sources()!=sources or time.monotonic()-start>900:raise ValueError("Diagnostic source/time boundary")
  if sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file())>128*1024**2:raise ValueError("Diagnostic disk boundary")
  evidence=Path(record["evidence"])
  if sha(evidence/"report.json")!=record["report_sha256"]:raise ValueError("Old evidence changed")
  original=strict_file(evidence/"report.json",8*1024**2)
  attempt=next(x for x in original["attempts"] if x.get("status")=="passed")
  if attempt["index"]!=0:raise ValueError("Diagnostic expects first-attempt historical pass")
  for name,hsh in attempt["artifact_hashes"].items():
   if Path(name).name!=name or sha(evidence/"attempt-00/output"/name)!=hsh:raise ValueError("Historical artifact changed")
  folder=a.output/task;folder.mkdir();out=folder/"output";out.mkdir()
  payload=strict_file(evidence/"attempt-00/trace-payload.json",4*1024**2)
  payload["api_inventory"]=inventory;dump(folder/"payload.json",payload)
  code=sandbox.run(folder/"payload.json",out,[str(VENV/"bin/python"),"-B","-c",WORKER],folder/"trace.log",seconds=60,helpers=True)
  row=dict(id=task,status="trace_failed",exit_code=code,topology=record["topology"],
           historical_evidence=str(evidence),historical_report_sha256=record["report_sha256"])
  if code==0:
   old=evidence/"attempt-00/output"
   same,lineage=compare_artifacts(out,old)
   row.update(status="identical_artifacts" if all(same.values()) else "artifact_mismatch",artifact_comparison=same,constant_lineage=lineage,
     files={p.name:sha(p) for p in out.iterdir() if p.is_file()},calls=strict_file(out/"api-calls.json",1024**2))
  rows.append(row);dump(a.output/"progress.json",rows)
 if runtime_sources()!=sources:raise ValueError("Diagnostic source drift")
 report=dict(format="poseidon-api-retrace-v1",plan_binding=plan["binding"],rows=rows,seconds=time.monotonic()-start,
  actual_frontend_tracing=True,actual_compile=False,new_encrypted_executions=0,paid_calls=0,
  call_presence_is_not_contribution=True,all_api_behaviors_accepted=False)
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps(dict(planned=len(rows),identical=sum(r["status"]=="identical_artifacts" for r in rows),
   failures=[dict(id=r["id"],status=r["status"]) for r in rows if r["status"]!="identical_artifacts"],
   seconds=report["seconds"],paid_calls=0)))
 return int(any(r["status"]!="identical_artifacts" for r in rows))
if __name__=="__main__":raise SystemExit(main())
