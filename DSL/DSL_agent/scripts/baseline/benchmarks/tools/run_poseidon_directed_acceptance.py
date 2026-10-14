#!/usr/bin/env python3
"""Bounded developer-only Poseidon acceptance of frozen directed DSL tasks."""
import argparse, contextlib, copy, hashlib, json, shutil, sys, time
from pathlib import Path
ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT / "scripts/baseline"))
from workspace_paths import WORK
from benchmark_graph import canonical, digest
from benchmark_runner import load
from benchmark_math import evaluate as mathematical
from benchmark_graph import samples
from benchmark_torch import evaluate as torch_reference
from compiler_configuration import configuration
from unified_graph_contract import prepare, validate_candidate
from unified_graph_lowering import candidate_source
from check_poseidon_helpers import snapshot
from run_candidate import inside, parse_args
from audit_unified_candidate import verify_candidate
import numpy as np

DEFAULT_REQUIREMENTS = (
 "array.reshape_transpose", "array.storage", "array.view_inplace",
 "native.call", "native.return", "native.starred", "loop.augmented",
 "scalar.augmented", "copy", "call.forward", "call.nested_array",
 "call.repeated", "call.two_cipher_arguments", "return.rank4",
 "star.multiple", "star.call_result")
def require(value, message):
 if not value: raise ValueError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, value):
 data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"
 temp=path.with_name(path.name+".partial");temp.write_text(data);temp.replace(path)
def byte_size(root): return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--plan",action="store_true")
 p.add_argument("--requirement",action="append")
 p.add_argument("--profile",choices=("native","public"),default="native")
 a=p.parse_args()
 out=a.output
 require(out.resolve().parent==(WORK/"results").resolve() and not out.exists(),"Fresh platform result directory required")
 suite=ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
 corpus,index=load(suite,check_sources=False)
 ledger=json.loads((suite/"coverage.json").read_text())
 profile="hecate-unified-"+a.profile+"-v1"
 groups=a.requirement or (list(DEFAULT_REQUIREMENTS) if a.profile=="native" else sorted({t["requirement"] for t in ledger["directed_tasks"] if t["profile"]==profile})[:16])
 require(len(groups)==len(set(groups)) and 1<=len(groups)<=16,"At most 16 unique partitions")
 tasks=[t for t in ledger["directed_tasks"] if t["profile"]==profile and t["requirement"] in groups]
 require(len(tasks)==3*len(groups) and len(tasks)<=48,"Three original contexts required")
 models={r["model"]["id"]:r for r in corpus}
 cfg=configuration("poseidon-cpu-eva-w45-v1")
 sources=snapshot()
 require(cfg.get("runtime_plan_entry_policy")=="hecate-private-symbol-dce-v1","Current Poseidon entry policy required")
 plan=dict(format="poseidon-directed-acceptance-v1",backend="poseidon_cpu",evaluation_source="manual_rule_fixture_not_agent",
  suite_index_sha256=sha(suite/"index.json"),coverage_sha256=sha(suite/"coverage.json"),
  historical_generator_sources=index["source_hashes"],historical_generator_not_reused=True,
  runner_sha256=sha(Path(__file__)),sources=sources,compiler_configuration=cfg,construction_profile=profile,
  max_wall_seconds=1800,max_artifact_bytes=1024**3,min_free_disk_bytes=4*1024**3,
  global_artifact_limit_bytes=32*1024**3,native_concurrency=1,paid_calls=0,
  numeric_tolerance=dict(atol=1e-5,rtol=1e-4),tasks=[])
 out.mkdir(); start=time.monotonic()
 records=[]
 for task in tasks:
  row=models[task["model_id"]]
  require(row["model_sha256"]==task["model_sha256"] and digest(row["model"])==task["model_sha256"],"Frozen model hash mismatch")
  item=dict(task=task,model_sha256=row["model_sha256"],topology=row["topology"],status="ready")
  folder=out/task["id"];folder.mkdir()
  (folder/"model.json").write_bytes(canonical(row["model"]))
  try:
   request=prepare(row["model"],cfg["ckks_config_sha256"],cfg,task["exercise"],construction_profile=profile if a.profile=="public" else None)
   source,lowering=candidate_source(request)
   validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
   (folder/"request.json").write_bytes(canonical(request))
   (folder/"candidate.py").write_text(source)
   item.update(request_id=request["request_id"],candidate_sha256=sha(folder/"candidate.py"),lowering=lowering)
   errors=[]
   for inputs in samples(row["model"],16):
    x=mathematical(row["model"],inputs);y=torch_reference(row["model"],inputs)
    for name in x:
     np.testing.assert_allclose(x[name],y[name],atol=1e-12,rtol=1e-12)
     errors.append(float(np.max(np.abs(x[name]-y[name]))))
   item.update(reference_groups=16,reference_max_abs_error=max(errors))
  except (ValueError,TypeError,IndexError,AssertionError) as exc:
   item.update(status="blocked_preflight",failure_layer="static_or_reference",diagnostic=str(exc)[:2000])
  plan["tasks"].append(item)
 plan["binding"]=digest(plan)
 dump(out/"plan.json",plan)
 if a.plan:
  print(json.dumps(dict(binding=plan["binding"],tasks=len(tasks),ready=sum(t["status"]=="ready" for t in plan["tasks"]),blocked=sum(t["status"]!="ready" for t in plan["tasks"]))),flush=True)
  return 0
 initial_global=byte_size(WORK/"results")
 stop=None
 def report():
  evidence_bytes=byte_size(out)+sum(byte_size(Path(r["evidence"])) for r in records if r.get("evidence"))
  d=dict(format=plan["format"],binding=plan["binding"],plan_sha256=sha(out/"plan.json"),planned=len(tasks),
   passed=sum(r["status"]=="passed" and r.get("audited") for r in records),
   failed=sum(r["status"]=="failed" for r in records),
   blocked=sum(r["status"]=="blocked_preflight" for r in records),
   not_run=len(tasks)-len(records),skipped=0,rows=records,seconds=time.monotonic()-start,
   artifact_bytes=evidence_bytes,paid_calls=0,agent_generated_successes=0,stop_reason=stop)
  dump(out/"report.json",d);return d
 for item in plan["tasks"]:
  require(snapshot()==sources,"Shared source changed during frozen acceptance")
  if time.monotonic()-start>=1800 or shutil.disk_usage(WORK).free<4*1024**3:
   stop="time_or_disk_boundary";report();break
  if item["status"]!="ready":
   records.append(copy.deepcopy(item));report();continue
  folder=out/item["task"]["id"]
  request=json.loads((folder/"request.json").read_text())
  args=parse_args(["--inside","--case",str(folder/"model.json"),"--golden-file",str(folder/"candidate.py"),
   "--unified-exercise",item["task"]["exercise"],"--compiler-configuration",cfg["name"],"--max-repairs","0"])
  if a.profile=="public":args.unified_profile="public-v1"
  terminal={}
  record=copy.deepcopy(item);tick=time.monotonic()
  try:
   with (folder/"run.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    code=inside(args,expected_request=request,report_sink=lambda r,p:terminal.update(report=copy.deepcopy(r),evidence=str(p)))
   require("report" in terminal,"Candidate terminated without report")
   result=terminal["report"];last=result["attempts"][-1] if result["attempts"] else result
   record.update(status="passed" if code==0 and result["status"]=="passed" else "failed",
    exit_code=code,evidence=terminal["evidence"],failure_layer=last.get("failure_layer"),
    diagnostic=last.get("diagnostic"),comparison=last.get("comparison"),coverage=last.get("public_expression_coverage" if a.profile=="public" else "packed_native_coverage"),
    compiled=last.get("compiled",False),executed=last.get("executed",False))
   if record["status"]=="passed":
    audited=verify_candidate(Path(terminal["evidence"]))
    record.update(audited=True,audit_comparison=audited["comparison"],audit_coverage=audited["coverage"])
  except Exception as exc:
   record.update(status="failed",failure_layer="acceptance_driver",diagnostic=str(exc)[:2000],
    evidence=terminal.get("evidence"),audited=False)
  record["seconds"]=time.monotonic()-tick
  records.append(record);summary=report()
  print(item["task"]["id"],item["task"]["requirement"],record["status"],record.get("failure_layer"),flush=True)
  if record.get("failure_layer") in ("integrity","environment","acceptance_driver"):
   stop="integrity_environment_or_audit_fault";report();break
  if summary["artifact_bytes"]>1024**3 or initial_global+summary["artifact_bytes"]>32*1024**3:
   stop="artifact_budget";report();break
 require(snapshot()==sources,"Final source binding changed")
 final=report()
 print(json.dumps({k:v for k,v in final.items() if k!="rows"}),flush=True)
 return int(final["failed"]+final["blocked"]+final["not_run"]>0)
if __name__=="__main__":raise SystemExit(main())
