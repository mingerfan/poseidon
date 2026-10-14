#!/usr/bin/env python3
"""Bounded Poseidon CPU acceptance of unchanged frozen mathematical contexts."""
import argparse,contextlib,copy,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
import numpy as np
from workspace_paths import WORK
from benchmark_graph import canonical,digest,require,samples
from benchmark_runner import load
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import candidate_source
from check_poseidon_helpers import snapshot
from run_candidate import inside,parse_args
from audit_unified_candidate import verify_candidate

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):
 temp=path.with_name(path.name+".partial");temp.write_bytes(canonical(value));temp.replace(path)
def byte_size(path):return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def population():
 suite=ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
 corpus,index=load(suite,check_sources=False)
 ledger=read(suite/"coverage.json");component=ledger["components"]["math_contexts"]
 path=suite/component["file"];require(sha(path)==component["sha256"],"Frozen mathematical component changed")
 ctx=read(path)
 models={r["model_sha256"]:r for r in corpus}
 for row in ctx["supplemental_models"]:
  require(digest(row["model"])==row["model_sha256"],"Supplemental model hash")
  require(row["model_sha256"] not in models,"Supplemental model duplicates corpus")
  models[row["model_sha256"]]=row
 prior_directed=WORK/"results/poseidon-directed-aggregate-r258.json"
 prior_helpers=WORK/"results/poseidon-semantic-contexts-r260/report.json"
 passed={}
 for row in read(prior_directed)["rows"]:
  require(row["status"]=="passed" and row.get("audited"),"Prior directed evidence must be accepted")
  passed.setdefault(row["model_sha256"],[]).append(dict(evidence=row["evidence"],parent_report=str(prior_directed),
      parent_report_sha256=sha(prior_directed),kind="scoped_directed_source_binding"))
 for row in read(prior_helpers)["historical_helpers"]:
  require(row["status"]=="historical_poseidon_binding_verified","Prior helper mapping must be verified")
  for entry in row["contexts"]:
   passed.setdefault(entry["model_sha256"],[]).append(dict(evidence=entry["evidence"],parent_report=str(prior_helpers),
       parent_report_sha256=sha(prior_helpers),kind="historical_helper_source_binding"))
 order=[];groups={}
 for task in ctx["tasks"]:
  require(digest({k:v for k,v in task.items() if k!="task_sha256"})==task["task_sha256"],"Frozen mathematical task hash")
  model=models[task["model_sha256"]]
  require(task["model_id"]==model["model"]["id"] and task["topology"]==model["topology"],"Task model/topology")
  if task["model_sha256"] not in groups:groups[task["model_sha256"]]=[];order.append(task["model_sha256"])
  groups[task["model_sha256"]].append(task)
 require(len(ctx["tasks"])==459 and len(order)==280 and len(ctx["supplemental_models"])==92,"Original population changed")
 covered=[h for h in order if h in passed]
 require(len(covered)==72,"Known original mathematical model population changed")
 remaining=[dict(model=models[h]["model"],model_sha256=h,topology=models[h]["topology"],
                 tasks=groups[h],supplemental=groups[h][0]["supplemental"]) for h in order if h not in passed]
 require(len(remaining)==208,"Remaining original model population changed")
 return suite,index,ctx,remaining,dict(components={str(path):sha(path)},prior_reports={str(p):sha(p) for p in [prior_directed,prior_helpers]},
                                     known_models=72,remaining_models=208,covered_task_count=sum(len(groups[h]) for h in covered))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--offset",type=int,default=0)
 p.add_argument("--limit",type=int,default=48);p.add_argument("--plan",action="store_true")
 p.add_argument("--case",action="append",help="Retest unchanged original model IDs only")
 p.add_argument("--compiler-configuration",choices=("poseidon-cpu-eva-w45-v1","poseidon-cpu-eva-w40-v1"),default="poseidon-cpu-eva-w45-v1")
 p.add_argument("--source-directory",type=Path,help="Explicit developer fixtures; never Agent outputs")
 a=p.parse_args();require(a.offset>=0 and 1<=a.limit<=48,"Mathematical shard bounds")
 out=a.output;require(out.resolve().parent==(WORK/"results").resolve() and not out.exists(),"Fresh result directory required")
 suite,index,contexts,allmodels,prior=population()
 if a.case:
  require(a.offset==0 and len(a.case)==len(set(a.case)) and len(a.case)<=a.limit,"Explicit retest population bounds")
  require(set(a.case)<={r["model"]["id"] for r in allmodels},"Unknown original model")
  allmodels=[r for r in allmodels if r["model"]["id"] in set(a.case)]
 selected=allmodels[a.offset:a.offset+a.limit];require(selected,"Empty original model shard")
 cfg=configuration(a.compiler_configuration);sources=snapshot()
 require(cfg.get("runtime_plan_entry_policy")=="hecate-private-symbol-dce-v1","Current entry policy required")
 start=time.monotonic();out.mkdir()
 plan=dict(format="poseidon-frozen-mathematical-acceptance-v1",backend="poseidon_cpu",evaluation_source="manual_rule_fixture_not_agent",
  offset=a.offset,limit=a.limit,planned=len(selected),prior=prior,original_tasks=len(contexts["tasks"]),
  original_models=1200,supplemental_models=92,model_set_regenerated=False,
  coverage_sha256=sha(suite/"coverage.json"),index_sha256=sha(suite/"index.json"),
  historical_generator_sources=index["source_hashes"],historical_generator_not_reused=True,
  compiler_configuration=cfg,sources=sources,runner_sha256=sha(Path(__file__)),
  max_wall_seconds=1800,max_artifact_bytes=1024**3,global_artifact_limit_bytes=32*1024**3,
  min_free_disk_bytes=4*1024**3,native_concurrency=1,compile_jobs=2,link_jobs=1,
  paid_calls=0,numeric_tolerance=dict(atol=1e-5,rtol=1e-4),models=[])
 for row in selected:
  folder=out/row["model"]["id"];folder.mkdir();(folder/"model.json").write_bytes(canonical(row["model"]))
  item={k:copy.deepcopy(row[k]) for k in ["model_sha256","topology","tasks","supplemental"]};item["model_id"]=row["model"]["id"];item["status"]="ready"
  try:
   require(digest(row["model"])==row["model_sha256"],"Frozen positive model changed")
   errors=[]
   for inputs in samples(row["model"],16):
    ref=mathematical(row["model"],inputs);other=torch_reference(row["model"],inputs)
    require(set(ref)==set(other),"Independent reference named outputs")
    for name in ref:
     np.testing.assert_allclose(ref[name],other[name],atol=1e-12,rtol=1e-12)
     errors.append(float(np.max(np.abs(ref[name]-other[name]))))
   item.update(reference_groups=16,reference_max_absolute_error=max(errors))
   request=prepare(row["model"],cfg["ckks_config_sha256"],cfg)
   if a.source_directory is None:
    source,lowering=candidate_source(request)
   else:
    fixture=a.source_directory/(row["model"]["id"]+".py")
    require(fixture.is_file() and not fixture.is_symlink(),"Missing explicit developer fixture")
    source=fixture.read_text()
    lowering=dict(strategy="explicit_manual_fixture",source_path=str(fixture),source_sha256=sha(fixture),
                  request_changed=False,reference_changed=False)
   
   validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
   (folder/"request.json").write_bytes(canonical(request));(folder/"candidate.py").write_text(source)
   item.update(request_id=request["request_id"],candidate_sha256=sha(folder/"candidate.py"),lowering=lowering)
  except (ValueError,TypeError,IndexError,AssertionError) as error:
   item.update(status="blocked_preflight",failure_layer="static_or_reference_or_rule_lowering",diagnostic=str(error)[:3000])
  plan["models"].append(item)
 require(snapshot()==sources,"Source changed during model preparation")
 plan["binding"]=digest(plan);dump(out/"plan.json",plan)
 if a.plan:
  print(json.dumps(dict(binding=plan["binding"],planned=len(selected),ready=sum(r["status"]=="ready" for r in plan["models"]),
                       blocked=sum(r["status"]!="ready" for r in plan["models"]),paid_calls=0)),flush=True);return 0
 initial_global=byte_size(WORK/"results");records=[];stop=None
 def report():
  used=byte_size(out)+sum(byte_size(Path(r["evidence"])) for r in records if r.get("evidence"))
  d=dict(format=plan["format"],binding=plan["binding"],plan_sha256=sha(out/"plan.json"),planned=len(selected),
    passed=sum(r["status"]=="passed" and r.get("audited") for r in records),
    failed=sum(r["status"]=="failed" for r in records),blocked=sum(r["status"]=="blocked_preflight" for r in records),
    not_run=len(selected)-len(records),skipped=0,rows=records,artifact_bytes=used,seconds=time.monotonic()-start,
    stop_reason=stop,paid_calls=0,agent_generated_successes=0)
  dump(out/"report.json",d);return d
 for item in plan["models"]:
  require(snapshot()==sources,"Shared source changed during frozen execution")
  if time.monotonic()-start>=1800 or shutil.disk_usage(WORK).free<4*1024**3:
   stop="time_or_disk_boundary";report();break
  record=copy.deepcopy(item)
  if item["status"]!="ready":records.append(record);report();continue
  folder=out/item["model_id"];request=read(folder/"request.json")
  args=parse_args(["--inside","--case",str(folder/"model.json"),"--golden-file",str(folder/"candidate.py"),
                   "--compiler-configuration",cfg["name"],"--max-repairs","0"])
  terminal={};tick=time.monotonic()
  try:
   with (folder/"run.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    code=inside(args,expected_request=request,report_sink=lambda r,p:terminal.update(report=copy.deepcopy(r),evidence=str(p)))
   require("report" in terminal,"Candidate missing terminal report")
   result=terminal["report"];last=result["attempts"][-1] if result["attempts"] else result
   record.update(status="passed" if code==0 and result["status"]=="passed" else "failed",exit_code=code,evidence=terminal["evidence"],
      failure_layer=last.get("failure_layer"),diagnostic=last.get("diagnostic"),comparison=last.get("comparison"),
      compiled=last.get("compiled",False),executed=last.get("executed",False))
   if record["status"]=="passed":
    audited=verify_candidate(Path(terminal["evidence"]))
    record.update(audited=True,audit_comparison=audited["comparison"],audit_request_id=audited["request_id"])
  except Exception as error:
   record.update(status="failed",failure_layer="acceptance_driver",diagnostic=str(error)[:3000],evidence=terminal.get("evidence"),audited=False)
  record["seconds"]=time.monotonic()-tick;records.append(record);d=report()
  print(item["model_id"],record["status"],record.get("failure_layer"),flush=True)
  if record.get("failure_layer") in ("integrity","environment","acceptance_driver"):
   stop="integrity_environment_or_audit_fault";report();break
  if d["artifact_bytes"]>1024**3 or initial_global+d["artifact_bytes"]>32*1024**3:
   stop="artifact_budget";report();break
 require(snapshot()==sources,"Final source binding changed")
 final=report();print(json.dumps({k:v for k,v in final.items() if k!="rows"}),flush=True)
 return int(final["failed"]+final["blocked"]+final["not_run"]>0)
if __name__=="__main__":raise SystemExit(main())
