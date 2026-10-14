"""Continue only unfinished package reuse; retain every qualification and prior execution."""
import copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from numerical_results import compare
from semantic_benchmark_execution import runtime_sources
from application_execution.context import WorkerContext
from application_execution.backend import application
from candidate_bundle import replay_bundle,check_execution_dependencies,load_model_provenance
from model_output_contract import build,restore
from program_execution import pack_samples
from poseidon_validation import input_payload
work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
original=work/"agent-common-pytorch-backend-r1082";out=work/"agent-common-pytorch-reuse-continuation-r1091"
pre=work/"agent-common-pytorch-offline-prepared-r1040"
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proposal=load(ROOT/"docs/baseline/agent-common-pytorch-reuse-continuation-r1092.json")
require(not out.exists(),"Fresh continuation; no automatic repetition")
for name,h in proposal["retained_files"].items():require(sha(name)==h,"Original qualification evidence")
source=load(original/"report.json");oldplan=load(original/"plan.json")
require(source["status"]=="stopped" and source["not_run"]==0 and source["planned"]==151,"All qualifications already terminal")
scope=load(pre/"approved-backend-scope-r1082.json");require(scope["deadline_wall"]>time.time(),"Original deadline remains")
record=load(original/"tasks/worker-context.json");binding=record.pop("binding")
context=WorkerContext(canonical(record));require(context.binding==binding,"Original frozen worker");app=application(context)
require(runtime_sources()==oldplan["source_hashes"] and app.backend.identity()==oldplan["backend_identity"],"Unchanged source/backend")
out.mkdir(mode=0o700);rows=copy.deepcopy(source["rows"]);started=time.monotonic()
plan=copy.deepcopy(oldplan);plan.pop("binding");plan.update(
 retained_plan_binding=oldplan["binding"],retained_plan_file_sha256=sha(original/"plan.json"),
 retained_report_file_sha256=sha(original/"report.json"),continuation_proposal_binding=proposal["binding"],
 continuation_tool_sha256=sha(Path(__file__)),continuation_audit_tool_sha256=sha(Path(__file__).with_name("audit.py")),
 source_evidence_root=str(original),scope=scope,continuation_reason="Use explicit output projection; preserve completed ciphertext executions")
plan["binding"]=digest(plan);(out/"plan.json").write_bytes(canonical(plan))
def measured(p,allocated=False):
 argv=["du","-s","--block-size=1","--one-file-system",str(p)]
 if not allocated:argv.insert(1,"--apparent-size")
 return int(subprocess.check_output(argv,text=True,timeout=30).split()[0])
def boundary():
 require(scope["deadline_wall"]-time.time()>8,"Original cumulative offline time budget")
 require(runtime_sources()==oldplan["source_hashes"] and app.backend.identity()==oldplan["backend_identity"],"Frozen current code and backend")
 require(all(sha(p)==h for p,h in proposal["retained_files"].items()),"Retained evidence unchanged")
 require(all(sha(p)==h for p,h in proposal["tool_hashes"].items()),"Frozen continuation tools")
 require(shutil.disk_usage(work).free>=scope["reserve_bytes"],"Disk reserve")
 require(max(measured(work),measured(work,True))<=scope["global_bytes"],"Global artifact limit")
 initial=set(scope["initial_result_paths"])
 new=sum(max(measured(p),measured(p,True)) for p in work.iterdir() if p.name not in initial)
 new+=max(0,max(measured(pre),measured(pre,True))-scope["initial_pre_bytes"])
 require(new<=scope["artifact_bytes"],"Original cumulative artifact limit")
def save(status):
 report={k:v for k,v in source.items() if k not in ("rows","binding","status","seconds")}
 report.update(status=status,binding=plan["binding"],rows=rows,seconds=time.monotonic()-started,
   original_qualification_seconds=source["seconds"],source_evidence_root=str(original),qualification_reexecutions=0,
   continued_reuse=sum(r.get("reuse_status")=="passed" for r in rows)-7)
 (out/"report.json").write_bytes(canonical(report))
class NeverGenerate:
 kind="scripted_replay";agent_calls=0
 def generate(self,*args):raise AssertionError("Verified existing package expected")
def flatten(value,tree):
 if tree["kind"]=="tensor":return [np.asarray(value).reshape(-1)]
 if tree["kind"]=="dict":return [v for item in tree["items"] for v in flatten(value[item["key"]],item["value"])]
 return [a for v,t in zip(value,tree["items"]) for a in flatten(v,t)]
case_by_id={c["id"]:c for c in oldplan["cases"]}
save("running")
try:
 for name in load(ROOT/"docs/baseline/agent-common-pytorch-offline-proposal-r1081.json")["package_reuse_cases"]:
  row=next(r for r in rows if r["id"]==name)
  if row["status"]!="passed":
   row["reuse_status"]="not_run_original_failed";save("running");continue
  if row.get("reuse_status")=="passed":
   row["supplied_inputs_file"]=str(original/(name+"-new-inputs.json"));save("running");continue
  boundary();program=Path(row["package"]);metadata,request=check_execution_dependencies(program)
  if "reload" not in row:
   before=set(work.iterdir());require(replay_bundle(program)==0,"Real program reload")
   new=[p for p in set(work.iterdir())-before if p.name.startswith("poseidon-package-replay-")]
   require(len(new)==1,"Unique reload execution");row["reload"]=load(new[0]/"report.json")
   require(row["reload"]["encrypted_execution"] and row["reload"]["numerically_validated"],"Reload numeric evidence");save("running")
  model=load(Path(row["imported"])/"model.json");provenance=load_model_provenance(Path(row["imported"]),model)
  logical=[];expected=[]
  for probe in range(3):
   rng=np.random.default_rng(20261027+probe);values={s["name"]:rng.uniform(-.3,.3,s["shape"]) for s in model["inputs"]}
   a,b=mathematical(model,values),tensor_reference(model,values)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
   logical.append({k:v.tolist() for k,v in values.items()})
   expected.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
  path=original/(name+"-new-inputs.json")
  if path.exists():require(load(path)==logical,"Original approved supplied inputs preserved")
  else:path=out/(name+"-new-inputs.json");path.write_bytes(canonical(logical))
  row["supplied_inputs_file"]=str(path)
  tree=provenance["reference_report"].get("capture_metadata",{}).get("output_structure")
  contract=build(model,tree,source_binding=digest(dict(request_id=request["request_id"],provenance=digest(provenance)))) if tree is not None else None
  for repetition in range(len(row["new_input_runs"]),2):
   boundary();executed=None
   if name=="v8_subtract_tensor_left_operator" and repetition==0:
    retained=Path(proposal["retained_partial_execution"])
    raw=load(retained/"report.json")
    require(raw["bundle_binding"]==row["package_binding"] and raw["request_id"]==request["request_id"] and raw["samples"]==3,"Exact partial execution identity")
    require(load(retained/"output/inputs.json")==input_payload(pack_samples(request,logical),request["layout"]),"Exact physical input binding")
    executed=raw;origin="retained_actual_execution; no duplicate ciphertext run"
   else:executed=app.execute(row["task_id"],logical);origin="new_approved_reuse_execution"
   require(executed["status"]=="executed" and executed["encrypted_execution"] and not executed["executor_secret_key"],"Actual ciphertext execution")
   require(executed["agent_calls"]==executed["compiler_calls"]==0,"No generation or recompilation")
   actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1) for o in model["outputs"]]) for s in executed["outputs"]])
   if contract is not None:
    projected=[restore(contract,s,expected_binding=contract["binding"]) for s in executed["outputs"]]
    structured=np.asarray([np.concatenate(flatten(s,tree)) for s in projected])
    require(np.array_equal(structured,actual),"Explicit checked output projection preserves decoded values")
   comparison=compare(actual,np.asarray(expected),1e-5,1e-4);require(comparison["passed"],"Frozen pointwise threshold")
   row["new_input_runs"].append(dict(repetition=repetition,evidence=executed["evidence"],comparison=comparison,
      origin=origin,output_projection="explicit checked capture tree; flat named execution API"))
   save("running")
  case=case_by_id[name]
  limits=dict(generations=1,http_attempts=1,wall_seconds=min(180,int(scope["deadline_wall"]-time.time())-20),api_timeout_seconds=1200,max_tokens=384000)
  cached=app.submit_source(case["source_api"],**case["source_requirements"],limits=limits) if case.get("source_api") is not None else app.submit_import(row["imported"],options=case["options"],limits=limits)
  result=app.run(cached,NeverGenerate())
  require(result["status"]=="succeeded" and result["package"]["reused"] and result["generations"]==result["http_attempts"]==0,"Verified cache without model call")
  row["cache_reuse"]=dict(status="succeeded",reused=True,generations=0,http_attempts=0);row["reuse_status"]="passed"
  save("running");print(json.dumps(dict(id=name,reuse="passed",qualification_reexecutions=0,api_calls=0)),flush=True)
 boundary();save("complete")
except BaseException:
 save("stopped");raise
