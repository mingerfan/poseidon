"""Unchanged prior-Agent DSL replay through Application; no API implementation."""
import argparse,copy,json,math,os,shlex,shutil,signal,sys,time
from pathlib import Path
from replay_scope import *
from exact_source import ExactAgentReplay

LOCK=RESULTS/"agent-static-broadcast-replay-v1.lock"
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_bytes(origin_scope.canonical(value));os.chmod(tmp,0o600);tmp.replace(path)
def boundary(plan,p):
 require(time.time()<plan["deadline"],"replay_batch_time_limit")
 require(tool_hashes()==p["tool_hashes"],"replay_tools_changed_during_run")
 total=supervision.measure(RESULTS)
 require(total<=BOUNDS["global_result_bytes"]and total-plan["initial_bytes"]<=BOUNDS["new_result_bytes"]and
  shutil.disk_usage(WORK).free>=BOUNDS["minimum_free_bytes"],"replay_storage_limit")
def numeric(actual,reference):
 from numerical_results import compare
 value=compare(actual,reference,1e-5,1e-4)
 require(value["passed"],"replay_frozen_numeric_tolerance")
 return value
def one(case,p,plan):
 import numpy as np
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from candidate_bundle import check_bundle
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from program_execution import package_execution_metadata
 from pytorch_source_contract import restore_outputs
 out=RUN/case["id"];out.mkdir(mode=0o700)
 started=time.monotonic();deadline=min(plan["deadline"],time.time()+BOUNDS["case_wall_seconds"])
 def checked():
  boundary(plan,p);require(time.time()<deadline,"replay_case_time_limit")
 checked()
 source=case["source"];q=read(source["request_file"]);old=case["origin"]
 raw=read(old["response_file"])["raw"];receipt=read(old["receipt_file"])
 replay=ExactAgentReplay(q,raw,receipt,generation_identity_sha256=old["generation_identity_sha256"])
 qualified=Path(source["qualification"]["source_task_directory"])
 require(origin_scope.regular(qualified/"task.json")==qualified/"task.json"and
  sha(qualified/"task.json")==source["qualification"]["source_task_metadata_sha256"],
  "replay_exact_qualified_metadata")
 app=application(WorkerContext.capture(qualified.parent))
 require(app.backend.identity_manifest()==p["stage_identity"],"replay_current_backend_identity")
 options=dict(origin_scope.OPTIONS,capacity_packing_version="capacity-public-tensors-v9")
 item=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",
  manifest=str(ROOT/source["source_manifest"]),trust_local_code=True)
 limits=dict(generations=1,http_attempts=1,wall_seconds=max(1,int(deadline-time.time())),
  api_timeout_seconds=1200,max_tokens=384000)
 task=qualified.name;state=read(qualified/"state.json")
 require(state["status"]=="queued"and state["generations"]==state["http_attempts"]==0 and not state["ledger"]and
  state["intake"]["reference_verified"]is True,"replay_unused_qualified_task")
 write(out/"task-id.json",dict(task_id=task,task_directory=str(qualified),source_capture_reused=True))
 meta=app._checked_task_metadata(task)
 require(meta["request"]==q and meta["identity"]==p["stage_identity"],"replay_exact_request_and_identity")
 checked();result=app.run(task,replay);write(out/"application-result.json",result)
 if result["status"]!="succeeded":
  return dict(id=case["id"],status="failed",failure=result.get("failure"),task_id=task,
   paid_generations=0,http_attempts=result["http_attempts"],new_agent_generation=False)
 require(replay.calls==1 and result["http_attempts"]==0 and result["package"]["reused"]is False,
  "replay_single_unchanged_candidate")
 package=app._dir(task)/"program";manifest=check_bundle(package)
 require(manifest["agent_generated"]is False and
  sha(package/"candidate.py")==old["candidate_sha256"]and
  (package/"candidate.py").read_text()==json.loads(raw)["hecate_source"],"replay_honest_original_source_package")
 require(manifest["validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True),
  "replay_numerical_package")
 qual=read(app._dir(task)/"validation-0.json")
 require(qual["compiled_validated"]and qual["encrypted_execution"]and qual["numerically_validated"],
  "replay_real_initial_qualification")
 evidence=Path(qual["evidence"]);attempt=read(evidence/"attempt-00/report.json")
 require(attempt["compiled"]is True and attempt["executed"]is True and
  attempt["execution"]["backend"]=="poseidon_cpu"and attempt["execution"]["executor_secret_key"]is False,
  "replay_true_poseidon_execution")
 model=q["model"];samples=[];expected=[]
 for index in range(3):
  rng=np.random.default_rng(20261008+index)
  inputs={s["name"]:rng.uniform(-.2,.2,s["shape"]).astype(np.float64)for s in model["inputs"]}
  a=math_ref(model,inputs);b=torch_ref(model,inputs)
  require(set(a)==set(b)=={o["name"]for o in model["outputs"]},"replay_reference_names")
  for name in a:np.testing.assert_allclose(a[name],b[name],atol=2e-12,rtol=2e-12)
  samples.append({name:value.tolist()for name,value in inputs.items()})
  expected.append(np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]]))
 tree=package_execution_metadata(package,manifest,model)["output_structure"];phases={}
 for name,batch,refs in (("new_inputs",samples,expected),("repeat",samples[:1],expected[:1])):
  checked();execution=app.execute(task,batch)
  require(execution["encrypted_execution"]is True and execution["executor_secret_key"]is False and
   execution["agent_calls"]==execution["compiler_calls"]==0 and
   execution["backend"]=="poseidon_cpu","replay_reloaded_encrypted_execution")
  actual=np.asarray([np.concatenate([np.asarray(row[o["name"]],dtype=np.float64).reshape(-1)
   for o in model["outputs"]])for row in execution["outputs"]])
  comparison=numeric(actual,np.asarray(refs,dtype=np.float64))
  require(execution.get("output_structure")==tree,"replay_output_structure")
  if tree is not None:
   require(execution["structured_outputs"]==[restore_outputs(values,tree,allow_none=True)
    for values in execution["outputs"]],"replay_original_output_tree")
  require(check_bundle(package)==manifest,"replay_package_integrity_after_reload")
  phases[name]=dict(status="passed",execution=execution,comparison=comparison,
   input_groups=len(batch),sample_format="named",paid_generations=0,http_attempts=0,
   compiled=False,encrypted_execution=True,package_reloaded=True)
  write(out/(name+".json"),phases[name])
 checked();again=app.submit_source(item,resource_profile="general-tensors-v1",options=options,limits=limits)
 class NeverGenerate:
  kind="scripted_replay";agent_calls=0
  def generate(self,*args,**kwargs):raise AssertionError("replay_own_source_cache_required")
 cached=app.run(again,NeverGenerate())
 require(cached["status"]=="succeeded"and cached["package"]["reused"]is True and
  cached["package"]["binding"]==manifest["binding"]and cached["generations"]==cached["http_attempts"]==0,
  "replay_same_source_cache")
 write(out/"cache.json",cached)
 require(app.backend.identity_manifest()==p["stage_identity"],"replay_final_backend_identity")
 return dict(id=case["id"],status="passed",task_id=task,task_directory=str(qualified),source_capture_reused=True,request_id=q["request_id"],
  source_context=case["context"],representative_case=case["representative_case"],
  original_generation_receipt_binding=replay.origin["receipt_binding"],
  original_generation_identity_sha256=old["generation_identity_sha256"],
  candidate_sha256=old["candidate_sha256"],new_agent_generation=False,paid_generations=0,http_attempts=0,
  scripted_delivery_attempts=1,compiled=True,encrypted_execution=True,package=str(package),
  package_binding=manifest["binding"],package_origin="scripted_replay; original genuine generation bound separately",
  qualification=qual,new_inputs=phases["new_inputs"],repeat=phases["repeat"],cache_reused=True,
  input_groups=8,compared_values=qual["compared_values"]+sum(np.asarray(x).size for x in expected)+np.asarray(expected[0]).size,
  maximum_absolute_error=max(qual["max_absolute_error"],*(v["comparison"]["max_absolute_error"]for v in phases.values())),
  elapsed_seconds=time.monotonic()-started,source_unchanged=True,original_application_accuracy_certified=False)
def inside():
 from hecate_python_env import VENV
 from application_backend import PoseidonBackend
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"replay_pinned_environment")
 p=verify();plan=read(RUN/"plan.json");bound(plan)
 require(plan["proposal_binding"]==p["binding"]and supervision.same_process(plan["owner"]),
  "replay_actual_owner")
 require(PoseidonBackend().identity_manifest()==p["stage_identity"],"replay_start_backend_identity")
 rows=[]
 try:
  for case in p["cases"]:
   boundary(plan,p)
   if case["status"]=="blocked_origin_failure":
    row=dict(id=case["id"],status="blocked_origin_failure",failure=case.get("origin_failure"),paid_generations=0,http_attempts=0)
   else:
    started=time.monotonic()
    try:row=one(case,p,plan)
    except Exception as error:
     row=dict(id=case["id"],status="failed",failure=dict(layer="offline_replay",code=type(error).__name__,
      diagnostic=str(error)[:300]),paid_generations=0,http_attempts=0,elapsed_seconds=time.monotonic()-started)
     # Integrity/environment ceilings stop the affected batch.
     if any(word in str(error)for word in("identity","changed","storage_limit","time_limit","pinned","scope","binding")):
      write(RUN/case["id"]/"case-report.json",row);rows.append(row);raise
   rows.append(row);write(RUN/case["id"]/"case-report.json",row)if(RUN/case["id"]).exists()else None
   summary=terminal(rows,p["cases"]);write(RUN/"report.json",dict(status="running",proposal_binding=p["binding"],rows=rows,**summary))
   print(json.dumps(dict(case=case["id"],status=row["status"],new_agent_generation=False,paid_generations=0,http_attempts=0)),flush=True)
 except Exception as error:
  summary=terminal(rows,p["cases"]);write(RUN/"report.json",dict(status="stopped",rows=rows,**summary,
   stop=dict(code=type(error).__name__,diagnostic=str(error)[:300]),proposal_binding=p["binding"]))
  raise
 summary=terminal(rows,p["cases"]);report=dict(status="complete",proposal_binding=p["binding"],rows=rows,**summary)
 report["binding"]=digest(report);write(RUN/"report.json",report)
 return 0 if summary["benchmark_status"]=="passed"else 1
def execute():
 from hecate_python_env import enter_nix,VENV
 p=verify();supervision.no_active();require(not RUN.exists(),"replay_fresh_run")
 with origin_scope.owned_lock(LOCK):
  initial=supervision.measure(RESULTS)
  require(initial<=BOUNDS["global_result_bytes"]and shutil.disk_usage(WORK).free>=BOUNDS["minimum_free_bytes"],
   "replay_initial_storage")
  RUN.mkdir(mode=0o700);now=time.time()
  plan=dict(format="poseidon-static-broadcast-replay-run-v1",proposal_binding=p["binding"],
   owner=supervision.process_identity(os.getpid()),started=now,deadline=now+BOUNDS["batch_wall_seconds"],initial_bytes=initial)
  plan["binding"]=digest(plan);write(RUN/"plan.json",plan)
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  tree=supervision.OwnedTree().start()
  try:code=enter_nix(command,seconds=BOUNDS["batch_wall_seconds"]+20,keep_env=(),environment="poseidon-cpu")
  finally:tree.finish()
  return code
if __name__=="__main__":
 parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument("--execute",action="store_true");mode.add_argument("--inside",action="store_true")
 args=parser.parse_args()
 raise SystemExit(inside()if args.inside else execute())
