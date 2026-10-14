"""Bounded unchanged historical Agent DSL replay through the current Application."""
import sys,json,time,os,shlex,fcntl,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(B/"benchmarks/tools/generic_capacity_backend_v10"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
from application_backend import PoseidonBackend
def verify():
 p=q.read(HERE/"scope.json");q.bound(p)
 for path,want in p["frozen_files"].items():q.require(q.sha(path)==want,"frozen:"+path)
 q.require(PoseidonBackend().identity_manifest()==q.read(HERE/"identity.json"),"sdk")
 return p
def save(path,v):
 v=dict(v);v["binding"]=q.digest(v);q.write(path,v);return v
def inside():
 import numpy as np
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from candidate_bundle import check_bundle
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from numerical_results import compare
 p=verify();out=Path(p["campaign"]);owner=q.read(out/"owner.json")
 last=0.
 def boundary(force=False):
  nonlocal last
  verify();q.require(time.time()<owner["deadline"],"wall_budget")
  q.require(shutil.disk_usage(out).free>=p["budget"]["reserve_bytes"],"disk_reserve")
  if force or time.monotonic()-last>=10:
   q.require(live.measure(out)<=p["budget"]["new_artifact_bytes"]and
    live.measure(live.RESULTS)<=p["budget"]["global_result_bytes"],"artifact_budget");last=time.monotonic()
 rows=[]
 try:
  for c in p["cases"]:
   boundary(True);folder=out/c["id"];folder.mkdir(mode=0o700)
   app=application(WorkerContext.capture(folder/"tasks"))
   request=q.read(c["request_file"])
   source=dict(format=FORMAT,kind="graph",model=request["model"])
   limits=dict(generations=1,http_attempts=1,wall_seconds=180,api_timeout_seconds=1200,max_tokens=384000)
   task=app.submit_source(source,resource_profile="general-tensors-v1",options=c["options"],limits=limits)
   state=app.prepare_source(task);q.require(state["status"]=="queued","graph_prepare:"+str(state.get("failure")))
   q.require(app._checked_task_metadata(task)["request"]==request,"same_request")
   candidate=Path(c["candidate"]).read_text()
   class Replay:
    kind="scripted_replay";agent_calls=0
    def generate(self,current,feedback=None):
     boundary();q.require(current==request and feedback==[],"single_exact_replay")
     return json.dumps(dict(schema=1,request_id=current["request_id"],hecate_source=candidate))
   result=app.run(task,Replay());save(folder/"qualification.json",result)
   row=dict(case=c["id"],status="failed",origin="unchanged_historical_real_agent_dsl",new_agent_generation=False,
    original_response_sha256=c["raw_response_sha256"],candidate_sha256=c["candidate_sha256"],
    original_case_report_sha256=c["prior_case_report_sha256"],request_id=request["request_id"],task_id=task,
    result=result,source_route="same_frozen_graph; not new trusted PyTorch intake")
   if result["status"]!="succeeded":
    save(folder/"case-report.json",row);rows.append(row)
    if (result.get("failure")or{}).get("layer")in("environment","integrity","resources"):raise ValueError("shared_failure")
    continue
   a=result["acceptance"]
   q.require(all(a[k]is True for k in ("compiled","artifact_checked","encrypted_execution","numerically_validated"))
      and a["input_groups"]==4 and result["http_attempts"]==0,"fresh_real_numerical")
   package=app._dir(task)/"program";bundle=check_bundle(package)
   q.require((package/"candidate.py").read_bytes()==Path(c["candidate"]).read_bytes(),"unchanged_candidate_bytes")
   extra=[]
   if c["new_inputs"]:
    samples=[];references=[]
    for i in range(3):
     rng=np.random.default_rng(20261013+i)
     values={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in request["model"]["inputs"]}
     m,t=math_ref(request["model"],values),torch_ref(request["model"],values)
     for k in m:np.testing.assert_allclose(m[k],t[k],atol=1e-12,rtol=1e-12)
     samples.append({k:v.tolist()for k,v in values.items()})
     references.append(np.concatenate([m[o["name"]].reshape(-1)for o in request["model"]["outputs"]]))
    for name,indices in (("new_inputs",[0,1,2]),("repeat",[0])):
     boundary(True);execution=app.execute(task,[samples[i]for i in indices])
     actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)
       for o in request["model"]["outputs"]])for v in execution["outputs"]])
     cmp=compare(actual,np.asarray([references[i]for i in indices]),1e-5,1e-4)
     q.require(cmp["passed"]and execution["encrypted_execution"]is True and
       execution["agent_calls"]==execution["compiler_calls"]==0 and execution["executor_secret_key"]is False,"reuse_encrypted_numeric")
     save(folder/(name+".json"),dict(execution=execution,comparison=cmp))
     extra.append(dict(phase=name,groups=len(indices),values=cmp["compared_values"],maxabs=cmp["max_absolute_error"]))
   boundary(True)
   again=app.submit_source(source,resource_profile="general-tensors-v1",options=c["options"],limits=limits)
   class Never:
    kind="scripted_replay";agent_calls=0
    def generate(self,*a):raise AssertionError("cache_generated")
   cached=app.run(again,Never())
   q.require(cached["status"]=="succeeded"and cached["package"]["reused"]is True and
      cached["package"]["binding"]==bundle["binding"]and cached["generations"]==cached["http_attempts"]==0,"cache")
   q.require(check_bundle(package)==bundle,"immutable_package")
   save(folder/"cache.json",cached)
   row.update(status="passed",package=str(package),package_binding=bundle["binding"],package_reloaded=True,
      package_reused=True,extra_execution=extra);save(folder/"case-report.json",row);rows.append(row)
   print(json.dumps(dict(case=c["id"],status="passed",candidate_unchanged=True)),flush=True)
  boundary(True)
 except BaseException as e:
  report=save(out/"report.json",dict(format="poseidon-static-diagnostic-historical-replay-v1",status="stopped",
    planned=len(p["cases"]),passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),
    not_run=len(p["cases"])-len(rows),cases=rows,error_type=type(e).__name__,diagnostic=str(e)[:240],scope_binding=p["binding"],
    sdk=p["sdk"],provider_calls=0,new_agent_generation=False,compiler_calls_limit=2,encrypted_calls_limit=4))
  raise
 report=save(out/"report.json",dict(format="poseidon-static-diagnostic-historical-replay-v1",
   status="passed"if all(r["status"]=="passed"for r in rows)else"failed",planned=len(p["cases"]),
   passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),not_run=0,skipped=0,
   cases=rows,scope_binding=p["binding"],sdk=p["sdk"],provider_calls=0,new_agent_generation=False,
   compiler_calls_confirmed=sum(r["result"].get("acceptance",{}).get("compiled")is True for r in rows),
   encrypted_calls_confirmed=sum(r["result"].get("acceptance",{}).get("encrypted_execution")is True for r in rows)+sum(len(r.get("extra_execution",[]))for r in rows)))
 print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
 return 0 if report["status"]=="passed"else 1
def outer():
 p=verify();live.no_active();out=Path(p["campaign"]);q.require(not out.exists(),"fresh_campaign")
 q.require(live.measure(live.RESULTS)+p["budget"]["new_artifact_bytes"]<=p["budget"]["global_result_bytes"]and
  shutil.disk_usage(live.RESULTS).free>=p["budget"]["reserve_bytes"]+p["budget"]["new_artifact_bytes"],"initial_storage")
 with (HERE/"execution.lock").open("a+")as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(mode=0o700);now=time.time()
  save(out/"owner.json",dict(pid=os.getpid(),start=live.process_info(os.getpid())["start"],deadline=now+p["budget"]["wall_seconds"]))
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  tree=live.OwnedTree().start()
  try:code=enter_nix(command,seconds=p["budget"]["wall_seconds"],environment="poseidon-cpu")
  finally:background=tree.finish()
  q.require(not background,"owned_background");verify()
  save(out/"terminal.json",dict(exit_code=code,finished=time.time(),scope_binding=p["binding"]))
  return code
if __name__=="__main__":
 if sys.argv[1:]==["--execute"]:raise SystemExit(outer())
 if sys.argv[1:]==["--inside"]:raise SystemExit(inside())
 raise SystemExit("replay.py --execute")
