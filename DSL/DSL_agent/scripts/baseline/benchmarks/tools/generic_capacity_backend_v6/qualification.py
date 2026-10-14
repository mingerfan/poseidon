"""Finite qualification through existing source, task, generation and package APIs."""
import fcntl,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as s
from hecate_python_env import VENV,enter_nix
def save(out,data):
 data=dict(data);data.pop("binding",None);data["binding"]=s.digest(data)
 tmp=out/"report.tmp";tmp.write_bytes(s.canonical(data));os.chmod(tmp,0o600);tmp.replace(out/"report.json")
 return data
def identity(plan):
 from application_backend import PoseidonBackend
 s.require(PoseidonBackend().identity_manifest()==s.read(HERE/"identity-snapshot.json"),"current_sdk")
def prepare(app,case,mode,remaining):
 from application_model_source import FORMAT
 from application_planning.layout import check_plan
 frozen=s.read(case["request_file"])
 selected=check_plan(s.read(case["layout_plan"]),frozen["model"])
 s.require(selected==frozen and case["options"]==s.read(case["layout_plan"])["selection"]["options"],"planning_same_request")
 source=(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],trust_local_code=True)
   if case["execution_source_kind"]=="trusted_pytorch" else
   dict(format=FORMAT,kind="graph",model=case["execution_source_graph"]))
 budget=dict(generations=1 if mode=="offline"else 4,http_attempts=1 if mode=="offline"else 16,
  wall_seconds=remaining,api_timeout_seconds=1200,max_tokens=384000)
 task=app.submit_source(source,resource_profile="general-tensors-v1",options=s.options_for(case),limits=budget)
 state=app.prepare_source(task)
 s.require(state["status"]=="queued","source_prepared:"+str(state.get("failure")))
 s.require(app._checked_task_metadata(task)["request"]==s.read(case["request_file"]),"same_frozen_request")
 return task
def package_checks(app,task,case,folder,checkpoint):
 import numpy as np
 from candidate_bundle import check_bundle
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from numerical_results import compare
 from program_execution import package_execution_metadata
 from application_model_source import FORMAT
 from pytorch_source_contract import restore_outputs
 before=check_bundle(app._dir(task)/"program")
 meta=app._checked_task_metadata(task);model=meta["request"]["model"]
 structure=package_execution_metadata(app._dir(task)/"program",before,model)["output_structure"]
 results={}
 for phase,count in ((("new_inputs",3),("repeat",1)) if case["id"]in s.REUSE_CASES else ()):
  checkpoint();samples=[];expected=[]
  for i in range(count):
   rng=np.random.default_rng(20261008+i)
   values={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
   a,b=math_ref(model,values),torch_ref(model,values)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
   samples.append({k:v.tolist()for k,v in values.items()})
   expected.append(np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]]))
  output=app.execute(task,samples)
  actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)
    for o in model["outputs"]])for v in output["outputs"]])
  compared=compare(actual,np.asarray(expected,dtype=np.float64),1e-5,1e-4)
  s.require(output["encrypted_execution"]is True and output["agent_calls"]==output["compiler_calls"]==0
    and output["executor_secret_key"]is False and compared["passed"],"package_encrypted_numerical")
  pre=output["resource_precheck"];ev=Path(output["evidence"])/"output"
  s.require(s.read(ev/"resource-precheck.json")==pre and
    s.read(ev/"execution-metadata.json")==output["execution"]and pre["failure"]is None,"resource_receipt")
  s.require(output.get("output_structure")==structure,"output_structure")
  if structure is not None:s.require(output["structured_outputs"]==
    [restore_outputs(v,structure,allow_none=True)for v in output["outputs"]],"structured_output")
  s.write(folder/(phase+".json"),dict(status="passed",execution=output,comparison=compared))
  results[phase]=dict(groups=count,values=compared["compared_values"],maximum_absolute_error=compared["max_absolute_error"])
 checkpoint()
 source=s.read(app._dir(task)/"source.json")
 again=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",
   manifest=case["source_manifest"],trust_local_code=True),resource_profile="general-tensors-v1",
   options=source["requirements"]["options"],limits=source["budget"])
 class Never:
  kind="scripted_replay";agent_calls=0
  def generate(self,*args):raise AssertionError("cache_must_not_generate")
 result=app.run(again,Never())
 s.require(result["status"]=="succeeded"and result["package"]["reused"]is True
  and result["package"]["binding"]==before["binding"]and
  result["generations"]==result["http_attempts"]==0,"source_package_cache")
 s.require(check_bundle(app._dir(task)/"program")==before,"package_unchanged")
 s.write(folder/"cache.json",result)
 results.update(package_reloaded=True,package_reused=True,package_binding=before["binding"])
 return results
def one(mode,name):
 p=s.verify(mode);out=Path(p["campaign"]);folder=out/name;owner=s.read(out/"owner.json")
 c=next(c for c in p["cases"]if c["id"]==name);deadline=s.read(folder/"case-boundary.json")["deadline"]
 def checkpoint():
  s.verify(mode);data=s.boundary(p,out,owner,deadline)
  s.require(live.measure(folder)<=p["budget"]["per_case_bytes"],"per_case_artifacts")
  return data
 identity(p);checkpoint()
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 app=application(WorkerContext.capture(folder/"tasks"))
 row=dict(case=name,status="failed",origin="deterministic_rule"if mode=="offline"else"current_real_agent",
  compiled=False,encrypted_execution=False,execution_source_kind=c["execution_source_kind"],
   source_intake_failure=c["source_intake_failure"],plan_binding=p["binding"],request_id=c["request_id"])
 try:
  task=prepare(app,c,mode,max(1,int(deadline-time.time())));row["task_id"]=task
  s.write(folder/"task-id.json",dict(task_id=task))
  q=app._checked_task_metadata(task)["request"]
  if mode=="offline":
   from unified_graph_lowering import candidate_source
   try:source,strategy=candidate_source(q)
   except (ValueError,TypeError,KeyError,NotImplementedError)as error:
    row.update(failure_layer="baseline",error_type=type(error).__name__,diagnostic=str(error)[:240])
    s.write(folder/"case-report.json",row);return 1
   s.write(folder/"rule-strategy.json",strategy)
   (folder/"rule.py").write_text(source)
   class Rule:
    kind="scripted_replay";agent_calls=0
    def generate(self,request,feedback=None):
     checkpoint();s.require(request==q and feedback==[],"unchanged_single_rule")
     return json.dumps(dict(schema=1,request_id=q["request_id"],hecate_source=source))
   provider=Rule()
  else:
   from agent_credentials import load_api_key
   from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport
   key=load_api_key(ROOT,provider="deepseek")
   config=Config(model="deepseek-flash",reasoning_effort="high",max_tokens=384000,max_calls=4,
    timeout_seconds=1200,service_provider="deepseek",stream=True,provider_retries=3,loopback_proxy_port=0)
   class Bound(HTTPSTransport):
    def post(self,body,timeout):checkpoint();return super().post(body,timeout)
   transport=Bound(api_key=key,approved_request_id=c["request_id"],approved_config=config,enabled=True);key=""
   provider=DeepSeekProvider(config,transport=transport)
  checkpoint();result=app.run(task,provider);s.write(folder/"qualification.json",result)
  row.update(result=result,failure_layer=(result.get("failure")or{}).get("layer"))
  validations=sorted(app._dir(task).glob("validation-*.json"))
  row["compiler_calls_confirmed"]=sum(s.read(v).get("observed_stages",{}).get("compiled")is True for v in validations)
  row["encrypted_calls_confirmed"]=sum(s.read(v).get("encrypted_execution")is True for v in validations)
  if (app._dir(task)/"provider-transport.json").exists():
   from application_transport_receipt import snapshot
   receipt=app._dir(task)/"provider-transport.json"
   s.require(s.read(receipt)==snapshot(s.read(receipt)),"provider_receipt")
   row.update(transport_receipt=str(receipt),transport_receipt_sha256=s.sha(receipt))
  if result["status"]!="succeeded":s.write(folder/"case-report.json",row);return 1
  from candidate_bundle import check_bundle
  bundle=check_bundle(app._dir(task)/"program");accept=result["acceptance"]
  s.require(all(accept.get(k)is True for k in("compiled","artifact_checked","encrypted_execution","numerically_validated"))
    and accept["input_groups"]==4 and not result["package"]["reused"],"fresh_numerical_acceptance")
  s.require(bundle["agent_generated"]is(mode=="live")and result["generation"]["agent_generated"]is(mode=="live")
    and ((result["http_attempts"]>0)if mode=="live"else(result["http_attempts"]==0)),"candidate_origin")
  row.update(compiled=True,encrypted_execution=True,package=str(app._dir(task)/"program"))
  row.update(package_checks(app,task,c,folder,checkpoint))
  row["encrypted_calls_confirmed"]+=2 if c["id"]in s.REUSE_CASES else 0
  row["status"]="passed";s.write(folder/"case-report.json",row);return 0
 except Exception as e:
  row.update(error_type=type(e).__name__,diagnostic=str(e)[:240]);row["failure_layer"]=row.get("failure_layer")or"environment"
  (folder/"failure-private.txt").write_text(traceback.format_exc())
  s.write(folder/"case-report.json",row);return 1
def inside(mode):
 p=s.verify(mode);out=Path(p["campaign"]);owner=s.read(out/"owner.json")
 s.require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"pinned_worker")
 identity(p);rows=[]
 def report(status):
  data=s.calls(s.states(out),p["budget"]["max_generations"],p["budget"]["max_http"])
  v=dict(format="poseidon-generic-capacity-qualification-report-v1",mode=mode,status=status,plan_binding=p["binding"],
   sdk=p["sdk"],cases=rows,planned=len(p["cases"]),passed=sum(r["status"]=="passed"for r in rows),
   failed=sum(r["status"]!="passed"for r in rows),not_run=len(p["cases"])-len(rows),skipped=0,**data,
   compiler_calls_confirmed=sum(r.get("compiler_calls_confirmed",0)for r in rows),
   encrypted_calls_confirmed=sum(r.get("encrypted_calls_confirmed",0)for r in rows),
   bytes=live.measure(out),global_result_bytes=live.measure(s.RESULTS),seconds=time.time()-owner["started"])
  s.require(v["compiler_calls_confirmed"]<=p["budget"]["max_compiler_calls"]and
    v["encrypted_calls_confirmed"]<=p["budget"]["max_encrypted_calls"],"backend_call_budget")
  return save(out,v)
 report("running")
 try:
  for c in p["cases"]:
   s.require(s.boundary(p,out,owner)["calls_resolved"],"unresolved_previous_http")
   folder=out/c["id"];folder.mkdir(mode=0o700)
   deadline=min(owner["deadline"],time.time()+p["budget"]["case_seconds"])
   s.write(folder/"case-boundary.json",dict(deadline=deadline,plan_binding=p["binding"]))
   allow=("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG")
   env={k:os.environ[k]for k in allow if k in os.environ}
   env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(BASE),
     OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HOME="/nonexistent")
   tree=live.OwnedTree().start();child=None
   try:
    with (folder/"worker.log").open("xb")as log:
     child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--case",mode,c["id"]],
      env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     while child.poll()is None:
      s.boundary(p,out,owner,deadline);s.verify(mode)
      s.require(live.measure(folder)<=p["budget"]["per_case_bytes"],"case_artifact_budget")
      time.sleep(2)
   finally:
    background=tree.finish(exclude_reap=()if child is None else(child.pid,))
    if child is not None:child.wait(timeout=5)
   s.require(not background,"owned_background")
   row=s.read(folder/"case-report.json")
   s.require(row["case"]==c["id"]and row["plan_binding"]==p["binding"],"case_receipt")
   rows.append(row);data=report("running")
   print(json.dumps(dict(case=c["id"],status=row["status"],generations=data["generations"],
     http_attempts=data["http_attempts"])),flush=True)
   s.require(data["calls_resolved"],"unresolved_http")
   s.require(row.get("failure_layer")not in("environment","integrity","resources"),"campaign_environment_failure")
   s.require(not(row.get("result",{}).get("status")=="service_error"and
     (row.get("result",{}).get("failure")or{}).get("layer")!="provider"),"campaign_service_failure")
  data=report("complete");return 0 if data["passed"]==len(p["cases"]) else 1
 except BaseException:
  report("stopped");raise
def outer(mode):
 p=s.verify(mode);live.no_active();out=Path(p["campaign"])
 s.require(not out.exists(),"single_use_campaign")
 s.require(live.measure(s.RESULTS)+p["budget"]["new_artifact_bytes"]<=p["budget"]["global_result_bytes"]
   and __import__("shutil").disk_usage(s.RESULTS).free>=p["budget"]["reserve_bytes"]+p["budget"]["new_artifact_bytes"],
   "initial_storage_budget")
 with (HERE/"execution.lock").open("a+")as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out.mkdir(mode=0o700);now=time.time()
  s.write(out/"owner.json",dict(pid=os.getpid(),start=str(live.process_info(os.getpid())["start"]),
    started=now,deadline=now+p["budget"]["wall_seconds"],plan_binding=p["binding"]))
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
    [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",mode])
  tree=live.OwnedTree().start()
  try:code=enter_nix(command,seconds=p["budget"]["wall_seconds"],environment="poseidon-cpu")
  finally:background=tree.finish()
  s.require(not background,"nix_background")
  s.verify(mode)
  s.write(out/"terminal.json",dict(exit_code=code,finished=time.time(),plan_binding=p["binding"]))
  return code
if __name__=="__main__":
 args=sys.argv[1:]
 if len(args)==2 and args[0]=="--execute":raise SystemExit(outer(args[1]))
 if len(args)==2 and args[0]=="--inside":raise SystemExit(inside(args[1]))
 if len(args)==3 and args[0]=="--case":raise SystemExit(one(args[1],args[2]))
 raise SystemExit("qualification.py --execute offline|live")
