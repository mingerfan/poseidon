"""Bounded CLI/package compatibility qualification; uses existing Application APIs."""
import fcntl,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),
 str(BASE/"benchmarks/tools/public_join_intake_v1"),str(BASE)]
import run as live
import qualification_scope as common
from hecate_python_env import VENV,enter_nix
CASES=("nested_cat","nested_entry_stack","nested_linear_residual")
RESULTS=common.RESULTS
CAMPAIGN=RESULTS/"agent-source-cli-offline-r1482"
OPTIONS=dict(generation_guidance="explicit-v14",public_weight_policy="descriptors-math-roles-v2",
 chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
def require(ok,code):
 if not ok:raise ValueError(code)
def write(path,value):
 path=Path(path);require(not path.exists(),"evidence_exists")
 with path.open("xb")as f:os.chmod(path,0o600);f.write(common.canonical(value))
def read(path):return common.read(path)
def digest(value):return common.digest(value)
def plan():
 p=read(HERE/"offline-plan.json");common.bound(p)
 require(p["authorization"]=="human_explicit_current_plan_test_approval","authorization")
 require(p["campaign"]==str(CAMPAIGN)and tuple(c["id"]for c in p["cases"])==CASES,"case_scope")
 require(p["budget"]==dict(wall_seconds=720,case_seconds=180,new_bytes=512*1024**2,
  global_bytes=48*1024**3,reserve_bytes=4*1024**3,compiler_calls=3,encrypted_calls=9,
  api_calls=0,compiler_jobs=2,link_jobs=1,encrypted_concurrency=1),"budget_scope")
 require(p["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"tolerance")
 a=read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");common.bound(a)
 require(a["approved"]is True and a["binding"]==p["storage_activation_binding"],"storage_approval")
 for path,want in p["files"].items():require(common.sha(common.regular(path))==want,"frozen_file:"+path)
 return p
def identity(p):
 from application_backend import PoseidonBackend
 require(PoseidonBackend().identity_manifest()==read(HERE/"identity-snapshot.json"),"sdk_identity")
def boundary(p,owner,deadline=None):
 require(time.time()<min(owner["deadline"],deadline or owner["deadline"]),"wall_budget")
 info=live.process_info(owner["pid"])
 require(info is not None and str(info["start"])==owner["start"]and info["state"]not in("Z","X"),"owner")
 require(live.measure(CAMPAIGN)<=p["budget"]["new_bytes"]and
  live.measure(RESULTS)<=p["budget"]["global_bytes"],"artifact_budget")
 require(__import__("shutil").disk_usage(RESULTS).free>=p["budget"]["reserve_bytes"],"disk_reserve")
 for state in common.states(CAMPAIGN):require(state["http_attempts"]==0 and state["generations"]<=1,"no_api")
def checks(app,task,case,folder,checkpoint):
 import numpy as np
 from benchmark_math import evaluate as math_reference
 from benchmark_torch import evaluate as torch_reference
 from candidate_bundle import check_bundle
 from numerical_results import compare
 from pytorch_source_contract import restore_outputs
 from program_execution import package_execution_metadata
 model=app._checked_task_metadata(task)["request"]["model"]
 parent=read(case["parent_request"])["model"]
 require(model["inputs"]==parent["inputs"],"input_signature_unchanged")
 require([o["name"]for o in model["outputs"]]==[o["name"]for o in parent["outputs"]],"output_names_unchanged")
 before=check_bundle(app._dir(task)/"program")
 structure=package_execution_metadata(app._dir(task)/"program",before,model)["output_structure"]
 outcomes={}
 for stage,count in (("new_inputs",3),("repeat",1)):
  checkpoint();samples=[];expected=[];parent_expected=[]
  for i in range(count):
   rng=np.random.default_rng(20261008+i)
   values={d["name"]:rng.uniform(-.2,.2,d["shape"]).astype(np.float64)for d in model["inputs"]}
   a,b=math_reference(model,values),torch_reference(model,values)
   old=math_reference(parent,values)
   for key in a:
    np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(a[key],old[key],atol=1e-12,rtol=1e-12)
   samples.append({k:v.tolist()for k,v in values.items()})
   expected.append(np.concatenate([a[o["name"]].reshape(-1)for o in model["outputs"]]))
  response=app.execute(task,samples)
  actual=np.asarray([np.concatenate([np.asarray(v[o["name"]],dtype=np.float64).reshape(-1)
   for o in model["outputs"]])for v in response["outputs"]])
  comparison=compare(actual,np.asarray(expected),1e-5,1e-4)
  require(response["encrypted_execution"]is True and
   response["agent_calls"]==response["compiler_calls"]==0 and
   response["executor_secret_key"]is False and comparison["passed"],"package_numerical")
  require(response.get("output_structure")==structure,"output_structure")
  if structure is not None:require(response["structured_outputs"]==
   [restore_outputs(v,structure,allow_none=True)for v in response["outputs"]],"structured_outputs")
  write(folder/(stage+".json"),dict(status="passed",execution=response,comparison=comparison))
  outcomes[stage]=dict(groups=count,values=comparison["compared_values"],
   maximum_absolute_error=comparison["max_absolute_error"])
 checkpoint()
 from application_model_source import FORMAT
 src=read(app._dir(task)/"source.json")
 again=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],
  trust_local_code=True),resource_profile="general-tensors-v1",options=src["requirements"]["options"],
  limits=src["budget"])
 class Never:
  kind="scripted_replay";agent_calls=0
  def generate(self,*args):raise AssertionError("cache_must_not_generate")
 cached=app.run(again,Never())
 require(cached["status"]=="succeeded"and cached["package"]["reused"]is True and
  cached["package"]["binding"]==before["binding"]and cached["generations"]==cached["http_attempts"]==0,
  "cache_reuse")
 require(check_bundle(app._dir(task)/"program")==before,"package_unchanged")
 write(folder/"cache.json",cached)
 return dict(**outcomes,package_reloaded=True,package_reused=True,package_binding=before["binding"])
def one(name):
 p=plan();identity(p);owner=read(CAMPAIGN/"owner.json")
 case=next(c for c in p["cases"]if c["id"]==name);folder=CAMPAIGN/name
 deadline=read(folder/"case-boundary.json")["deadline"]
 def checkpoint():plan();boundary(p,owner,deadline)
 checkpoint()
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from unified_graph_lowering import candidate_source
 app=application(WorkerContext.capture(folder/"tasks"))
 row=dict(case=name,status="failed",origin="deterministic_rule",plan_binding=p["binding"],
  source_manifest=case["source_manifest"],compiled=False,encrypted_execution=False)
 try:
  task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=case["source_manifest"],
   trust_local_code=True),resource_profile="general-tensors-v1",options=OPTIONS,
   limits=dict(generations=1,http_attempts=1,wall_seconds=max(1,int(deadline-time.time())),
    api_timeout_seconds=1200,max_tokens=384000))
  prepared=app.prepare_source(task)
  require(prepared["status"]=="queued","source_prepare:"+str(prepared.get("failure")))
  q=app._checked_task_metadata(task)["request"];row.update(task_id=task,request_id=q["request_id"])
  write(folder/"request.json",q)
  source,strategy=candidate_source(q)
  (folder/"rule.py").write_text(source);write(folder/"rule-strategy.json",strategy)
  class Rule:
   kind="scripted_replay";agent_calls=0
   def generate(self,request,feedback=None):
    checkpoint();require(request==q and feedback==[],"single_frozen_rule")
    return json.dumps(dict(schema=1,request_id=q["request_id"],hecate_source=source))
  result=app.run(task,Rule());write(folder/"qualification.json",result)
  row.update(result=result,failure_layer=(result.get("failure")or{}).get("layer"))
  validations=list(app._dir(task).glob("validation-*.json"))
  row["compiler_calls_confirmed"]=sum(read(v).get("observed_stages",{}).get("compiled")is True for v in validations)
  row["encrypted_calls_confirmed"]=sum(read(v).get("encrypted_execution")is True for v in validations)
  if result["status"]=="succeeded":
   accept=result["acceptance"]
   require(all(accept.get(k)is True for k in("compiled","artifact_checked","encrypted_execution",
    "numerically_validated"))and accept["input_groups"]==4 and result["http_attempts"]==0
    and result["generation"]["agent_generated"]is False,"real_rule_acceptance")
   row.update(compiled=True,encrypted_execution=True,package=str(app._dir(task)/"program"))
   row.update(checks(app,task,case,folder,checkpoint))
   row["encrypted_calls_confirmed"]+=2;row["status"]="passed"
 except Exception as exc:
  row.update(error_type=type(exc).__name__,diagnostic=str(exc)[:240])
  row["failure_layer"]=row.get("failure_layer")or"environment"
  (folder/"failure-private.txt").write_text(traceback.format_exc())
 write(folder/"case-report.json",row)
 return 0 if row["status"]=="passed"else 1
def inside():
 p=plan();identity(p);owner=read(CAMPAIGN/"owner.json");rows=[]
 def report(status):
  value=dict(format="poseidon-source-cli-offline-report-v1",status=status,plan_binding=p["binding"],
   planned=3,passed=sum(r["status"]=="passed"for r in rows),
   failed=sum(r["status"]!="passed"for r in rows),not_run=3-len(rows),skipped=0,
   api_calls=0,compiler_calls_confirmed=sum(r.get("compiler_calls_confirmed",0)for r in rows),
   encrypted_calls_confirmed=sum(r.get("encrypted_calls_confirmed",0)for r in rows),
   cases=rows,seconds=time.time()-owner["started"],bytes=live.measure(CAMPAIGN))
  value["binding"]=digest(value);tmp=CAMPAIGN/"report.tmp"
  tmp.write_bytes(common.canonical(value));os.chmod(tmp,0o600);tmp.replace(CAMPAIGN/"report.json")
  return value
 report("running")
 try:
  for case in p["cases"]:
   boundary(p,owner);folder=CAMPAIGN/case["id"];folder.mkdir(mode=0o700)
   write(folder/"case-boundary.json",dict(deadline=min(owner["deadline"],time.time()+180)))
   allowed=("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT",
    "HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG")
   env={k:os.environ[k]for k in allowed if k in os.environ}
   env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(BASE),
    OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HOME="/nonexistent")
   tree=live.OwnedTree().start();child=None
   try:
    with(folder/"worker.log").open("xb")as log:
     child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--case",case["id"]],
      env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     while child.poll()is None:
      boundary(p,owner,read(folder/"case-boundary.json")["deadline"]);plan();time.sleep(2)
   finally:
    background=tree.finish(exclude_reap=()if child is None else(child.pid,))
    if child is not None:child.wait(timeout=5)
   require(not background,"owned_background")
   row=read(folder/"case-report.json");rows.append(row);r=report("running")
   print(json.dumps(dict(case=case["id"],status=row["status"])),flush=True)
   require(row.get("failure_layer")not in("environment","integrity","resources"),"environment_stop")
  r=report("complete");return 0 if r["passed"]==3 else 1
 except BaseException:report("stopped");raise
def outer():
 p=plan();live.no_active();require(not CAMPAIGN.exists(),"single_use_campaign")
 require(live.measure(RESULTS)+p["budget"]["new_bytes"]<=p["budget"]["global_bytes"]and
  __import__("shutil").disk_usage(RESULTS).free>=p["budget"]["reserve_bytes"]+p["budget"]["new_bytes"],
  "initial_storage_budget")
 with(HERE/"execution.lock").open("a+")as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  CAMPAIGN.mkdir(mode=0o700);now=time.time()
  write(CAMPAIGN/"owner.json",dict(pid=os.getpid(),start=str(live.process_info(os.getpid())["start"]),
   started=now,deadline=now+720,plan_binding=p["binding"]))
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  tree=live.OwnedTree().start();code=1
  try:code=enter_nix(command,seconds=720,environment="poseidon-cpu")
  finally:
   background=tree.finish()
   write(CAMPAIGN/"terminal.json",dict(exit_code=code,finished=time.time(),plan_binding=p["binding"],
    owned_background=background))
  require(not background,"nix_background");plan();return code
if __name__=="__main__":
 if sys.argv[1:]==["--execute"]:raise SystemExit(outer())
 if sys.argv[1:]==["--inside"]:raise SystemExit(inside())
 if len(sys.argv)==3 and sys.argv[1]=="--case":raise SystemExit(one(sys.argv[2]))
 raise SystemExit("qualify.py --execute")
