"""Execute only the freshly rebound, already approved pending groups."""
import copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from semantic_benchmark_execution import runtime_sources
from application_backend import PoseidonBackend
from application_component import Application
from component_contract import prepare_task
from hecate_python_env import VENV
work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
pre=work/"agent-pending-live-preparation-r1084";state=work/"agent-approved-pending-execution-r1088"
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
approval=load(pre/"approval.json")
require(digest({k:v for k,v in approval.items() if k!="binding"})==approval["binding"],"Current approval identity")
require(approval["approval_text"]=="批准当前计划内积累的所有待批准执行","Actual human approval")
require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned execution")
started=time.monotonic();groups=[]
def boundary():
 require(runtime_sources()==approval["source_hashes"],"Frozen production sources")
 require(all(sha(p)==h for p,h in approval["tool_hashes"].items()),"Frozen execution tools")
 require(all(sha(p["path"])==p["sha256"] for p in approval["groups"].values() for p in [p["parent"]]),"Original approval documents unchanged")
 import shutil
 require(shutil.disk_usage(work).free>=4*1024**3,"Disk reserve")
 require(int(subprocess.check_output(["du","-sb",str(work)],text=True,timeout=30).split()[0])<=32*1024**3,"Global artifact limit")
def save(status):
 (state/"report.json").write_bytes(canonical(dict(format="poseidon-approved-pending-execution-v1",
  status=status,approval_binding=approval["binding"],groups=groups,seconds=time.monotonic()-started,
  planned=76,automatic_restart=False)))
def run_child(argv,log,seconds):
 with log.open("x") as f:
  child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:return child.wait(timeout=max(1,int(seconds)))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   return 124
def reuse(plan,output,deadline):
 import numpy as np
 from benchmark_math import evaluate as math_ref
 from benchmark_torch import evaluate as torch_ref
 from numerical_results import compare
 from candidate_bundle import check_execution_dependencies
 report=load(output/"report.json");app=Application(output/"tasks",PoseidonBackend())
 results=[]
 for row in report["rows"]:
  if row["status"]!="passed":continue
  require(time.time()<deadline,"Shared approved group deadline");boundary()
  item=next(c for c in plan["cases"] if c["id"]==row["id"])
  folder=Path(item["imported_directory"]);model=load(folder/"model.json")
  require(digest(model)==item["model_sha256"],"Immutable mathematical model")
  _,request=check_execution_dependencies(row["package"]);require(request["request_id"]==item["request_id"],"Package request")
  inputs=[];reference=[]
  for i in range(3):
   rng=np.random.default_rng(20261007+i)
   values={s["name"]:rng.uniform(-.2,.2,s["shape"]) for s in model["inputs"]}
   a,b=math_ref(model,values),torch_ref(model,values)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
   inputs.append({k:v.tolist() for k,v in values.items()})
   reference.append(np.concatenate([a[o["name"]].reshape(-1) for o in model["outputs"]]))
  execution=app.execute(row["task_id"],inputs)
  require(execution["status"]=="executed" and execution["encrypted_execution"] and not execution["executor_secret_key"],
     "Fresh input ciphertext execution")
  require(execution["agent_calls"]==execution["compiler_calls"]==0,"Program reuse without regeneration")
  actual=np.asarray([np.concatenate([np.asarray(s[o["name"]]).reshape(-1) for o in model["outputs"]]) for s in execution["outputs"]])
  comparison=compare(actual,np.asarray(reference),1e-5,1e-4)
  results.append(dict(id=row["id"],task_id=row["task_id"],status="passed" if comparison["passed"] else "failed",
    execution=execution,comparison=comparison,agent_calls=0,compiler_calls=0))
  (output/"new-input-acceptance.json").write_bytes(canonical(dict(format="poseidon-live-package-new-inputs-v1",rows=results,
     passed=sum(r["status"]=="passed" for r in results),failed=sum(r["status"]=="failed" for r in results),inputs_per_model=3)))
 return results
def attention(plan,output,deadline):
 from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport
 from agent_credentials import load_api_key
 from candidate_bundle import check_bundle,replay_bundle
 boundary();require(not output.exists(),"Fresh Attention result")
 require(digest({k:v for k,v in plan.items() if k!="binding"})==plan["binding"],"Attention plan identity")
 require(PoseidonBackend().identity()==plan["backend_identity"],"Attention execution identity")
 item=plan["cases"][0];model=load(item["model_file"])
 require(sha(item["model_file"])==item["model_file_sha256"] and digest(model)==item["model_sha256"],"Attention model")
 request=prepare_task(model,item["options"])
 from deepseek_provider import public_request
 require(request["request_id"]==item["request_id"] and digest(public_request(request))==item["public_context_sha256"],"Attention public request")
 output.mkdir();(output/"plan.json").write_bytes(canonical(plan))
 provider_config=plan["provider"]
 config=Config(model=provider_config["model"],service_provider=provider_config["provider"],reasoning_effort="high",
   max_tokens=384000,max_calls=4,timeout_seconds=1200,provider_retries=3,loopback_proxy_port=0,stream=True)
 app=Application(output/"tasks",PoseidonBackend())
 limits=dict(generations=4,http_attempts=16,wall_seconds=min(14400,max(1,int(deadline-time.time()))),
    api_timeout_seconds=1200,max_tokens=384000)
 task=app.submit(model,options=item["options"],limits=limits)
 key=load_api_key(ROOT,provider="deepseek");os.environ.pop("DEEPSEEK_API_KEY",None)
 try:
  transport=HTTPSTransport(api_key=key,approved_request_id=item["request_id"],approved_config=config,enabled=True)
  provider=DeepSeekProvider(config,transport=transport);result=app.run(task,provider)
 finally:key=""
 row=dict(id=item["id"],task_id=task,result=result,status="failed")
 if result["status"]=="succeeded":
  package=output/"tasks"/task/"program";manifest=check_bundle(package)
  require(manifest["agent_generated"] and result["http_attempts"]>0 and not result["package"]["reused"],"Real Agent source")
  require(replay_bundle(package)==0,"Real package reload")
  row.update(status="passed",package=str(package),package_binding=manifest["binding"],reloaded=True)
 (output/"report.json").write_bytes(canonical(dict(format="poseidon-approved-attention-report-v1",status="complete",
     planned=1,rows=[row],passed=int(row["status"]=="passed"),failed=int(row["status"]!="passed"),
     generations=result["generations"],api_attempts=result["http_attempts"],cost_usd=None,binding=plan["binding"])))
 return result["status"] not in ("service_error","cancelled","timed_out")
if len(sys.argv)>1 and sys.argv[1] in ("--attention","--reuse"):
 mode,plan_path,output,deadline=sys.argv[1:]
 if mode=="--attention":raise SystemExit(0 if attention(load(plan_path),Path(output),float(deadline)) else 1)
 reuse(load(plan_path),Path(output),float(deadline));raise SystemExit(0)
require(not state.exists(),"Never repeat an uncertain live execution")
state.mkdir(mode=0o700)
save("running")
try:
 for group in ("functional15","matrix_instance60","attention1"):
  boundary();spec=approval["groups"][group];deadline=time.time()+spec["shared_batch_seconds"]
  current=dict(id=group,status="running",plans=spec["plans"],outputs=[],deadline_wall=deadline,
    max_generations=spec["max_generations"],max_http_attempts=spec["max_http_attempts"])
  groups.append(current);save("running")
  if not spec["plans"]:current["status"]="input_blocked";save("running");continue
  for number,entry in enumerate(spec["plans"],1):
   boundary();path=Path(entry["path"]);require(sha(path)==entry["sha256"],"Frozen live plan")
   plan=load(path);require(plan["binding"]==entry["binding"] and plan["source_hashes"]==approval["source_hashes"],"Current exact live binding")
   output=work/("agent-approved-"+group+"-live-r1088-"+str(number))
   current["outputs"].append(str(output));save("running")
   require(time.time()<deadline,"Approved shared group time bound")
   if group=="attention1":
    # A separate bounded child owns key loading and the existing application.
    code=run_child([sys.executable,"-B",str(Path(__file__).resolve()),"--attention",str(path),str(output),str(deadline)],
       state/(group+"-"+str(number)+".log"),deadline-time.time())
   else:
    tool=ROOT/"scripts/baseline/benchmarks/tools/run_application_matrix.py"
    code=run_child([sys.executable,"-B",str(tool),"--inside","--execute",str(path),"--output",str(output),
       "--approved-binding",entry["binding"]],state/(group+"-"+str(number)+".log"),deadline-time.time())
   current["last_exit_code"]=code;save("running")
   require((output/"report.json").is_file(),"Execution report missing")
   report=load(output/"report.json")
   require(report["generations"]<=plan["max_generations"] and report["api_attempts"]<=plan["max_http_attempts"],"Original call ceilings")
   require(report["status"]=="complete","Related group stopped; never repeat uncertain calls")
   if group!="attention1":
    code=run_child([sys.executable,"-B",str(Path(__file__).resolve()),"--reuse",str(path),str(output),str(deadline)],
       state/(group+"-"+str(number)+"-reuse.log"),deadline-time.time())
    require(code==0,"Program reuse acceptance stopped")
   save("running")
  generated=sum(load(Path(p)/"report.json")["generations"] for p in current["outputs"])
  http=sum(load(Path(p)/"report.json")["api_attempts"] for p in current["outputs"])
  require(generated<=current["max_generations"] and http<=current["max_http_attempts"],"Shared original group call budget")
  current.update(status="complete",generations=generated,http_attempts=http);save("running")
 save("complete")
except BaseException:
 save("stopped");raise
