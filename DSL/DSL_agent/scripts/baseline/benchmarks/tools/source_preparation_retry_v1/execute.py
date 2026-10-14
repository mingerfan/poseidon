"""Finite serial live Agent qualification over explicitly frozen source requests."""
from pathlib import Path
import sys,shlex,json,time,shutil,os,fcntl
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=14400,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport
from agent_credentials import load_api_key
from numerical_results import compare
from task_numerical_assessment import mathematical
from candidate_bundle import check_bundle
plan=q.read(HERE/"live-plan.json");q.bound(plan)
launch=q.read(HERE/"launch.json");q.bound(launch)
q.require(launch["plan_binding"]==plan["binding"]and
 launch["runner_sha256"]==q.sha(Path(__file__))and
 launch["authorization"]=="human_current_plan_all_tests_approved","bound_runner_and_authorization")
OUT=Path(plan["campaign"]);BUDGET=plan["budget"]
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==plan["sdk"],"sdk")
for name,want in plan["source_files"].items():q.require(q.sha(Path(name))==want,"frozen_files")
lock=(OUT/"execution.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
q.require(not(OUT/"execution-start.json").exists(),"no_automatic_restart")
start=time.monotonic();deadline=start+BUDGET["wall_seconds"]
baseline=live.measure(q.RESULTS)
q.write(OUT/"execution-start.json",dict(plan_binding=plan["binding"],launch_binding=launch["binding"],
 owner=live.process_identity(os.getpid()),start_monotonic=start,deadline_monotonic=deadline,
 original_result_bytes=baseline))
used=dict(generations=0,http_attempts=0,qualification_reservations=0,execution_reservations=0)
rows=[];case_deadline=deadline
def save_journal():
 value=dict(format="poseidon-live-global-call-ledger-v1",plan_binding=plan["binding"],used=dict(used))
 value["binding"]=q.digest(value);path=OUT/"global-call-ledger.json"
 tmp=path.with_name(path.name+".partial");tmp.write_bytes(q.canonical(value));tmp.replace(path)
def check():
 q.require(time.monotonic()<deadline and time.monotonic()<case_deadline,"hard_wall_budget")
 total=live.measure(q.RESULTS)
 q.require(total<=BUDGET["global_result_bytes"]and total-baseline<=BUDGET["new_artifact_bytes"],"global_artifact_budget")
 q.require(shutil.disk_usage(q.RESULTS).free>=BUDGET["reserve_bytes"],"disk_reserve")
 q.require(PoseidonBackend().identity_manifest()==sdk,"sdk_changed")
 for name,want in plan["source_files"].items():q.require(q.sha(Path(name))==want,"frozen_files_changed")
def reserve(kind,limit):
 check();q.require(used[kind]<limit,"global_call_budget")
 used[kind]+=1;save_journal()
class Backend(PoseidonBackend):
 def qualify(self,request,candidate,level,*,artifact_observer=None):
  reserve("qualification_reservations",BUDGET["max_compiler_calls"])
  def observed(path):
   check()
   if artifact_observer is not None:artifact_observer(path)
  result=super().qualify(request,candidate,level,artifact_observer=observed)
  check();return result
 def execute(self,folder,samples,*,artifact_observer=None,sample_format="named"):
  reserve("execution_reservations",2)
  result=super().execute(folder,samples,artifact_observer=artifact_observer,sample_format=sample_format)
  check();return result
class Provider(DeepSeekProvider):
 def generate(self,request,feedback_history):
  reserve("generations",BUDGET["max_generations"])
  return super().generate(request,feedback_history)
class Transport(HTTPSTransport):
 def post(self,body,timeout):
  reserve("http_attempts",BUDGET["max_http_attempts"])
  remaining=min(deadline,case_deadline)-time.monotonic()
  q.require(remaining>0,"http_deadline")
  return super().post(body,min(timeout,remaining))
save_journal()
for case in plan["cases"]:
 check();name=case["id"];folder=OUT/name;task=case["task_id"]
 q.require(not(folder/"live-result.json").exists(),"previous_result_not_retried")
 case_deadline=min(deadline,time.monotonic()+BUDGET["case_seconds"])
 app=application(WorkerContext.capture(folder/"tasks"));app.backend=Backend()
 metadata=app._checked_task_metadata(task)
 q.require(metadata["request"]==q.read(case["request_file"])and metadata["binding"]==case["task_metadata_binding"],"same_frozen_request")
 q.write(folder/"case-execution-start.json",dict(plan_binding=plan["binding"],deadline_monotonic=case_deadline,
   request_id=case["request_id"],used_before=dict(used)))
 config=Config(model="deepseek-flash",reasoning_effort="high",max_tokens=384000,max_calls=4,
   timeout_seconds=1200,service_provider="deepseek",stream=True,provider_retries=3,loopback_proxy_port=0)
 key=load_api_key(ROOT,provider="deepseek")
 transport=Transport(api_key=key,approved_request_id=case["request_id"],approved_config=config,enabled=True)
 key=""
 provider=Provider(config,transport=transport)
 result=app.run(task,provider);q.write(folder/"live-result.json",result)
 row=dict(case=name,status=result["status"],task_id=task,request_id=case["request_id"],
  generations=result.get("generations",0),http_attempts=result.get("http_attempts",0),
  failure=result.get("failure"),first_attempt_passed=False,reuse_verified=False,agent_generated=True)
 if result["status"]=="succeeded":
  q.require(result["acceptance"]["level"]=="numerical","original_numerical_grade")
  bundle=check_bundle(app._dir(task)/"program")
  row.update(package_binding=bundle["binding"],maximum_absolute_error=result["acceptance"]["maximum_absolute_error"],
      compared_values=result["acceptance"]["compared_values"],input_groups=4,
      first_attempt_passed=result.get("generations",0)==1)
  model=metadata["request"]["model"]
  rng=np.random.default_rng(2026101646)
  samples=[{i["name"]:rng.uniform(-.2,.2,i["shape"]).tolist()for i in model["inputs"]}for _ in range(3)]
  q.write(folder/"reuse-inputs.json",samples)
  expected=[mathematical(model,{n:np.asarray(v,dtype=np.float64)for n,v in sample.items()})for sample in samples]
  for label,values,reference in(("new",samples,expected),("repeat",samples[:1],expected[:1])):
   check();output=app.execute(task,values)
   q.require(output["status"]=="executed"and output["encrypted_execution"]is True and
     output["executor_secret_key"]is False,"real_package_execution")
   left=[];right=[]
   for got,want in zip(output["outputs"],reference):
    for item in model["outputs"]:
     n=item["name"];a=np.asarray(got[n],dtype=np.float64).reshape(-1);b=np.asarray(want[n],dtype=np.float64).reshape(-1)
     q.require(a.shape==b.shape,"output_shape");left.extend(a.tolist());right.extend(b.tolist())
   comparison=compare(np.asarray(left).reshape(1,-1),np.asarray(right).reshape(1,-1),1e-5,1e-4)
   q.require(comparison["passed"],"new_repeat_numeric")
   q.write(folder/(label+"-execution.json"),dict(execution=output,comparison=comparison))
   row[label+"_maximum_absolute_error"]=comparison["max_absolute_error"]
  q.require(check_bundle(app._dir(task)/"program")==bundle,"unchanged_exported_package")
  row["reuse_verified"]=True
 q.write(folder/"case-report.json",row);rows.append(row)
 print(json.dumps({k:v for k,v in row.items()if k not in("failure",)}),flush=True)
 check()
report=dict(format="poseidon-source-preparation-live-acceptance-v1",plan_binding=plan["binding"],launch_binding=launch["binding"],
 cases=rows,planned=len(plan["cases"]),passed=sum(r["status"]=="succeeded"for r in rows),
 failed=sum(r["status"]!="succeeded"for r in rows),not_run=len(plan["cases"])-len(rows),skipped=0,
 first_attempt_passed=sum(r["first_attempt_passed"]for r in rows),
 used=dict(used),sdk=plan["sdk"],seconds=time.monotonic()-start,bytes=live.measure(OUT),
 original_application_accuracy_certified=False,network_changed=False,manual_candidate_substitution=False)
report["binding"]=q.digest(report);q.write(OUT/"report.json",report);q.write(HERE/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
