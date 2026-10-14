"""Reuse a qualified rule program through actual CLI assessment commands; API/compile0."""
import hashlib,io,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
from hecate_python_env import WORK,VENV,enter_nix
RESULTS=WORK/"results";OUT=RESULTS/"agent-assessment-total-deadline-acceptance-v1"
OLD=RESULTS/"agent-construction-guidance-acceptance-v1"
SCOPE=dict(wall_seconds=300,new_bytes=128*1024**2,global_bytes=32*1024**3,reserve_bytes=4*1024**3,
 native_cpu_seconds=150,native_address_bytes=4*1024**3,native_workers=1,compile_jobs=2,link_jobs=1,provider_calls=0)
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for chunk in iter(lambda:f.read(1024**2),b""):h.update(chunk)
 return h.hexdigest()
def write(p,v):
 with Path(p).open("xb")as f:f.write(canonical(v))
def measure(p):return int(subprocess.check_output(["du","--apparent-size","-s","-B1",str(p)],text=True,timeout=30).split()[0])
def boundary(plan):
 require(time.time()<plan["deadline_wall"],"assessment_offline_deadline")
 require(shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=SCOPE["global_bytes"],"global_artifact_budget")
 added=[p for p in RESULTS.iterdir()if p.name not in plan["existing_results"]]
 require(sum(measure(p)for p in added)<=SCOPE["new_bytes"],"new_artifact_budget")
 for n,h in plan["sources"].items():require(sha(ROOT/n)==h,"frozen_agent_source_changed")
 for n,h in plan["history"].items():require(sha(n)==h,"old_program_or_evidence_changed")
def outer():
 require(not OUT.exists(),"fresh_result_directory")
 controls=read(HERE/"controls-affected.json");require(controls["status"]=="passed" and controls["tests_run"]==29,"controls_required")
 old=read(OLD/"report.json");require(old["status"]=="passed","qualified_program_required")
 row=next(r for r in old["rows"]if r["case"]=="group_mean_center")
 protected=read(HERE/"baseline.json")["protected_sources"]
 require(all(sha(Path(n)if Path(n).is_absolute()else ROOT/n)==h for n,h in protected.items()),"protected_sources_changed")
 require(measure(RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"] and shutil.disk_usage(WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"space_precheck")
 package=Path(row["package"]);manifest=read(package/"manifest.json")
 history={str(package/n):sha(package/n)for n in manifest["files"]}
 history.update({str(OLD/"report.json"):sha(OLD/"report.json"),str(package/"manifest.json"):sha(package/"manifest.json"),
  str(HERE/"controls-affected.json"):sha(HERE/"controls-affected.json")})
 files=("application_assessment.py","assessment_time_budget.py","application_assessment_entry/cli.py",
  "application_assessment_entry/supervision.py","task_numerical_assessment.py","task_dataset_evaluation.py")
 plan=dict(format="poseidon-assessment-total-budget-offline-plan-v1",scope=SCOPE,
  sources={"scripts/baseline/"+n:sha(BASE/n)for n in files},history=history,
  controls_binding=controls["binding"],identity_sha256=controls["identity"]["manifest_sha256"],
  existing_results=sorted(p.name for p in RESULTS.iterdir()),deadline_wall=time.time()+SCOPE["wall_seconds"],
  program_task=row["task_id"],program_package=str(package),program_binding=row["package_binding"],
  original_tasks=str(OLD/"tasks"),candidate_origin=row["candidate_origin"],protected_sources_checked=len(protected))
 plan["binding"]=hashlib.sha256(canonical(plan)).hexdigest()
 OUT.mkdir(mode=0o700);write(OUT/"plan.json",plan)
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 code=enter_nix(cmd,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
 if code not in (0,1):write(OUT/"termination.json",dict(status="timed_out_or_worker_exit",code=code,automatic_retry=False,provider_calls=0))
 return code
def inside():
 require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"pinned_worker")
 import numpy as np
 from application_identity import current_identity_manifest
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_assessment_entry import cli
 from assessment_array_dataset import export_dataset
 from benchmark_graph import samples
 plan=read(OUT/"plan.json");boundary(plan);identity=current_identity_manifest()
 require(identity["manifest_sha256"]==plan["identity_sha256"],"SDK_identity_changed")
 app=application(WorkerContext.capture(Path(plan["original_tasks"])))
 before=app.result(plan["program_task"])
 request=app._checked_task_metadata(plan["program_task"])["request"]
 model=request["model"];probes=list(samples(model,4))[1:3]
 dataset=export_dataset(request,{n:np.stack([p[n]for p in probes])for n in probes[0]},OUT/"dataset")
 args=["--tasks",plan["original_tasks"],"--store",str(OUT/"assessment-store"),"--timeout","120"]
 commands=[];started=time.monotonic()
 def invoke(parts):
  boundary(plan);stream=io.StringIO();code=cli.main(args+parts,stdout=stream);response=json.loads(stream.getvalue())
  evidence=OUT/f"command-{len(commands)}.json";write(evidence,response)
  commands.append(dict(action=parts[0],exit_code=code,status=response["status"],
   cleanup=response.get("process_cleanup"),response_binding=response["binding"],evidence=str(evidence)))
  require(response.get("process_cleanup",{}).get("confirmed")is True,"worker_cleanup_unconfirmed")
  return code,response
 code,result=invoke(["submit","--task",plan["program_task"],"--dataset",str(OUT/"dataset/dataset.json"),
  "--dataset-binding",dataset["binding"],"--max-new-assessments","2","--part-samples","1","--total-wall-seconds","240"])
 require(code==0 and result["status"]=="submitted","submission_rejected")
 handle=result["handle"]
 tail=["--evaluation-id",handle["evaluation_id"],"--submission-binding",handle["binding"]]
 path=OUT/"assessment-store"/handle["evaluation_id"]
 original_plan=sha(path/"plan.json")
 code,first=invoke(["advance",*tail]);require(code==0 and first["status"]=="ready","first_assessment_rejected")
 require(first["result"]["budget"]["used"]==1,"first_call_admission")
 code,status=invoke(["status",*tail]);require(code==0 and status["result"]["budget"]["used"]==1,"status_changed_budget")
 code,second=invoke(["advance",*tail]);require(code==0 and second["status"]=="succeeded","second_assessment_rejected")
 require(second["result"]["budget"]["used"]==2 and second["result"]["completed_parts"]==2,"continuation_scope")
 require(sha(path/"plan.json")==original_plan,"time_budget_was_reset")
 code,reloaded=invoke(["result",*tail]);require(code==0 and reloaded["result"]["binding"]==second["result"]["binding"],"terminal_result_changed")
 code,cancel=invoke(["cancel",*tail]);require(code==0 and cancel["result"]["binding"]==second["result"]["binding"],"completed_cancel_changed")
 code,repeat=invoke(["advance",*tail]);require(code==0 and repeat["result"]["binding"]==second["result"]["binding"],"completed_advance_reexecuted")
 summary=second["result"]["summary"];require(summary["complete"]and summary["supplied_inputs_numerically_validated"],"numerical_scope_failed")
 receipts={p.stem:read(p)for p in (path/"receipts").glob("part_*.json")}
 require(len(receipts)==2 and all(r["encrypted_execution"]is True for r in receipts.values()),"real_execution_receipts")
 # A fresh optional assessment may adopt independently verified numeric receipts.
 from application_assessment import ApplicationAssessment
 c=ApplicationAssessment(app,OUT/"cache-assessment-store")
 cached=c.submit(plan["program_task"],OUT/"dataset/dataset.json",expected_dataset_binding=dataset["binding"],
  part_samples=1,cached_receipts={n:dict(receipt=r,binding=r["binding"])for n,r in receipts.items()},
  total_wall_seconds=240)
 cached_result=c.advance(cached,max_new_parts=2)
 require(cached_result["status"]=="succeeded"and cached_result["budget"]["used"]==0,"cached_receipt_reexecuted")
 write(OUT/"cache-result.json",cached_result)
 require(app.result(plan["program_task"])==before,"original_program_task_changed")
 boundary(plan);require(current_identity_manifest()==identity,"SDK_source_changed")
 protected=read(HERE/"baseline.json")["protected_sources"]
 require(all(sha(Path(n)if Path(n).is_absolute()else ROOT/n)==h for n,h in protected.items()),"protected_sources_changed")
 comparison=summary["ckks_comparison"]
 report=dict(format="poseidon-assessment-total-budget-backend-report-v1",status="passed",plan_binding=plan["binding"],
  controls_binding=plan["controls_binding"],SDK_identity_sha256=identity["manifest_sha256"],
  candidate_origin=plan["candidate_origin"],program_task=plan["program_task"],program_binding=plan["program_binding"],
  commands=commands,total_wall_seconds=240,time_budget_plan_unchanged=True,
  compiler_calls=0,provider_calls=0,http_attempts=0,encrypted_calls=2,input_groups=2,
  compared_values=comparison["compared_values"],maximum_absolute_error=comparison["max_absolute_error"],
  numeric_tolerance=dict(atol=1e-5,rtol=1e-4),package_reload_and_new_inputs=True,
  repeated_result_cancel_advance_added_calls=0,cache_added_calls=0,
  original_task_and_program_unchanged=True,protected_sources_checked=len(protected),protected_sources_changed=0,
  boundary_cancellation="control/real-owned-process tests only; no forced FHE interruption claim",
  direct_API_deadline="boundary checks; CLI advance owns hard process stop",
  seconds=time.monotonic()-started,artifact_bytes=measure(OUT))
 report["binding"]=hashlib.sha256(canonical(report)).hexdigest();write(OUT/"report.json",report)
 print(json.dumps({k:v for k,v in report.items()if k!="commands"}),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
