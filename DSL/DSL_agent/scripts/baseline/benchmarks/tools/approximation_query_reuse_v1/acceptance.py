"""New controller binding, reuse old source receipts, execute only tail parts."""
from pathlib import Path
import sys,json,shlex,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=1200,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_approximation_assessment import ApplicationApproximationAssessment,identity
from task_approximation_assessment import implementation as reference_identity
live.no_active()
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==29,"current_controls")
sdk=PoseidonBackend().identity_manifest();q.require(sdk["manifest_sha256"]==controls["sdk"],"unchanged_sdk")
for p,want in controls["source_files"].items():q.require(q.sha(p)==want,"tested_source")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
old=q.RESULTS/"agent-approximation-dataset-acceptance-r1612"
oldreport=q.read(old/"report.json");q.bound(oldreport)
q.require(oldreport["reference_identity"]==reference_identity(),"same_receipt_producer")
def hashes(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
oldhashes=hashes(old)
OUT=q.RESULTS/"agent-approximation-query-reuse-r1616"
q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-approximation-query-reuse-scope-v1",authorization="human_explicit_current_plan_test_approval",
 cases=["matrix","negative_axes","scalar_intermediate"],sdk=sdk["manifest_sha256"],
 controller_identity=identity(),reference_identity=reference_identity(),prior_acceptance_binding=oldreport["binding"],
 controls_binding=controls["binding"],source_files={**controls["source_files"],str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 budget=dict(wall_seconds=1200,max_native_workers=3,new_artifact_bytes=128*1024**2,
  total_artifact_bytes=48*1024**3,min_free_bytes=4*1024**3,api_calls=0,compiler_calls=0,encrypted_calls=0),
 original_source_receipts_reused=True,old_records_rewritten=False,network_changed=False,
 backend_changed=False,parameters_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
started=time.monotonic();rows=[]
for name in scope["cases"]:
 q.require(time.monotonic()-started<1100 and live.measure(OUT)<=scope["budget"]["new_artifact_bytes"]and
  live.measure(q.RESULTS)<=scope["budget"]["total_artifact_bytes"]and
  shutil.disk_usage(q.RESULTS).free>=scope["budget"]["min_free_bytes"],"case_boundary")
 folder=OUT/name;folder.mkdir(mode=0o700)
 oldcase=old/name;handle=q.read(oldcase/"handle.json")
 plan=q.read(oldcase/"assessment-store"/handle["evaluation_id"]/"plan.json");q.bound(plan)
 source=q.read(oldcase/"assessment-store"/handle["evaluation_id"]/"receipts/part_0.json");q.bound(source)
 task=plan["program_task"];task_store=Path(prior["campaign"])/name/"tasks"
 frozen=hashes(task_store/task)
 app=application(WorkerContext.capture(task_store));outcome=app.result(task)
 control=ApplicationApproximationAssessment(app,folder/"assessment-store")
 h=control.submit(task,oldcase/"dataset/dataset.json",expected_dataset_binding=plan["dataset_file_binding"],
  input_dtypes=plan["input_dtypes"],trust_local_code=True,max_new_assessments=1,part_samples=16,
  application_criterion=plan["application_criterion"],total_wall_seconds=500,
  cached_receipts={"part_0":dict(receipt=source,binding=source["binding"])})
 q.write(folder/"handle.json",h)
 before=control.status(h)
 q.require(before["budget"]["used"]==0 and before["completed_parts"]==1 and before["summary"]["complete"] is False,
   "cached_first_part_has_zero_new_calls")
 result=control.advance(h,max_new_parts=1);q.write(folder/"result.json",result)
 q.require(result["status"]=="succeeded" and result["budget"]["used"]==1 and result["completed_parts"]==2 and
  result["summary"]["evaluated_samples"]==17 and result["acceptance"]["application_criterion_validated"] is True,
  "only_tail_admitted")
 job=folder/"assessment-store"/h["evaluation_id"]
 native=list((job/"native").glob("*/native-reference.json"));q.require(len(native)==1 and native[0].parent.name=="part_1","one_actual_source_worker")
 q.require(q.read(job/"receipts/part_0.json")==source,"original_receipt_byte_values_preserved")
 other=ApplicationApproximationAssessment(app,folder/"assessment-store")
 q.require(other.result(h)["binding"]==result["binding"] and other.advance(h)["binding"]==result["binding"],"reload_no_execution")
 q.require(app.result(task)==outcome and hashes(task_store/task)==frozen,"original_agent_program_unchanged")
 previous=q.read(oldcase/"result.json")
 q.require(result["summary"]["application_criterion"]==previous["summary"]["application_criterion"] and
  result["summary"]["maximum_absolute_error"]==previous["summary"]["maximum_absolute_error"],
  "same_original_and_approximation_values")
 q.require(result["encrypted_execution"] is False and result["ckks_validation_performed"] is False,
  "no_new_encrypted_claim")
 q.require(identity()==scope["controller_identity"] and reference_identity()==scope["reference_identity"]and
  PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 row=dict(case=name,status="passed",samples=17,values=result["summary"]["compared_values"],
  reused_samples=16,newly_evaluated_samples=1,actual_new_native_workers=1,
  old_receipt_binding=source["binding"],result_binding=result["binding"],
  maximum_absolute_error=result["summary"]["maximum_absolute_error"],
  api_calls=0,compiler_calls=0,encrypted_calls=0)
 rows.append(row);print(json.dumps(row),flush=True)
q.require(hashes(old)==oldhashes,"old_records_byte_unchanged")
report=dict(format="poseidon-approximation-query-reuse-acceptance-v1",passed=3,failed=0,skipped=0,cases=rows,
 completed_dataset_samples=51,completed_dataset_values=sum(r["values"]for r in rows),
 reused_samples=48,newly_evaluated_samples=3,new_native_workers=3,api_calls=0,compiler_calls=0,encrypted_calls=0,
 old_records_byte_unchanged=True,scope_binding=scope["binding"],sdk=scope["sdk"],
 seconds=time.monotonic()-started,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
