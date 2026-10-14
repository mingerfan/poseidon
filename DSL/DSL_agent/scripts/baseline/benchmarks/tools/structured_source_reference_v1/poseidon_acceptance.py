"""Three fresh original-source assessments through the existing application API."""
from pathlib import Path
import sys,shlex,json,time,shutil,hashlib
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=1200,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_assessment import ApplicationAssessment
from task_numerical_assessment import NATIVE_POLICY,implementation
from assessment_array_dataset import export_dataset
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed","focused_assessment_controls")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
audit=q.read(T/"tensor_contraction_live_audit_v1/report.json");q.bound(audit)
identity=PoseidonBackend().identity_manifest()
q.require(identity["manifest_sha256"]==prior["sdk"]and audit["audit_status"]=="passed","reuse_existing_agent_packages")
OUT=q.RESULTS/"agent-native-call-poseidon-regression-r1638"
q.require(not OUT.exists(),"new_assessment_directory")
q.require(live.measure(q.RESULTS)+256*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+256*1024**2,"assessment_storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-current-application-assessment-scope-v1",authorization="human_explicit_current_plan_test_approval",
 sdk=prior["sdk"],reference_implementation=implementation(policy=NATIVE_POLICY),
 cases=[dict(id=name,dtype=dtype)for name,dtype in(("matrix","float32"),("negative_axes","float64"),("scalar_intermediate","float64"))],
 expected_criterion=dict(kind="max_absolute_error",limit=1e-5),historical_agent_audit_binding=audit["binding"],
 budget=dict(wall_seconds=1200,max_new_assessments=3,max_encrypted_calls=3,max_native_workers=3,new_bytes=256*1024**2,
  total_bytes=48*1024**3,reserve_bytes=4*1024**3,compiler_calls=0,api_calls=0,compiler_jobs=2,link_jobs=1),
 source_files={str(p):q.sha(p)for p in HERE.glob("*.py")},network_changed=False,backend_changed=False,parameters_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"poseidon-scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
started=time.monotonic();rows=[]
for c in scope["cases"]:
 try:
  q.require(time.monotonic()-started<1100,"assessment_time")
  oldcase=Path(prior["campaign"])/c["id"];task_id=q.read(oldcase/"task-id.json")["task_id"]
  oldtask=oldcase/"tasks"/task_id;old_hashes=hashes(oldtask)
  folder=OUT/c["id"];folder.mkdir(mode=0o700);store=folder/"tasks";store.mkdir(mode=0o700)
  shutil.copytree(oldtask,store/task_id)
  q.require(hashes(store/task_id)==old_hashes,"copied_program_unchanged")
  app=application(WorkerContext.capture(store));before=app.result(task_id);q.require(before["status"]=="succeeded","existing_program")
  metadata=app._checked_task_metadata(task_id);request=metadata["request"]
  rng=np.random.default_rng(2026101401);dtype=np.float32 if c["dtype"]=="float32"else np.float64
  inputs={spec["name"]:rng.uniform(-.2,.2,(2,*spec["shape"])).astype(dtype)for spec in request["model"]["inputs"]}
  dataset=folder/"dataset"
  exported=export_dataset(request,inputs,dataset,sample_ids=["sample0","sample1"])
  manifest=q.read(dataset/"dataset.json")
  service=ApplicationAssessment(app,folder/"assessment-store")
  handle=service.submit(task_id,dataset/"dataset.json",expected_dataset_binding=manifest["binding"],
    reference_policy=NATIVE_POLICY,application_criterion=scope["expected_criterion"],max_new_assessments=1,
    part_samples=2,input_dtypes={spec["name"]:c["dtype"]for spec in request["model"]["inputs"]},
    trust_local_code=True,total_wall_seconds=350)
  q.write(folder/"handle.json",handle)
  result=service.advance(handle,max_new_parts=1);q.write(folder/"result.json",result)
  q.require(result["status"]=="succeeded"and result["budget"]["used"]==1 and result["completed_parts"]==1,"native_numerical_assessment")
  summary=result["summary"]
  q.require(result["acceptance"]["numerical_on_declared_dataset"]is True and
    result["acceptance"]["application_criterion_validated"]is True and summary["evaluated_samples"]==2,"split_acceptance_levels")
  receipts=list((folder/"assessment-store"/handle["evaluation_id"]/"receipts").glob("*.json"))
  q.require(len(receipts)==1,"single_receipt");receipt=q.read(receipts[0])
  q.require(receipt["encrypted_execution"]is True and receipt["source_reference_evaluated"]is True and
    receipt["ckks_comparison"]["passed"]is True and receipt["samples"]==2,"actual_runtime_and_source")
  repeat=ApplicationAssessment(app,folder/"assessment-store").result(handle)
  q.require(repeat["binding"]==result["binding"]and service.advance(handle)["binding"]==result["binding"],"assessment_reload_without_execution")
  q.require(app.result(task_id)==before and hashes(oldtask)==old_hashes and
    hashes(store/task_id)==old_hashes and PoseidonBackend().identity_manifest()==identity,"original_package_and_sdk_preserved")
  row=dict(case=c["id"],status="passed",input_dtype=c["dtype"],samples=2,values=receipt["ckks_comparison"]["compared_values"],
    ckks_maximum_absolute_error=receipt["ckks_comparison"]["max_absolute_error"],
    original_source_maximum_absolute_error=summary["application_criterion"]["max_absolute_error"],
    receipt_binding=receipt["binding"],result_binding=result["binding"],reload_without_execution=True,
    new_encrypted_calls=1,new_native_workers=1,compiler_calls=0,api_calls=0)
 except Exception as error:
  import traceback
  (HERE/(c["id"]+"-private-failure.txt")).write_text(traceback.format_exc())
  row=dict(case=c["id"],status="failed",error_type=type(error).__name__,diagnostic="private_traceback_retained")
 rows.append(row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=scope["budget"]["new_bytes"],"assessment_artifact_budget")
report=dict(format="poseidon-current-original-source-assessment-acceptance-v1",scope_binding=scope["binding"],sdk=prior["sdk"],
 cases=rows,passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]!="passed"for r in rows),skipped=0,
 new_encrypted_calls=sum(r.get("new_encrypted_calls",0)for r in rows),new_native_workers=sum(r.get("new_native_workers",0)for r in rows),
 compiler_calls=0,api_calls=0,seconds=time.monotonic()-started,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(OUT/"report.json",report);q.write(HERE/"poseidon-report.json",report)
raise SystemExit(0 if report["failed"]==0 else 1)
