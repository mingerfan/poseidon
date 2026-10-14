"""Fresh native dataset assessment, durable reload, no FHE or generation."""
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
import numpy as np
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_approximation_assessment import ApplicationApproximationAssessment,identity
from task_approximation_assessment import implementation as reference_identity
from assessment_array_dataset import export_dataset
live.no_active()
controls=q.read(HERE/"guard-controls.json");q.bound(controls)
q.require(controls["status"]=="passed" and controls["tests"]==25 and controls["sdk_unchanged"],"current_controls")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==controls["sdk"],"sdk")
for path,want in controls["source_files"].items():q.require(q.sha(path)==want,"tested_source")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
q.require(prior["sdk"]==sdk["manifest_sha256"],"reusable_historical_agent_packages")
OUT=q.RESULTS/"agent-approximation-dataset-acceptance-r1612"
q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-approximation-dataset-scope-v1",authorization="human_explicit_current_plan_test_approval",
 sdk=sdk["manifest_sha256"],controller_identity=identity(),reference_identity=reference_identity(),
 controls_binding=controls["binding"],source_files={**controls["source_files"],str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 cases=[dict(id=name,dtype=dtype,metric=metric)for name,dtype,metric in (
  ("matrix","float32","top1_accuracy_drop"),("negative_axes","float64","max_absolute_error"),
  ("scalar_intermediate","float64","max_absolute_error"))],samples_per_case=17,part_samples=16,
 budget=dict(wall_seconds=1200,max_native_workers=6,api_calls=0,compiler_calls=0,encrypted_calls=0,
  new_artifact_bytes=128*1024**2,total_artifact_bytes=48*1024**3,min_free_bytes=4*1024**3),
 network_changed=False,backend_changed=False,parameters_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
executions_before=set(q.RESULTS.glob("poseidon-program-execution-*"))
def hashes(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
started=time.monotonic();rows=[]
for spec in scope["cases"]:
 q.require(time.monotonic()-started<1100 and live.measure(OUT)<=scope["budget"]["new_artifact_bytes"]and
  live.measure(q.RESULTS)<=scope["budget"]["total_artifact_bytes"]and
  shutil.disk_usage(q.RESULTS).free>=scope["budget"]["min_free_bytes"],"case_boundary")
 folder=OUT/spec["id"];folder.mkdir(mode=0o700)
 old=Path(prior["campaign"])/spec["id"];task=q.read(old/"task-id.json")["task_id"];store=old/"tasks"
 frozen_task=hashes(store/task);app=application(WorkerContext.capture(store));outcome=app.result(task)
 q.require(outcome["status"]=="succeeded","existing_program")
 request=app._checked_task_metadata(task)["request"]
 dtype=np.float32 if spec["dtype"]=="float32"else np.float64
 rng=np.random.default_rng(2026101412)
 inputs={entry["name"]:rng.uniform(-.3,.3,(17,*entry["shape"])).astype(dtype)for entry in request["model"]["inputs"]}
 labels=None
 if spec["metric"]=="top1_accuracy_drop":
  labels=dict(output="out",class_axis=1,values=np.random.default_rng(2026101413).integers(0,2,(17,2),dtype=np.int64))
 manifest=export_dataset(request,inputs,folder/"dataset",labels=labels)
 criterion=dict(kind="top1_accuracy_drop",max_drop=.01)if labels is not None else dict(kind="max_absolute_error",limit=1e-5)
 control=ApplicationApproximationAssessment(app,folder/"assessment-store")
 handle=control.submit(task,folder/"dataset/dataset.json",expected_dataset_binding=manifest["binding"],
  input_dtypes={entry["name"]:spec["dtype"]for entry in request["model"]["inputs"]},trust_local_code=True,
  application_criterion=criterion,max_new_assessments=2,part_samples=16,total_wall_seconds=500)
 q.write(folder/"handle.json",handle)
 queued=control.status(handle)
 q.require(queued["status"]=="queued" and queued["budget"]["used"]==0,"zero_cost_submit")
 partial=control.advance(handle,max_new_parts=1);q.write(folder/"partial.json",partial)
 q.require(partial["status"]=="ready" and partial["budget"]["used"]==1 and
  partial["completed_parts"]==1 and partial["summary"]["complete"] is False and
  partial["acceptance"]["application_criterion_validated"] is False,"partial_never_passes")
 other_app=application(WorkerContext.capture(store));other=ApplicationApproximationAssessment(other_app,folder/"assessment-store")
 resumed=other.result(handle)
 q.require(resumed["binding"]==partial["binding"],"partial_reload_without_execution")
 final=other.advance(handle,max_new_parts=1);q.write(folder/"result.json",final)
 q.require(final["status"]=="succeeded" and final["budget"]["used"]==2 and final["completed_parts"]==2 and
  final["summary"]["evaluated_samples"]==17 and final["acceptance"]["application_criterion_validated"] is True,"complete_dataset_criterion")
 q.require(final["encrypted_execution"] is False and final["ckks_validation_performed"] is False and
  final["acceptance"]["numerical_on_declared_dataset"] is False,"distinct_reference_only_grade")
 q.require(other.result(handle)["binding"]==final["binding"] and
  other.advance(handle)["binding"]==final["binding"] and other.cancel(handle)["binding"]==final["binding"],"stable_terminal_no_execution")
 q.require(outcome==app.result(task) and hashes(store/task)==frozen_task,"historical_agent_package_preserved")
 native_reports=list((folder/"assessment-store"/handle["evaluation_id"]/"native").glob("*/native-reference.json"))
 q.require(len(native_reports)==2,"two_actual_native_workers")
 for path in native_reports:
  native=q.read(path);q.bound(native)
  q.require(native["native_dtype_preserved"] is True and native["model_imported_in_host"] is False and
   native["sandbox"]["network_isolated"] is True and native["encrypted_execution"] is False,"actual_isolated_source")
 q.require(identity()==scope["controller_identity"] and reference_identity()==scope["reference_identity"]and
  PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 row=dict(case=spec["id"],status="passed",input_dtype=spec["dtype"],samples=17,parts=2,
  values=final["summary"]["compared_values"],maximum_absolute_error=final["summary"]["maximum_absolute_error"],
  metric=final["summary"]["application_criterion"],result_binding=final["binding"],
  reload_and_reuse_passed=True,historical_agent_package_unchanged=True,new_native_workers=2,
  api_calls=0,compiler_calls=0,encrypted_calls=0)
 rows.append(row);print(json.dumps(row),flush=True)
q.require(set(q.RESULTS.glob("poseidon-program-execution-*"))==executions_before,"no_new_encrypted_execution")
report=dict(format="poseidon-approximation-dataset-acceptance-v1",passed=3,failed=0,skipped=0,cases=rows,
 samples=51,values=sum(row["values"]for row in rows),new_native_workers=6,
 api_calls=0,compiler_calls=0,encrypted_calls=0,scope_binding=scope["binding"],sdk=scope["sdk"],
 controller_identity=scope["controller_identity"],reference_identity=scope["reference_identity"],
 seconds=time.monotonic()-started,bytes=live.measure(OUT),declared_datasets_only=True,
 original_source_accuracy_universally_certified=False)
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
