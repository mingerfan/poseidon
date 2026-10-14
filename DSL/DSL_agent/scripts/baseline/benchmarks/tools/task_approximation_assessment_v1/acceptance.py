"""Original versus approximation interface; real isolated source, no FHE."""
from pathlib import Path
import sys,shlex,time,json,shutil
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
from task_approximation_assessment import TaskApproximationAssessment,implementation
from task_numerical_assessment import SOURCE_SELECTION
live.no_active()
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed" and controls["sdk_unchanged"],"controls")
identity=PoseidonBackend().identity_manifest();sdk=identity["manifest_sha256"]
q.require(sdk==controls["sdk"],"current_sdk")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
OUT=q.RESULTS/"agent-approximation-reference-acceptance-r1608"
q.require(not OUT.exists(),"new_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-reference-only-assessment-scope-v1",authorization="human_explicit_current_plan_test_approval",
 sdk=sdk,assessment_identity=implementation(),controls_binding=controls["binding"],
 cases=[dict(id=name,dtype=dtype)for name,dtype in (("matrix","float32"),("negative_axes","float64"),("scalar_intermediate","float64"))],
 samples_per_case=8,criterion=dict(kind="max_absolute_error",limit=1e-5),
 budget=dict(wall_seconds=1200,max_native_workers=3,new_artifact_bytes=128*1024**2,total_artifact_bytes=48*1024**3,min_free_bytes=4*1024**3,api_calls=0,compiler_calls=0,encrypted_calls=0),
 source_files={str(p):q.sha(p)for p in (HERE/"acceptance.py",B/"task_approximation_assessment.py",B/"test_task_approximation_assessment.py")},
 backend_changed=False,parameters_changed=False,network_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"reference-scope.json",scope);q.write(OUT/"scope.json",scope)
executions_before=set(q.RESULTS.glob("poseidon-program-execution-*"))
def hashes(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
started=time.monotonic();rows=[]
for case in scope["cases"]:
 q.require(time.monotonic()-started<1100 and live.measure(OUT)<=scope["budget"]["new_artifact_bytes"] and
  live.measure(q.RESULTS)<=scope["budget"]["total_artifact_bytes"] and shutil.disk_usage(q.RESULTS).free>=scope["budget"]["min_free_bytes"],"case_boundary")
 folder=OUT/case["id"];folder.mkdir(mode=0o700)
 oldcase=Path(prior["campaign"])/case["id"];task=q.read(oldcase/"task-id.json")["task_id"]
 store=oldcase/"tasks";before_hashes=hashes(store/task)
 app=application(WorkerContext.capture(store));outcome=app.result(task)
 q.require(outcome["status"]=="succeeded","existing_true_agent_program")
 request=app._checked_task_metadata(task)["request"]
 rng=np.random.default_rng(2026101409);dtype=np.float32 if case["dtype"]=="float32"else np.float64
 samples=[{entry["name"]:rng.uniform(-.3,.3,entry["shape"]).astype(dtype).tolist()for entry in request["model"]["inputs"]}for _ in range(8)]
 selection=dict(format=SOURCE_SELECTION,input_dtypes={entry["name"]:case["dtype"]for entry in request["model"]["inputs"]},
  trust_local_code=True,evidence_directory=str(folder/"native"))
 receipt=TaskApproximationAssessment(app).assess(task,samples,expected_package_binding=outcome["package"]["binding"],
   native_source=selection,application_criterion=scope["criterion"])
 q.write(folder/"receipt.json",receipt);q.bound(receipt)
 q.require(receipt["status"]=="passed" and receipt["source_reference_evaluated"] is True and
  receipt["application_criterion_validated"] is True and receipt["samples"]==8,"source_approximation_acceptance")
 q.require(receipt["encrypted_execution"] is False and receipt["ckks_validation_performed"] is False and
  receipt["supplied_inputs_numerically_validated"] is False and receipt["agent_calls"]==receipt["compiler_calls"]==0,"no_encrypted_claim")
 native=q.read(folder/"native/native-reference.json");q.bound(native)
 q.require(native["native_dtype_preserved"] is True and native["model_imported_in_host"] is False and
  native["sandbox"]["network_isolated"] is True and native["samples"]==8,"actual_original_source")
 q.require(app.result(task)==outcome and hashes(store/task)==before_hashes,"historical_program_unchanged")
 q.require(implementation()==scope["assessment_identity"] and PoseidonBackend().identity_manifest()==identity,"frozen_identity")
 metric=receipt["application_criterion"]
 rows.append(dict(case=case["id"],status="passed",samples=8,values=receipt["original_source_vs_approximate_comparison"]["compared_values"],
  source_input_dtypes=receipt["source_input_dtypes"],maximum_absolute_error=metric["max_absolute_error"],
  binding=receipt["binding"],native_report_binding=native["binding"],historical_program_unchanged=True,
  new_native_workers=1,encrypted_execution=False))
 print(json.dumps(rows[-1]),flush=True)
q.require(set(q.RESULTS.glob("poseidon-program-execution-*"))==executions_before,"no_new_encrypted_execution_directory")
report=dict(format="poseidon-reference-only-assessment-acceptance-v1",passed=3,failed=0,skipped=0,cases=rows,
 samples=24,values=sum(r["values"]for r in rows),api_calls=0,compiler_calls=0,encrypted_calls=0,new_native_workers=3,
 sdk=sdk,scope_binding=scope["binding"],assessment_identity=scope["assessment_identity"],seconds=time.monotonic()-started,
 bytes=live.measure(OUT),finite_metric_only=True,original_source_accuracy_universally_certified=False)
report["binding"]=q.digest(report);q.write(HERE/"reference-acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
