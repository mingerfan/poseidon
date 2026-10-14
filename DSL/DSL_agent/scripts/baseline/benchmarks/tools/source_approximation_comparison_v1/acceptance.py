"""Evaluate original source and declared approximation; no generation or FHE."""
from pathlib import Path
import sys,shlex,time,json,shutil,hashlib
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
sdk=PoseidonBackend().identity_manifest();assessment_identity=implementation()
sources=[("compound","agent-compound-source-live-r1687",T/"source_compound_calibration_v1/live-audit-r1688.json"),
         ("normalization","agent-normalization-source-live-r1693",T/"source_normalization_calibration_v1/live-audit-r1694.json")]
cases=[]
for group,campaign,audit_path in sources:
 audit=q.read(audit_path);q.bound(audit)
 q.require(audit["passed"]==6 and audit["sdk"]==sdk["manifest_sha256"],"prior_audited_scope")
 plan=q.read(q.RESULTS/campaign/"live-plan.json");q.bound(plan)
 for c in plan["cases"]:
  cases.append(dict(group=group,id=c["id"],campaign=campaign,task_id=c["task_id"],request_id=c["request_id"],
    prior_plan_binding=plan["binding"],prior_audit_binding=audit["binding"]))
OUT=q.RESULTS/"agent-source-approximation-comparison-r1695"
q.require(not OUT.exists(),"no_automatic_restart")
baseline=live.measure(q.RESULTS);new_limit=128*1024**2
q.require(baseline+new_limit<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+new_limit,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-source-approximation-comparison-scope-v1",
 authorization="human_all_current_plan_offline_and_live_tests_approved",sdk=sdk["manifest_sha256"],
 assessment_identity=assessment_identity,cases=cases,samples_per_case=8,application_criterion=None,
 selection="zero, signed, boundary, five deterministic unseen random samples",
 reference_comparison_tolerance=dict(atol=1e-5,rtol=1e-4),
 decomposition_comparison_tolerance=dict(atol=1e-12,rtol=1e-12),
 budget=dict(wall_seconds=1200,max_native_workers=12,new_artifact_bytes=new_limit,
  global_artifact_bytes=48*1024**3,reserve_bytes=4*1024**3,api_calls=0,compiler_calls=0,encrypted_calls=0),
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 source_and_approximation_parameters_changed=False,backend_changed=False,network_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"scope-r1695.json",scope);q.write(OUT/"scope.json",scope)
q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-source-approximation-comparison-scope-r1695.json",scope)
before_executions=set(q.RESULTS.glob("poseidon-program-execution-*"))
start=time.monotonic();rows=[]
def snapshot(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
def guard():
 q.require(time.monotonic()-start<1150,"wall_budget")
 total=live.measure(q.RESULTS)
 q.require(total-baseline<=new_limit and total<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3,"storage_budget")
 q.require(PoseidonBackend().identity_manifest()==sdk and implementation()==assessment_identity,"frozen_identities")
for index,c in enumerate(cases):
 guard();name=c["group"]+"--"+c["id"];folder=OUT/name;folder.mkdir(mode=0o700)
 old=q.RESULTS/c["campaign"]/c["id"];store=old/"tasks";task=c["task_id"]
 app=application(WorkerContext.capture(store));outcome=app.result(task)
 q.require(outcome["status"]=="succeeded","existing_real_agent_package")
 metadata=app._checked_task_metadata(task);request=metadata["request"]
 q.require(request["request_id"]==c["request_id"],"same_original_request")
 original_hashes=snapshot(store/task)
 rng=np.random.default_rng(2026101695+index);samples=[]
 for sample_index in range(8):
  values={}
  for entry in request["model"]["inputs"]:
   shape=entry["shape"];size=int(np.prod(shape))
   if sample_index==0:value=np.zeros(shape,dtype=np.float64)
   elif sample_index==1:value=(np.where(np.arange(size)%2==0,.2,-.1)).reshape(shape)
   elif sample_index==2:value=(np.where(np.arange(size)%3==0,.4,-.4)).reshape(shape)
   else:value=rng.uniform(-.25,.25,shape)
   values[entry["name"]]=value.tolist()
  samples.append(values)
 q.write(folder/"inputs.json",samples)
 selection=dict(format=SOURCE_SELECTION,input_dtypes={entry["name"]:"float64"for entry in request["model"]["inputs"]},
  trust_local_code=True,evidence_directory=str(folder/"native"))
 try:
  receipt=TaskApproximationAssessment(app).assess(task,samples,
    expected_package_binding=outcome["package"]["binding"],native_source=selection,application_criterion=None)
  q.write(folder/"receipt.json",receipt);q.bound(receipt)
  q.require(receipt["status"]=="passed" and receipt["source_reference_evaluated"] is True and receipt["samples"]==8,"source_workflow")
  q.require(receipt["encrypted_execution"] is False and receipt["ckks_validation_performed"] is False and
    receipt["supplied_inputs_numerically_validated"] is False and receipt["agent_calls"]==receipt["compiler_calls"]==0,
    "reference_only_scope")
  q.require(receipt["application_criterion_validated"] is False and receipt["original_source_accuracy_certified"] is False,
    "no_application_accuracy_claim")
  native=q.read(folder/"native/native-reference.json");q.bound(native)
  q.require(native["native_dtype_preserved"] is True and native["model_imported_in_host"] is False and
    native["sandbox"]["network_isolated"] is True and native["samples"]==8,"real_isolated_source")
  cmp=receipt["original_source_vs_approximate_comparison"];decomp=receipt["decomposition_comparison"]
  q.require(decomp["passed"] and decomp["atol"]==1e-12 and decomp["rtol"]==1e-12,"independent_decomposition")
  row=dict(case=name,status="evaluated",samples=8,values=cmp["compared_values"],request_id=request["request_id"],
    original_vs_approximate_maximum_absolute_error=cmp["max_absolute_error"],
    original_vs_approximate_mae=cmp.get("mae"),original_vs_approximate_at_ckks_tolerance_passed=cmp["passed"],
    decomposition_passed=decomp["passed"],decomposition_maximum_absolute_error=decomp["max_absolute_error"],
    receipt_binding=receipt["binding"],source_reference_binding=native["binding"],package_binding=outcome["package"]["binding"],
    application_accuracy_certified=False,encrypted_execution=False,new_native_workers=1)
 except Exception as error:
  import traceback
  (folder/"private-failure.log").write_text(traceback.format_exc())
  row=dict(case=name,status="failed",code="source_approximation_assessment_rejected",error_type=type(error).__name__)
 q.require(app.result(task)==outcome and snapshot(store/task)==original_hashes,"historical_program_unchanged")
 row["historical_program_unchanged"]=True;rows.append(row);q.write(folder/"case-report.json",row)
 print(json.dumps(row),flush=True);guard()
q.require(set(q.RESULTS.glob("poseidon-program-execution-*"))==before_executions,"no_fhe_execution")
evaluated=[r for r in rows if r["status"]=="evaluated"]
report=dict(format="poseidon-source-approximation-comparison-report-v1",planned=12,evaluated=len(evaluated),
 workflow_failed=12-len(evaluated),not_run=0,skipped=0,cases=rows,
 original_vs_approximate_at_ckks_tolerance_passed=sum(r["original_vs_approximate_at_ckks_tolerance_passed"]for r in evaluated),
 original_vs_approximate_at_ckks_tolerance_failed=sum(not r["original_vs_approximate_at_ckks_tolerance_passed"]for r in evaluated),
 decomposition_passed=sum(r["decomposition_passed"]for r in evaluated),
 samples=sum(r["samples"]for r in evaluated),values=sum(r["values"]for r in evaluated),
 source_maximum_absolute_error=max((r["original_vs_approximate_maximum_absolute_error"]for r in evaluated),default=None),
 decomposition_maximum_absolute_error=max((r["decomposition_maximum_absolute_error"]for r in evaluated),default=None),
 sdk=sdk["manifest_sha256"],scope_binding=scope["binding"],assessment_identity=assessment_identity,
 api_calls=0,compiler_calls=0,encrypted_calls=0,new_native_workers=len(evaluated),
 original_application_accuracy_certified=False,application_criterion=None,
 source_and_approximation_parameters_changed=False,network_changed=False,backend_changed=False,
 seconds=time.monotonic()-start,new_bytes=live.measure(q.RESULTS)-baseline,
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
report["binding"]=q.digest(report);q.write(HERE/"report-r1695.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
