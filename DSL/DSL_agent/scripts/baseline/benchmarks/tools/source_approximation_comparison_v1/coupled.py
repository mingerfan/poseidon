"""Existing component numerical assessment of fixed source programs and inputs."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=1800,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_numerical_assessment import TaskNumericalAssessment,SOURCE_SELECTION,NATIVE_POLICY,implementation
from candidate_bundle import check_bundle
live.no_active()
REF=q.RESULTS/"agent-source-approximation-comparison-r1695"
reference=q.read(REF/"report.json");q.bound(reference)
refplan=q.read(REF/"scope.json");q.bound(refplan)
q.require(reference["evaluated"]==12 and reference["workflow_failed"]==0,"completed_original_source_assessment")
sdk=PoseidonBackend().identity_manifest();identity=implementation(policy=NATIVE_POLICY)
q.require(reference["sdk"]==sdk["manifest_sha256"],"unchanged_backend")
OUT=q.RESULTS/"agent-coupled-precision-assessment-r1696"
q.require(not OUT.exists(),"no_automatic_restart")
baseline=live.measure(q.RESULTS);new_limit=512*1024**2
q.require(baseline+new_limit<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+new_limit,"storage")
OUT.mkdir(mode=0o700);cases=[]
for c in refplan["cases"]:
 name=c["group"]+"--"+c["id"];receipt=q.read(REF/name/"receipt.json");q.bound(receipt)
 program=q.RESULTS/c["campaign"]/c["id"]/"tasks"/c["task_id"]/"program"
 package=check_bundle(program)
 q.require(package["binding"]==receipt["bundle_binding"] and package["agent_generated"] is True,"original_agent_package")
 cases.append(dict(c,id=name,original_case_id=c["id"],input_file=str(REF/name/"inputs.json"),
  input_file_sha256=q.sha(REF/name/"inputs.json"),reference_receipt_binding=receipt["binding"],package_binding=package["binding"]))
scope=dict(format="poseidon-coupled-precision-assessment-scope-v1",
 authorization="human_all_current_plan_offline_and_live_tests_approved",cases=cases,
 original_source_report_binding=reference["binding"],sdk=sdk["manifest_sha256"],assessment_identity=identity,
 samples_per_case=8,application_criterion=None,
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),decomposition_tolerance=dict(atol=1e-12,rtol=1e-12),
 budget=dict(wall_seconds=1800,max_native_workers=12,max_encrypted_calls=12,
  new_artifact_bytes=new_limit,global_artifact_bytes=48*1024**3,reserve_bytes=4*1024**3,
  encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,native_address_space_bytes=4*1024**3,native_cpu_seconds=150,
  api_calls=0,compiler_calls=0),
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 source_and_approximation_parameters_changed=False,backend_changed=False,network_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"coupled-scope-r1696.json",scope);q.write(OUT/"scope.json",scope)
q.write((Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"docs/baseline/agent-coupled-precision-assessment-scope-r1696.json",scope)
start=time.monotonic();used=dict(assessment_reservations=0,encrypted_calls=0)
def journal():
 value=dict(format="poseidon-coupled-precision-call-ledger-v1",scope_binding=scope["binding"],used=dict(used))
 value["binding"]=q.digest(value)
 path=OUT/"call-ledger.json";temp=path.with_name(path.name+".partial");temp.write_bytes(q.canonical(value));temp.replace(path)
def guard():
 q.require(time.monotonic()-start<1750,"wall_budget")
 total=live.measure(q.RESULTS)
 q.require(total-baseline<=new_limit and total<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3,"storage_budget")
 q.require(PoseidonBackend().identity_manifest()==sdk and implementation(policy=NATIVE_POLICY)==identity,"unchanged_identities")
class Backend(PoseidonBackend):
 def execute(self,*args,**kwargs):
  guard();q.require(used["encrypted_calls"]<12,"bounded_encrypted_calls")
  used["encrypted_calls"]+=1;journal()
  result=super().execute(*args,**kwargs);guard();return result
 def qualify(self,*args,**kwargs):raise RuntimeError("compiler_not_authorized_in_reuse_assessment")
def snapshots(folder):
 return {str(p.relative_to(folder)):q.sha(p)for p in folder.rglob("*")if p.is_file()and p.name!=".env"}
rows=[];journal()
for c in cases:
 guard();folder=OUT/c["id"];folder.mkdir(mode=0o700)
 old=q.RESULTS/c["campaign"]/c["original_case_id"];store=old/"tasks";task=c["task_id"]
 app=application(WorkerContext.capture(store));app.backend=Backend()
 outcome=app.result(task);q.require(outcome["status"]=="succeeded","real_agent_task")
 meta=app._checked_task_metadata(task);request=meta["request"];saved=snapshots(store/task)
 q.require(request["request_id"]==c["request_id"] and outcome["package"]["binding"]==c["package_binding"],"frozen_program")
 q.require(q.sha(Path(c["input_file"]))==c["input_file_sha256"],"same_inputs")
 samples=q.read(Path(c["input_file"]))
 native=dict(format=SOURCE_SELECTION,input_dtypes={i["name"]:"float64"for i in request["model"]["inputs"]},
  trust_local_code=True,evidence_directory=str(folder/"native"))
 used["assessment_reservations"]+=1;journal()
 try:
  receipt=TaskNumericalAssessment(app).assess(task,samples,expected_package_binding=c["package_binding"],
    application_criterion=None,native_source=native)
  q.write(folder/"receipt.json",receipt);q.bound(receipt)
  q.require(receipt["encrypted_execution"] is True and receipt["source_reference_evaluated"] is True and
   receipt["agent_calls"]==receipt["compiler_calls"]==0 and receipt["samples"]==8,"real_same_sample_execution")
  ckks=receipt["ckks_comparison"];decomp=receipt["decomposition_comparison"]
  delta=receipt["source_conversion_and_approximation_comparison"];total=receipt["source_total_comparison"]
  q.require(ckks["atol"]==1e-5 and ckks["rtol"]==1e-4 and decomp["atol"]==1e-12 and decomp["rtol"]==1e-12,"frozen_tolerances")
  prior=q.read(REF/c["id"]/"receipt.json");q.bound(prior)
  q.require(receipt["supplied_input_sha256"]==prior["supplied_input_sha256"] and
    receipt["bundle_binding"]==prior["bundle_binding"],"joined_input_and_package_bindings")
  q.require(abs(delta["max_absolute_error"]-prior["original_source_vs_approximate_comparison"]["max_absolute_error"])<=1e-12,
    "same_source_approximation_measurement")
  q.require(receipt["original_source_accuracy_certified"] is False and receipt["source_application_criterion_validated"] is False,
    "separate_application_accuracy")
  row=dict(case=c["id"],status=receipt["status"],samples=8,values=ckks["compared_values"],
   ckks_passed=ckks["passed"],ckks_maximum_absolute_error=ckks["max_absolute_error"],
   decomposition_passed=decomp["passed"],decomposition_maximum_absolute_error=decomp["max_absolute_error"],
   original_vs_approximate_passed=delta["passed"],original_vs_approximate_maximum_absolute_error=delta["max_absolute_error"],
   original_vs_encrypted_passed=total["passed"],original_vs_encrypted_maximum_absolute_error=total["max_absolute_error"],
   receipt_binding=receipt["binding"],package_binding=c["package_binding"],request_id=c["request_id"],
   reference_receipt_binding=prior["binding"],application_accuracy_certified=False,
   encrypted_execution=True,new_agent_calls=0,new_compiler_calls=0)
 except Exception as error:
  import traceback
  (folder/"private-failure.log").write_text(traceback.format_exc())
  row=dict(case=c["id"],status="failed",code="coupled_precision_assessment_rejected",error_type=type(error).__name__)
 q.require(app.result(task)==outcome and snapshots(store/task)==saved,"historical_records_unchanged")
 row["historical_records_unchanged"]=True;q.write(folder/"case-report.json",row);rows.append(row)
 print(json.dumps(row),flush=True);guard()
evaluated=[r for r in rows if r.get("encrypted_execution")]
report=dict(format="poseidon-coupled-precision-assessment-report-v1",planned=12,evaluated=len(evaluated),
 workflow_failed=12-len(evaluated),not_run=0,skipped=0,cases=rows,
 ckks_passed=sum(r["ckks_passed"]for r in evaluated),ckks_failed=sum(not r["ckks_passed"]for r in evaluated),
 decomposition_passed=sum(r["decomposition_passed"]for r in evaluated),
 original_vs_approximate_passed=sum(r["original_vs_approximate_passed"]for r in evaluated),
 original_vs_approximate_failed=sum(not r["original_vs_approximate_passed"]for r in evaluated),
 original_vs_encrypted_passed=sum(r["original_vs_encrypted_passed"]for r in evaluated),
 original_vs_encrypted_failed=sum(not r["original_vs_encrypted_passed"]for r in evaluated),
 ckks_maximum_absolute_error=max((r["ckks_maximum_absolute_error"]for r in evaluated),default=None),
 source_maximum_absolute_error=max((r["original_vs_approximate_maximum_absolute_error"]for r in evaluated),default=None),
 total_maximum_absolute_error=max((r["original_vs_encrypted_maximum_absolute_error"]for r in evaluated),default=None),
 samples=sum(r["samples"]for r in evaluated),values=sum(r["values"]for r in evaluated),
 sdk=sdk["manifest_sha256"],scope_binding=scope["binding"],assessment_identity=identity,used=used,
 api_calls=0,compiler_calls=0,encrypted_calls=used["encrypted_calls"],application_criterion=None,
 original_application_accuracy_certified=False,source_and_approximation_parameters_changed=False,
 network_changed=False,backend_changed=False,seconds=time.monotonic()-start,
 new_bytes=live.measure(q.RESULTS)-baseline,result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
report["binding"]=q.digest(report);q.write(HERE/"coupled-report-r1696.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
