"""Recompute each error boundary from retained source and real decoded outputs."""
from pathlib import Path
import sys,shlex,json,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from task_numerical_assessment import mathematical,flatten,implementation,NATIVE_POLICY
from candidate_bundle import check_bundle
from numerical_results import compare
live.no_active()
D=q.RESULTS/"agent-coupled-precision-assessment-r1696";R=q.RESULTS/"agent-source-approximation-comparison-r1695"
report=q.read(D/"report.json");q.bound(report);scope=q.read(D/"scope.json");q.bound(scope)
source=q.read(R/"report.json");q.bound(source)
q.require(report["scope_binding"]==scope["binding"] and scope["original_source_report_binding"]==source["binding"],"joined_report_bindings")
q.require(report["evaluated"]==report["ckks_passed"]==12 and report["workflow_failed"]==report["ckks_failed"]==report["not_run"]==report["skipped"]==0,"complete_scope")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==report["sdk"] and implementation(policy=NATIVE_POLICY)==scope["assessment_identity"],"same_runtime_and_assessment")
q.require(all(q.sha(Path(n))==h for n,h in scope["source_files"].items()),"frozen_runner")
rows=[];package_files=0;native_files=0
for c in scope["cases"]:
 folder=D/c["id"];receipt=q.read(folder/"receipt.json");q.bound(receipt)
 q.require(receipt["status"]=="passed" and receipt["bundle_binding"]==c["package_binding"],"valid_receipt")
 old=q.RESULTS/c["campaign"]/c["original_case_id"]
 app=application(WorkerContext.capture(old/"tasks"));metadata=app._checked_task_metadata(c["task_id"])
 request=metadata["request"];model=request["model"];original=metadata["model_provenance"]["original"]
 package=check_bundle(app._dir(c["task_id"])/"program")
 q.require(package["agent_generated"] is True and package["binding"]==c["package_binding"],"unchanged_real_agent_program")
 for name,want in package["files"].items():q.require(q.sha(app._dir(c["task_id"])/"program"/name)==want,"program_hash")
 package_files+=len(package["files"])
 q.require(q.sha(Path(c["input_file"]))==c["input_file_sha256"],"frozen_inputs")
 samples=q.read(Path(c["input_file"]))
 wire=[{n:np.asarray(v,dtype=np.float64)for n,v in s.items()}for s in samples]
 q.require(q.digest(samples)==receipt["supplied_input_sha256"],"same_evaluated_inputs")
 native=q.read(folder/"native/native-reference.json");q.bound(native)
 q.require(native["native_dtype_preserved"] and native["model_imported_in_host"] is False and
  native["sandbox"]["network_isolated"] and native["samples"]==8,"real_original_source")
 evidence=Path(receipt["execution_evidence"]);execution=q.read(evidence/"report.json")
 q.require(execution["status"]=="executed" and execution["backend"]=="poseidon_cpu" and
  execution["encrypted_execution"] is True and execution["executor_secret_key"] is False and
  execution["agent_calls"]==execution["compiler_calls"]==0 and execution["bundle_binding"]==package["binding"] and
  execution["samples"]==8,"actual_poseidon_execution")
 actual=[{n:np.asarray(v,dtype=np.float64)for n,v in s.items()}for s in execution["outputs"]]
 values=[{n:np.asarray(v,dtype=np.float64)for n,v in s.items()}for s in native["outputs"]]
 approximate=[mathematical(original,s,ideal=False)for s in wire]
 ideal=[mathematical(original,s,ideal=True)for s in wire]
 lowered=[mathematical(model,s)for s in wire]
 components=[
  ("source_vs_registered_mathematics",values,ideal,1e-12,1e-12),
  ("decomposition_comparison",lowered,approximate,1e-12,1e-12),
  ("ckks_comparison",actual,lowered,1e-5,1e-4),
  ("source_conversion_and_approximation_comparison",approximate,values,1e-5,1e-4),
  ("source_total_comparison",actual,values,1e-5,1e-4)]
 comparisons={}
 for name,left,right,atol,rtol in components:
  measured=compare(flatten(left,model),flatten(right,model),atol,rtol)
  q.require(measured==receipt[name],"recomputed_"+name);comparisons[name]=measured
 q.require(comparisons["source_vs_registered_mathematics"]["passed"] and
  comparisons["decomposition_comparison"]["passed"] and comparisons["ckks_comparison"]["passed"],"independent_error_separation")
 # Preserve diagnostic failures as failures; no accuracy criterion was requested.
 q.require(comparisons["source_conversion_and_approximation_comparison"]["passed"] is False and
  comparisons["source_total_comparison"]["passed"] is False,"retained_original_precision_gap")
 for f in (folder/"native").rglob("*"):
  if f.is_file():native_files+=1
 row=dict(case=c["id"],request_id=request["request_id"],package_binding=package["binding"],
  samples=8,values=comparisons["ckks_comparison"]["compared_values"],
  source_mathematics_passed=True,source_mathematics_maximum_absolute_error=comparisons["source_vs_registered_mathematics"]["max_absolute_error"],
  decomposition_passed=True,decomposition_maximum_absolute_error=comparisons["decomposition_comparison"]["max_absolute_error"],
  ckks_passed=True,ckks_maximum_absolute_error=comparisons["ckks_comparison"]["max_absolute_error"],
  original_vs_approximate_passed=False,original_vs_approximate_maximum_absolute_error=comparisons["source_conversion_and_approximation_comparison"]["max_absolute_error"],
  original_vs_encrypted_passed=False,original_vs_encrypted_maximum_absolute_error=comparisons["source_total_comparison"]["max_absolute_error"],
  source_receipt_binding=receipt["binding"],execution_report_sha256=q.sha(evidence/"report.json"),
  native_report_binding=native["binding"],program_origin="retained_current_real_agent",manually_modified=False)
 rows.append(row)
ledger=q.read(D/"call-ledger.json");q.bound(ledger)
q.require(ledger["used"]==report["used"]=={"assessment_reservations":12,"encrypted_calls":12},"actual_call_accounting")
out=dict(format="poseidon-coupled-precision-assessment-audit-v1",planned=12,evaluated=12,workflow_failed=0,not_run=0,skipped=0,
 source_mathematics_passed=12,decomposition_passed=12,ckks_passed=12,ckks_failed=0,
 original_vs_approximate_passed=0,original_vs_approximate_failed=12,
 original_vs_encrypted_passed=0,original_vs_encrypted_failed=12,cases=rows,
 samples=96,values=sum(r["values"]for r in rows),
 source_mathematics_maximum_absolute_error=max(r["source_mathematics_maximum_absolute_error"]for r in rows),
 decomposition_maximum_absolute_error=max(r["decomposition_maximum_absolute_error"]for r in rows),
 ckks_maximum_absolute_error=max(r["ckks_maximum_absolute_error"]for r in rows),
 original_vs_approximate_maximum_absolute_error=max(r["original_vs_approximate_maximum_absolute_error"]for r in rows),
 original_vs_encrypted_maximum_absolute_error=max(r["original_vs_encrypted_maximum_absolute_error"]for r in rows),
 report_binding=report["binding"],scope_binding=scope["binding"],source_report_binding=source["binding"],sdk=report["sdk"],
 package_files_verified=package_files,native_evidence_files_retained=native_files,
 actual_encrypted_calls=12,actual_native_workers=12,new_agent_calls=0,new_compiler_calls=0,
 audit_new_agent_calls=0,audit_new_compiler_calls=0,audit_new_encrypted_calls=0,
 application_criterion=None,original_application_accuracy_certified=False,backend_changed=False,network_changed=False,
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
out["binding"]=q.digest(out);q.write(HERE/"audit-r1697.json",out);q.write(D/"audit-r1697.json",out)
print(json.dumps({k:v for k,v in out.items()if k!="cases"}),flush=True)
