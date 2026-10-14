"""Read-only recheck of original-source and true execution assessment receipts."""
from pathlib import Path
import sys,json,shlex,hashlib,math,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
import numpy as np
from application_backend import PoseidonBackend
from task_numerical_assessment import implementation,NATIVE_POLICY
live.no_active()
base=T/"application_assessment_current_v1";scope=q.read(base/"scope.json");q.bound(scope)
summary=q.read(base/"report.json");q.bound(summary)
q.require(summary["passed"]==3 and summary["failed"]==summary["skipped"]==0,"case_summary")
q.require(summary["scope_binding"]==scope["binding"],"scope_binding")
q.require(PoseidonBackend().identity_manifest()["manifest_sha256"]==scope["sdk"],"current_sdk")
q.require(implementation(policy=NATIVE_POLICY)==scope["reference_implementation"],"current_reference")
for p,want in scope["source_files"].items():q.require(q.sha(p)==want,"frozen_assessment_script")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
out=q.RESULTS/"agent-current-application-assessment-r1604"
q.require(q.read(out/"report.json")==summary,"duplicate_summary")
def hashes(directory):
 return {str(p.relative_to(directory)):q.sha(p) for p in directory.rglob("*") if p.is_file() and p.name!=".env"}
rows=[];total_samples=total_values=0
for name in ("matrix","negative_axes","scalar_intermediate"):
 folder=out/name
 handle=q.read(folder/"handle.json")
 q.bound(handle)
 result=q.read(folder/"result.json");q.bound(result)
 store=folder/"assessment-store"/handle["evaluation_id"]
 receipts=list((store/"receipts").glob("*.json"));q.require(len(receipts)==1,"one_receipt")
 receipt=q.read(receipts[0]);q.bound(receipt)
 q.require(result["status"]=="succeeded" and result["budget"]["used"]==1 and result["completed_parts"]==1,"task_state")
 q.require(result["evidence"]["receipts"]=={"part_0":receipt["binding"]},"receipt_binding")
 state=q.read(store/"state.json");q.bound(state)
 q.require(len(state["ledger"])==1 and state["ledger"][0]["outcome"]=="passed","single_completed_call")
 q.require(receipt["reference_identity_sha256"]==scope["reference_implementation"],"reference_binding")
 q.require(receipt["source_reference_evaluated"] is True and receipt["encrypted_execution"] is True,"actual_calls")
 q.require(receipt["agent_calls"]==receipt["compiler_calls"]==0,"no_generation_or_compilation")
 native=q.read(Path(receipt["source_reference_evidence"]));q.bound(native)
 q.require(native["status"]=="evaluated" and native["native_dtype_preserved"] is True and native["model_imported_in_host"] is False,"native_source")
 q.require(native["samples"]==2 and native["input_dtypes"]==receipt["source_input_dtypes"] and native["output_dtypes"]==receipt["source_output_dtypes"],"native_dtypes")
 q.require(native["runtime"]==dict(python=[3,10,14],numpy="1.25.2",torch="2.0.1",cuda=None,hip=None),"pinned_runtime")
 q.require(native["api_calls"]==0 and native["encrypted_execution"] is False,"native_not_encrypted")
 dataset=q.read(folder/"dataset/dataset.json");q.bound(dataset)
 arrays={}
 for key,entry in dataset["inputs"].items():
  p=folder/"dataset"/entry["file"]
  q.require(p.stat().st_size==entry["bytes"] and q.sha(p)==entry["sha256"],"dataset_hash")
  a=np.load(p,allow_pickle=False)
  q.require(list(a.shape)==entry["shape"] and a.dtype.str==entry["dtype"],"dataset_dtype")
  arrays[key]=a.astype(np.float64)
 execution_dir=Path(receipt["execution_evidence"])
 execution=q.read(execution_dir/"report.json")
 raw=q.read(execution_dir/"output/execution.json")
 q.require(execution["status"]=="executed" and execution["backend"]=="poseidon_cpu" and execution["encrypted_execution"] is True and execution["executor_secret_key"] is False,"execution_identity")
 q.require(raw["encrypted_execution"] is True and raw["executor_secret_key"] is False,"runtime_identity")
 q.require(execution["agent_calls"]==execution["compiler_calls"]==0 and execution["samples"]==2,"runtime_count")
 payload=q.read(execution_dir/"payload.json")
 q.require(payload["request"]["request_id"]==receipt["request_id"] and execution["bundle_binding"]==receipt["bundle_binding"],"execution_binding")
 decoded=np.asarray([sample["out"] for sample in execution["outputs"]],dtype=np.float64).reshape(2,-1)
 native_flat=np.asarray([sample["out"] for sample in native["outputs"]],dtype=np.float64).reshape(2,-1)
 raw_decoded=np.load(execution_dir/"output/decrypted.npy",allow_pickle=False).reshape(2,-1)
 q.require(np.array_equal(decoded,raw_decoded),"raw_decode_matches_receipt")
 expected=[]
 for i in range(2):
  left=arrays["left"][i];right=arrays["right"][i]
  if name=="matrix":
   row=[[math.fsum(float(left[r,k])*float(right[k,c]) for k in range(3)) for c in range(2)]for r in range(2)]
  elif name=="negative_axes":
   row=[math.fsum(float(left[r,k,j])*float(right[j,k]) for j in range(2) for k in range(3)) for r in range(2)]
  else:
   mean=math.fsum(float(x) for x in left)/3
   row=[mean*float(x) for x in right]
  expected.append(np.asarray(row,dtype=np.float64).reshape(-1))
 expected=np.asarray(expected)
 error=np.abs(decoded-expected);source_error=np.abs(decoded-native_flat)
 q.require(np.all(error<=1e-5+1e-4*np.abs(expected)),"frozen_ckks_threshold")
 q.require(np.all(source_error<=scope["expected_criterion"]["limit"]),"native_application_threshold")
 q.require(math.isclose(float(error.max()),receipt["ckks_comparison"]["max_absolute_error"],rel_tol=0,abs_tol=1e-16),"ckks_record_matches_independent_formula")
 q.require(float(source_error.max())==receipt["source_total_comparison"]["max_absolute_error"],"source_record_matches_arrays")
 q.require(receipt["ckks_comparison"]["atol"]==1e-5 and receipt["ckks_comparison"]["rtol"]==1e-4 and receipt["ckks_comparison"]["passed"] is True,"immutable_tolerance")
 task=receipt["task_id"];old=Path(prior["campaign"])/name/"tasks"/task
 q.require(hashes(old)==hashes(folder/"tasks"/task),"unchanged_historical_agent_program")
 q.require(result["acceptance"]["formal_equivalence"] is False and result["acceptance"]["original_source_accuracy_certified"] is False,"finite_evidence_boundary")
 total_samples+=2;total_values+=decoded.size
 rows.append(dict(case=name,status="passed",samples=2,values=int(decoded.size),native_dtypes=receipt["source_input_dtypes"],ckks_maximum_absolute_error=float(error.max()),original_source_maximum_absolute_error=float(source_error.max()),receipt_binding=receipt["binding"],execution_report_sha256=q.sha(execution_dir/"report.json"),original_source_report_sha256=q.sha(Path(receipt["source_reference_evidence"])),historical_agent_files_unchanged=True))
report=dict(format="poseidon-current-original-source-assessment-audit-v1",status="passed",cases=rows,passed=3,failed=0,skipped=0,samples=total_samples,values=total_values,acceptance_binding=summary["binding"],sdk=scope["sdk"],reference_identity=scope["reference_implementation"],new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0,new_reference_calls=0,observed_new_encrypted_calls=3,observed_native_reference_calls=3,original_accuracy_universally_certified=False,formal_equivalence=False,source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
report["binding"]=q.digest(report);q.write(HERE/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
