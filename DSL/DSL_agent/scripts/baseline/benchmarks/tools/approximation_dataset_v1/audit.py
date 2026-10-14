"""Independent bound-ledger and native-array check; zero new executions."""
from pathlib import Path
import sys,shlex,json,re,math
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
from source_precision_reference import check_report
scope=q.read(HERE/"scope.json");q.bound(scope)
report=q.read(HERE/"acceptance.json");q.bound(report)
q.require(report["passed"]==3 and report["failed"]==report["skipped"]==0,"terminal_report")
for p,want in scope["source_files"].items():q.require(q.sha(p)==want,"frozen_source")
q.require(PoseidonBackend().identity_manifest()["manifest_sha256"]==scope["sdk"],"sdk")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
historical=q.read(T/"tensor_contraction_live_audit_v1/report.json");q.bound(historical)
originals={row["case"]:row for row in historical["cases"]}
OUT=q.RESULTS/"agent-approximation-dataset-acceptance-r1612";rows=[]
for case in report["cases"]:
 name=case["case"];folder=OUT/name
 h=q.read(folder/"handle.json");plan=q.read(folder/"assessment-store"/h["evaluation_id"]/"plan.json");q.bound(plan)
 state=q.read(folder/"assessment-store"/h["evaluation_id"]/"state.json");q.bound(state)
 final=q.read(folder/"result.json");q.bound(final)
 q.require(h["binding"]==plan["binding"]==state["submission_binding"]==final["submission_binding"],"plan_state_binding")
 q.require(state["status"]=="succeeded" and state["used"]==2 and
  len(state["ledger"])==2 and all(e["phase"]=="recorded"for e in state["ledger"]),"two_completed_source_attempts")
 q.require(set(state["receipts"])=={"part_0","part_1"} and len(final["evidence"]["receipts"])==2,"all_parts")
 q.require(final["encrypted_execution"] is False and final["ckks_validation_performed"] is False and
  final["acceptance"]["numerical_on_declared_dataset"] is False,"no_encrypted_regrading")
 q.require(final["package_binding"]==originals[name]["package_binding"],"historical_agent_package_binding")
 oldcase=Path(prior["campaign"])/name;task=q.read(oldcase/"task-id.json")["task_id"]
 candidates=list((oldcase/"tasks"/task).rglob("candidate.py"))
 q.require(any(q.sha(p)==originals[name]["candidate_sha256"]for p in candidates),"unmodified_original_agent_candidate")
 manifest=q.read(folder/"dataset/dataset.json");q.bound(manifest)
 inputs={}
 for key,item in manifest["inputs"].items():
  p=folder/"dataset"/item["file"]
  q.require(q.sha(p)==item["sha256"],"original_dataset_hash")
  a=np.load(p,allow_pickle=False);q.require(a.dtype.str==item["dtype"] and list(a.shape)==item["shape"],"data_precision")
  inputs[key]=a.astype(np.float64)
 all_actual=[];all_native=[]
 job=folder/"assessment-store"/h["evaluation_id"]
 for part_id in ("part_0","part_1"):
  rec=q.read(job/"receipts"/(part_id+".json"));q.bound(rec)
  entry=state["receipts"][part_id]
  q.require(q.sha(job/"receipts"/(part_id+".json"))==entry["sha256"]and rec["binding"]==entry["binding"],"committed_receipt_hash")
  request=q.read(job/"native"/part_id/"request.json")
  native=q.read(job/"native"/part_id/"native-reference.json");check_report(native,request)
  q.require(rec["source_forward_binding"]==native["binding"] and rec["source_reference_evaluated"] is True,"actual_original_reference")
  actual=np.asarray(rec["original_source_vs_approximate_comparison"]["actual"],dtype=np.float64)
  reference=np.asarray(rec["original_source_vs_approximate_comparison"]["reference"],dtype=np.float64)
  original=np.asarray([sample["out"]for sample in native["outputs"]],dtype=np.float64).reshape(len(native["outputs"]),-1)
  q.require(np.array_equal(reference,original),"receipt_reference_matches_native_arrays")
  all_actual.extend(actual);all_native.extend(original)
 actual=np.asarray(all_actual);native=np.asarray(all_native)
 expected=[]
 for i in range(17):
  left=inputs["left"][i];right=inputs["right"][i]
  if name=="matrix":
   value=[[math.fsum(float(left[r,k])*float(right[k,c])for k in range(3))for c in range(2)]for r in range(2)]
  elif name=="negative_axes":
   value=[math.fsum(float(left[r,k,j])*float(right[j,k])for j in range(2)for k in range(3))for r in range(2)]
  else:
   mean=math.fsum(float(x)for x in left)/3;value=[mean*float(x)for x in right]
  expected.append(np.asarray(value,dtype=np.float64).reshape(-1))
 expected=np.asarray(expected)
 q.require(np.all(np.abs(actual-expected)<=1e-12+1e-12*np.abs(expected)),"independent_math_reference")
 maximum=float(np.abs(actual-native).max())
 q.require(maximum==final["summary"]["maximum_absolute_error"],"dataset_maximum")
 if name=="matrix":
  label_entry=manifest["labels"]["array"];p=folder/"dataset"/label_entry["file"]
  q.require(q.sha(p)==label_entry["sha256"],"fixed_labels")
  labels=np.load(p,allow_pickle=False)
  original_correct=int(np.sum(np.argmax(native.reshape(17,2,2),axis=2)==labels))
  approximate_correct=int(np.sum(np.argmax(actual.reshape(17,2,2),axis=2)==labels))
  metric=final["summary"]["application_criterion"]
  q.require(metric["original_correct"]==original_correct and metric["approximate_correct"]==approximate_correct and
   metric["labels_compared"]==34 and metric["accuracy_drop"]==(original_correct-approximate_correct)/34,
   "exact_full_dataset_counts")
 rows.append(dict(case=name,status="passed",samples=17,parts=2,values=int(actual.size),
  maximum_absolute_error=maximum,source_attempts=state["used"],result_binding=final["binding"],
  original_agent_candidate_sha256=originals[name]["candidate_sha256"]))
def log_status(path):
 values={}
 for line in Path(path).read_text().splitlines():
  match=re.match(r"^(test_[A-Za-z0-9_]+) \(([A-Za-z0-9_.]+)\) \.\.\. (ok|FAIL|ERROR)$",line)
  if match:values[match[2]+"."+match[1]]=match[3]
 return values
initial=log_status(HERE/"controls.log");current=log_status(HERE/"guard-controls.log")
q.require(len(initial)==135 and len(current)==25,"control_count")
unaffected={k:v for k,v in initial.items()if not k.startswith(("test_approximation_dataset_assessment.","test_application_approximation_assessment."))}
q.require(len(unaffected)==113 and all(v=="ok"for v in unaffected.values()),"unaffected_compatibility")
combined={**unaffected,**current}
q.require(len(combined)==138 and all(v=="ok"for v in combined.values()),"current_control_union")
audit=dict(format="poseidon-approximation-dataset-independent-audit-v1",status="passed",cases=rows,
 passed=3,failed=0,skipped=0,samples=51,values=136,observed_native_workers=6,
 unique_current_controls=138,current_new_scope_controls=25,unaffected_compatibility_controls=113,
 original_failed_controls_retained=True,scope_binding=scope["binding"],acceptance_binding=report["binding"],
 sdk=scope["sdk"],new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0,new_native_workers=0,
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
audit["binding"]=q.digest(audit);q.write(HERE/"audit.json",audit)
print(json.dumps({k:v for k,v in audit.items()if k!="cases"}),flush=True)
