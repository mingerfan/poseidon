"""Read committed receipts after the bounded harness stopped; never execute a backend."""
import hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
OUT=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-assessment-total-deadline-acceptance-v1")
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb")as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def require(ok,code):
 if not ok:raise ValueError(code)
def read(p,bound=False):
 v=json.loads(Path(p).read_text())
 if bound:require(v["binding"]==hashlib.sha256(canonical({k:x for k,x in v.items()if k!="binding"})).hexdigest(),"binding_changed")
 return v
def write_new(p,v):
 with Path(p).open("xb")as f:f.write(canonical(v))
def main():
 plan=read(OUT/"plan.json",True);controls=read(HERE/"controls-affected.json",True)
 baseline=read(HERE/"baseline.json");termination=read(OUT/"termination.json")
 require(termination["code"]==124 and not termination["automatic_retry"],"termination_changed")
 require(plan["controls_binding"]==controls["binding"],"controls_mismatch")
 hashes={}
 for n,h in plan["sources"].items():
  require(sha(ROOT/n)==h,"frozen_implementation_changed");hashes[str(ROOT/n)]=h
 for n,h in plan["history"].items():require(sha(n)==h,"frozen_program_or_evidence_changed");hashes[n]=h
 for n,h in baseline["protected_sources"].items():
  p=Path(n)if Path(n).is_absolute()else ROOT/n
  require(sha(p)==h,"protected_backend_or_reference_changed")
 for stage in ("generation","compilation","execution","validation"):
  for field,base in (("sources",BASE),("configuration_files",ROOT)):
   for n,h in controls["identity"][stage][field].items():
    p=(base/n).resolve();require(p.is_relative_to(ROOT),"identity_path_invalid")
    require(sha(p)==h,"stage_source_changed");hashes[str(p)]=h
 commands=[read(OUT/f"command-{i}.json",True)for i in range(7)]
 require([v["action"]for v in commands]==["submit","advance","status","advance","result","cancel","advance"],"command_sequence")
 require([v["status"]for v in commands]==["submitted","ready","ready","succeeded","succeeded","succeeded","succeeded"],"command_status")
 require(all(v["process_cleanup"]["confirmed"] is True and v["process_cleanup"]["status"]=="completed"for v in commands),"owned_worker_cleanup")
 require(all((OUT/f"command-{i}.json").stat().st_mtime<=plan["deadline_wall"]for i in range(7)),"late_command_commit")
 handle=commands[0]["handle"];state_dir=OUT/"assessment-store"/handle["evaluation_id"]
 original_plan=read(state_dir/"plan.json",True);state=read(state_dir/"state.json",True)
 require(original_plan["binding"]==handle["binding"] and state["submission_binding"]==handle["binding"],"submission_changed")
 require(state["status"]=="succeeded" and state["used"]==2 and state["failure"] is None,"terminal_state")
 require(len(state["ledger"])==2 and all(v["phase"]=="recorded" and v["outcome"]=="passed"for v in state["ledger"]),"call_ledger")
 summary=commands[3]["result"]["summary"]
 require(summary["complete"] and summary["supplied_inputs_numerically_validated"] and summary["status"]=="passed","numerical_scope")
 require(commands[1]["result"]["budget"]["used"]==1 and commands[2]["result"]["budget"]["used"]==1,"admission_changed")
 same=commands[3]["result"]["binding"]
 require(all(v["result"]["binding"]==same for v in commands[4:]),"terminal_repeat_changed")
 require(original_plan["time_budget"]["seconds"]==240 and all(v["result"]["budget"]["deadline_wall"]==original_plan["time_budget"]["deadline_wall"]for v in commands[1:]),"deadline_reset")
 receipts={}
 for n,ref in state["receipts"].items():
  p=state_dir/"receipts"/(n+".json");v=read(p,True)
  require(sha(p)==ref["sha256"] and p.stat().st_size==ref["bytes"] and v["binding"]==ref["binding"],"receipt_changed")
  require(v["encrypted_execution"] is True and v["status"]=="passed","not_actual_encrypted")
  receipts[n]=v
 require(len(receipts)==2,"receipt_scope")
 caches=list((OUT/"cache-assessment-store").glob("*/state.json"));require(len(caches)==1,"cache_scope")
 cp=caches[0];cs=read(cp,True);cplan=read(cp.with_name("plan.json"),True)
 require(cs["status"]=="succeeded"and cs["used"]==0 and cs["ledger"]==[] and cs["failure"]is None,"cache_not_completed")
 require(cp.stat().st_mtime<=plan["deadline_wall"],"late_cache_commit")
 require(cplan["package_binding"]==original_plan["package_binding"] and cplan["task_result_sha256"]==original_plan["task_result_sha256"] and
  cplan["dataset_file_binding"]==original_plan["dataset_file_binding"]and cplan["reference_identity"]==original_plan["reference_identity"],"cache_identity")
 require(cs["receipts"]==state["receipts"],"cache_receipts_changed")
 for n,ref in cs["receipts"].items():
  require(read(cp.parent/"receipts"/(n+".json"),True)==receipts[n],"cache_different_receipt")
 comparison=summary["ckks_comparison"]
 # SDK source identities and program files are checked without recreating a
 # session or replaying any admitted operation.
 evidence=[OUT/"plan.json",OUT/"termination.json",HERE/"controls-affected.json",
  state_dir/"plan.json",state_dir/"state.json",cp,cp.with_name("plan.json"),
  *[OUT/f"command-{i}.json"for i in range(7)]]
 report=dict(format="poseidon-assessment-budget-committed-evidence-audit-v1",status="passed",
  evidence_role="post-termination read-only audit; no new execution",plan_binding=plan["binding"],
  harness_status="timed_out",harness_exit_code=124,harness_deadline_seconds=300,
  command_checks=7,command_checks_passed=7,owned_process_cleanup_confirmed=7,
  cache_committed_before_harness_deadline=True,cache_added_calls=0,repeated_terminal_added_calls=0,
  committed_execution=dict(backend="poseidon-cpu",candidate_origin=plan["candidate_origin"],
   program_task=plan["program_task"],program_binding=plan["program_binding"],
   compiler_calls=0,provider_calls=0,http_attempts=0,encrypted_calls=2,input_groups=2,
   compared_values=comparison["compared_values"],maximum_absolute_error=comparison["max_absolute_error"],
   numerical_tolerance=dict(atol=1e-5,rtol=1e-4),original_application_accuracy_certified=False),
  SDK_identity_sha256=controls["identity"]["manifest_sha256"],stage_source_bytes_preserved=True,
  frozen_program_files_preserved=True,protected_sources_checked=len(baseline["protected_sources"]),
  protected_sources_changed=0,assessment_total_seconds=240,time_budget_unchanged=True,
  audit_added_compiler_calls=0,audit_added_provider_calls=0,audit_added_encrypted_calls=0,
  boundary_cancellation="controlled test trees only; no forced FHE interruption claim",
  direct_API_deadline="boundary checks; CLI advance owns hard process stop",
  source_hashes={str(Path(n).relative_to(ROOT)):h for n,h in hashes.items()if Path(n).is_relative_to(ROOT)},
  evidence_sha256={str(p):sha(p)for p in evidence})
 report["binding"]=hashlib.sha256(canonical(report)).hexdigest()
 write_new(OUT/"committed-evidence-audit.json",report)
 print(json.dumps({k:v for k,v in report.items()if k not in {"source_hashes","evidence_sha256"}},ensure_ascii=False))
 return 0
if __name__=="__main__":raise SystemExit(main())
