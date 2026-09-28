"""Recover three omitted frozen helper tasks from their original encrypted evidence."""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 if a.output.exists():p.error("Preserve evidence")
 from hecate_python_env import enter_nix,VENV
 if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=180)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned Python required")
 from workspace_paths import RESULTS
 from historical_context_audit import verify_saved
 from unified_graph_contract import validate_candidate
 from upstream_candidate_helpers import verify_events
 from upstream_helper_coverage import verify_trace
 from semantic_benchmark_execution import runtime_sources
 from seal_artifact_gate import inspect_artifacts
 from compiler_configuration import verify_artifact_configuration
 from candidate_contract import request_input_names,request_rotations
 from cipher_abi import artifact_options
 from seal_cpu_golden import PROFILE
 folder=RESULTS/"upstream-virtual-r26-verified-helper-directed"
 plan=strict_file(folder/"plan.json",8*1024**2);report=strict_file(folder/"report.json",16*1024**2)
 if digest(plan["sources"])!=plan["source_sha256"] or report["source_sha256"]!=plan["source_sha256"]:raise ValueError("Historical source binding")
 ledger_path=BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json"
 ledger=strict_file(ledger_path,4*1024**2);tasks={t["id"]:t for t in ledger["helper_directed_tasks"]}
 selected={"upstream_bench_helper_0008","upstream_bench_helper_0072","upstream_bench_helper_0080"}
 records=[];sources=runtime_sources()
 for saved in report["records"]:
  tid="upstream_"+saved["case_id"]
  if tid not in selected:continue
  task=tasks[tid]
  if next(t for t in plan["tasks"] if t["id"]==tid)!=task:raise ValueError("Frozen task changed")
  evidence=Path(saved["evidence"]);original=strict_file(evidence/"report.json",8*1024**2)
  if any(plan["sources"].get("scripts/baseline/"+n)!=h for n,h in original["source_hashes"].items()):raise ValueError("Original source identity")
  verify_saved(evidence,saved,plan["sources"])
  request=strict_file(evidence/"request.json",4*1024**2);source=(evidence/"attempt-00/candidate.py").read_text()
  checked=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
  out=evidence/"attempt-00/output";events=strict_file(out/"upstream-calls.json",8*1024**2)
  contribution=verify_trace(checked["upstream_exercise"],events)
  if contribution!=saved["helper_coverage"] or verify_events(events,request,source)!=saved["upstream_helper_trace"]:raise ValueError("Trace/contribution drift")
  if set(contribution["witnesses"])!=set(task["required_helpers"]) or saved["model_sha256"]!=task["model_sha256"] or signature(request["model"],True)!=task["topology"]:raise ValueError("Helper/task identity")
  gate=inspect_artifacts((out/"lowered._hecate_golden.hevm").read_bytes(),(out/"_hecate_golden.cst").read_bytes(),rotation_steps=request_rotations(request),expected_inputs=len(request_input_names(request)),**artifact_options(request["layout"]))
  verify_artifact_configuration(request,gate,sha(PROFILE))
  if gate!=original["attempts"][0]["artifact_gate"]:raise ValueError("Artifact gate drift")
  records.append(dict(task_id=tid,task_sha256=task["task_sha256"],model_sha256=task["model_sha256"],
    topology=task["topology"],required_helpers=task["required_helpers"],evidence=str(evidence),
    report_sha256=sha(evidence/"report.json"),files=saved["files"],comparison=saved["comparison"],
    historical_runtime_sha256=plan["source_sha256"],current_contract_accepted=True,
    independently_rechecked_reference=True,actual_frontend_and_return_contribution_verified=True,
    historical_ciphertext_execution=True,new_ciphertext_execution=False))
 if {r["task_id"] for r in records}!=selected or sources!=runtime_sources():raise ValueError("Recovery incomplete or source drift")
 result=dict(format="poseidon-recovered-helper-contexts-v1",rows=records,paid_calls=0,
   original_plan_sha256=sha(folder/"plan.json"),original_report_sha256=sha(folder/"report.json"),
   original_batch=str(folder),ledger_sha256=sha(ledger_path),current_runtime_sha256=digest(sources),
   recovered_tasks=3,new_ciphertext_executions=0,original_frozen_task_count=60,
   runner_sha256=sha(Path(__file__)),historical_reference_verifier_sha256=sha(Path(__file__).with_name("historical_context_audit.py")))
 result["binding"]=digest(result);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k!="rows"}));return 0
if __name__=="__main__":raise SystemExit(main())
