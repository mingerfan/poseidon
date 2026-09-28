"""Re-audit historical helper witnesses and map public entry points explicitly.

No source hashes are rewritten; current validators/reference code must match the
historical binding. Private dependency branches are never credited by name alone.
"""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--batch",type=Path,action="append",required=True)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 if a.output.exists():p.error("Preserve historical reports")
 from hecate_python_env import enter_nix,VENV
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 from historical_context_audit import verify_saved
 from upstream_helper_coverage import verify_trace
 from upstream_candidate_helpers import verify_events
 from unified_graph_contract import validate_candidate
 ledger_path=BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json"
 ledger=strict_file(ledger_path,4*1024**2);tasks={t["id"]:t for t in ledger["helper_directed_tasks"]}
 records={};inputs=[]
 for batch in a.batch:
  plan=strict_file(batch/"plan.json",8*1024**2);report=strict_file(batch/"report.json",16*1024**2)
  if digest(plan["sources"])!=plan["source_sha256"] or report["source_sha256"]!=plan["source_sha256"]:raise ValueError("Historical helper binding")
  if report["agent_calls"]!=0 or report["failed"]!=0:raise ValueError("Unexpected helper batch state")
  for name in ("upstream_helper_coverage.py","upstream_candidate_helpers.py","unified_graph_contract.py"):
   if sha(BASE/name)!=plan["sources"]["scripts/baseline/"+name]:raise ValueError("Historical witness validator changed")
  planned={t["id"]:t for t in plan["tasks"]}
  inputs.append(dict(batch=str(batch),plan_sha256=sha(batch/"plan.json"),report_sha256=sha(batch/"report.json"),historical_runtime_sha256=plan["source_sha256"]))
  for saved in report["records"]:
   task_id="upstream_"+saved["case_id"]
   if task_id in records or tasks.get(task_id)!=planned.get(task_id):raise ValueError("Task identity or duplicate")
   task=tasks[task_id];evidence=Path(saved["evidence"])
   original=strict_file(evidence/"report.json",8*1024**2)
   for name,h in original["source_hashes"].items():
    if plan["sources"].get("scripts/baseline/"+name)!=h:raise ValueError("Candidate source binding")
   verified=verify_saved(evidence,saved,plan["sources"])
   request=strict_file(evidence/"request.json",4*1024**2)
   candidate=(evidence/"attempt-00/candidate.py").read_text()
   checked=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=candidate),request)
   events=strict_file(evidence/"attempt-00/output/upstream-calls.json",8*1024**2)
   witness=verify_trace(checked["upstream_exercise"],events)
   if witness!=saved["helper_coverage"] or verify_events(events,request,candidate)!=saved["upstream_helper_trace"]:raise ValueError("Helper witness drift")
   if set(witness["witnesses"])!=set(task["required_helpers"]):raise ValueError("Missing helper contribution")
   if saved["model_sha256"]!=task["model_sha256"] or signature(request["model"],True)!=task["topology"]:raise ValueError("Helper context identity")
   records[task_id]=dict(task_sha256=task["task_sha256"],topology=task["topology"],model_sha256=task["model_sha256"],
      historical_runtime_sha256=plan["source_sha256"],evidence=str(evidence),report_sha256=sha(evidence/"report.json"),
      actual_frontend_checked=True,finite_return_influence_checked=True,numerical=saved["comparison"]["passed"])
 partitions=[]
 for requirement in ledger["requirements"]:
  if requirement["layer"] not in ("upstream_helper","upstream_helper_layout"):continue
  ids=requirement.get("directed_task_ids",[])
  passed=[t for t in ids if t in records]
  count=len({records[t]["topology"] for t in passed})
  state="backend_blocked" if requirement["blocker"] else ("three_contexts_verified" if count>=3 else "missing_contexts")
  partitions.append(dict(id=requirement["id"],state=state,verified_contexts=count,task_ids=passed,
                         missing_task_ids=[t for t in ids if t not in records],blocker=requirement["blocker"],
                         scope=requirement.get("acceptance_scope") or requirement.get("capability_scope") or requirement.get("scope")))
 by={r["id"]:r for r in partitions};api=[];outside={"trusted_framework_io","frontend_metaclass_and_operator_factory",
 "defined_but_unregistered_inplace","trusted_framework_introspection","frontend_expression_handle","plaintext_diagnostic_utility","mutable_maximum_hook"}
 for row in ledger["upstream_api"]:
  if sha(ROOT/row["source"])!=row["source_sha256"]:raise ValueError("Upstream API source drift")
  item=dict(source=row["source"],symbol=row["symbol"],role=row["role"],source_sha256=row["source_sha256"])
  if row["backend_blocker"]:item.update(state="backend_blocked",reason=row["backend_blocker"])
  elif row["role"] in outside:item.update(state="outside_candidate_computation",reason=row["source_review"])
  elif row["role"]=="public_fhe_helper":
   partition=by["helper."+row["symbol"]]
   item.update(state="mapped_bounded_entry_partition",partition=partition,
               reason="Only recorded adapter parameter/layout partitions; not all helper arguments or implementation branches")
  else:item.update(state="execution_partition_mapping_pending",reason=row["source_review"])
  api.append(item)
 result=dict(format="poseidon-stage2-helper-mapping-v1",ledger_sha256=sha(ledger_path),
    input_reports=inputs,records=records,helper_partitions=partitions,api=api,
    api_counts=dict(Counter(x["state"] for x in api)),verified_helper_tasks=len(records),
    verified_helper_partitions=sum(x["state"]=="three_contexts_verified" for x in partitions),
    blocked_helper_partitions=sum(x["state"]=="backend_blocked" for x in partitions),
    all_api_behaviors_accepted=False,offline_stage_complete=False,agent_stage_complete=False,
    new_encrypted_executions=0,paid_calls=0,runner_sha256=sha(Path(__file__)),
    historical_verifier_sha256=sha(Path(__file__).with_name("historical_context_audit.py")))
 result["binding"]=digest(result);a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,result)
 print(json.dumps({k:v for k,v in result.items() if k not in ("records","api","helper_partitions","input_reports")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
