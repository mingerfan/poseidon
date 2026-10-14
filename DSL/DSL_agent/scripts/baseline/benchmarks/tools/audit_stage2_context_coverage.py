"""Read-only distinct-topology coverage; never replaces individual evaluation outcomes."""
import argparse
from collections import Counter
from pathlib import Path
from stage2_agent_campaign import ROOT
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed

def assess(partition,states,topologies):
 if partition["agent_status"] in ("backend_blocked","static_controls_separate","current_compiler_contexts_passed","current_agent_artifact_audit_required"):
  return dict(id=partition["id"],layer=partition["layer"],scope=partition["scope"],
   status=partition["agent_status"],executable_context_denominator=False)
 contexts=partition["contexts"];required=partition["required_distinct_contexts"]
 if required!=3:raise ValueError("Explicit three-context requirement expected")
 selected={}
 for item in contexts:
  task=item["task_id"]
  if task is None or task not in states or task not in topologies:raise ValueError("Missing context task")
  if topologies[task]!=item["topology"]:raise ValueError("Context topology differs from frozen model")
  selected[task]=item["topology"]
 if len(set(selected.values()))<required:raise ValueError("Planned topology coverage insufficient")
 passed=[]
 for task in selected:
  row=states[task]
  if row["status"]=="passed":
   if row.get("audited_result",{}).get("status")!="passed":raise ValueError("Pass without independent audit")
   passed.append(task)
 distinct=sorted({selected[t] for t in passed})
 unmet=sorted(set(selected)-set(passed))
 return dict(id=partition["id"],layer=partition["layer"],scope=partition["scope"],
  executable_context_denominator=True,required_distinct_topologies=required,
  planned_distinct_topologies=len(set(selected.values())),passed_distinct_topologies=len(distinct),
  passed_topology_signatures=distinct,passed_tasks=sorted(passed),
  minimum_three_topologies_verified=len(distinct)>=required,
  all_designated_tasks_passed=not unmet,
  status="minimum_three_verified" if len(distinct)>=required else "below_three",
  unmet_tasks=[dict(id=t,topology=selected[t],status=states[t]["status"],
    failure_layer=states[t].get("terminal_failure_layer")) for t in unmet])

def audit(summary_path,index_path):
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 summary=strict_file(summary_path,32*1024**2);check_binding(summary)
 index=strict_file(index_path,16*1024**2);check_binding(index)
 if summary["campaign_binding"]!=index["binding"] or summary["runtime_source_sha256"]!=digest(runtime_sources()):
  raise ValueError("Source or campaign mismatch")
 parents={str(summary_path):sha(summary_path),str(index_path):sha(index_path)}
 # Verify retained inputs; do not read credentials or private key files.
 for name,h in summary["parents"].items():
  p=Path(name)
  if p.name==".env" or p.is_symlink() or not any(p.resolve().is_relative_to(b.resolve()) for b in (ROOT,RESULTS)):
   raise ValueError("Unsafe summary evidence")
  if sha(p)!=h:raise ValueError("Summary evidence drift")
 states={r["id"]:r for r in summary["rows"]}
 if len(states)!=2030 or len(states)!=len(summary["rows"]):raise ValueError("Task denominator")
 topologies={}
 for ref in index["shards"]:
  p=index_path.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(p)!=ref["sha256"]:raise ValueError("Shard drift")
  plan=strict_file(p,8*1024**2);check_binding(plan)
  for spec in plan["cases"]:
   if spec["id"] in topologies:raise ValueError("Duplicate frozen task")
   row=states.get(spec["id"])
   if row is None or row["request_id"]!=spec["request_id"] or row["model_sha256"]!=spec["model_sha256"]:
    raise ValueError("Summary task identity")
   topologies[spec["id"]]=signature(spec["model"],True)
 if set(states)!=set(topologies):raise ValueError("Task coverage")
 records=[assess(p,states,topologies) for p in summary["semantic_partitions"]]
 if len(records)!=401 or len({r["id"] for r in records})!=401:raise ValueError("Partition denominator")
 applicable=[r for r in records if r["executable_context_denominator"]]
 if len(applicable)!=377:raise ValueError("Executable denominator")
 result=dict(format="poseidon-distinct-context-audit-v1",records=records,parents=parents,
  runtime_source_sha256=summary["runtime_source_sha256"],source_summary_binding=summary["binding"],
  executable_partitions=377,minimum_three_topologies_verified=sum(r["minimum_three_topologies_verified"] for r in applicable),
  all_designated_tasks_passed=sum(r["all_designated_tasks_passed"] for r in applicable),
  task_statuses=summary["summary"]["statuses"],other_partition_states=dict(Counter(r["status"] for r in records if not r["executable_context_denominator"])),
  new_paid_calls=0,new_compilations=0,new_encrypted_executions=0,new_task_passes=0,stage2_complete=False,
  limitation="Derived from independently audited task evidence. The three-topology metric neither converts failed/interrupted tasks to passes nor establishes full campaign completion or formal equivalence.")
 for name,h in parents.items():
  if sha(Path(name))!=h:raise ValueError("Input changed during audit")
 return sealed(result)

if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--summary",type=Path,required=True);p.add_argument("--campaign-index",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Fresh report required")
 result=audit(a.summary,a.campaign_index);dump(a.output,result)
 print({k:v for k,v in result.items() if k not in ("records","parents")})
