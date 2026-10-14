"""Inspect frozen capacity scope and settled artifacts, without executing requests."""
import json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2];SCOPE=BASE/"benchmarks/tools/generic_capacity_backend_v10"
sys.path[:0]=[str(BASE),str(SCOPE)]
import qualification_scope as s
def snapshot():
 plan=s.verify("live");root=Path(plan["campaign"])
 report=s.read(root/"report.json");s.bound(report)
 states=s.states(root);calls=s.calls(states,plan["budget"]["max_generations"],plan["budget"]["max_http"])
 terminal=(root/"terminal.json").exists()
 rows=[];compiler=encrypted=0
 for case in plan["cases"]:
  folder=root/case["id"];task_ids=list(folder.glob("tasks/*/state.json"))
  record=dict(case=case["id"],status="not_run",compiled=0,encrypted=0,tasks=[],
              source_kind=case["execution_source_kind"],source_intake_failure=case["source_intake_failure"])
  primary=(folder/"task-id.json")
  primary_id=s.read(primary)["task_id"]if primary.exists()else None
  for sf in sorted(task_ids):
   state=s.read(sf);is_primary=sf.parent.name==primary_id
   record["tasks"].append(dict(task_id=sf.parent.name,status=state["status"],phase=state["phase"],
    generations=state["generations"],http_attempts=state["http_attempts"],primary=is_primary))
   if is_primary:
    record.update(status=state["status"],phase=state["phase"],generations=state["generations"],
     http_attempts=state["http_attempts"],failure=state.get("failure"))
   validations=[]
   for vf in sorted(sf.parent.glob("validation-*.json")):
    value=s.read(vf);observed=value.get("observed_stages",{})
    compiled=observed.get("compiled")is True;executed=value.get("encrypted_execution")is True
    record["compiled"]+=compiled;record["encrypted"]+=executed
    validations.append(dict(file=str(vf),sha256=s.sha(vf),status=value["status"],
      compiled=compiled,encrypted_execution=executed,failure=value.get("failure"),
      evidence=value.get("evidence")))
   record.setdefault("validations",[]).extend(validations)
   transport=sf.parent/"provider-transport.json"
   metrics=sf.parent/"provider-metrics.json"
   if is_primary and transport.exists():
    receipt=s.read(transport);s.bound(receipt);record["transport"]=receipt
   if is_primary and metrics.exists():
    record["usage"]=[dict(index=c.get("index"),status=c.get("status"),usage=c.get("usage"))
                     for c in s.read(metrics).get("calls",[])]
  compiler+=record["compiled"];encrypted+=record["encrypted"];rows.append(record)
 return dict(format="poseidon-generic-capacity-readonly-snapshot-v1",terminal_present=terminal,
  sdk=plan["sdk"],plan_binding=plan["binding"],report_binding=report["binding"],
  batch_status=report["status"],**calls,compiler_confirmed_from_validation=compiler,
  encrypted_confirmed_from_validation=encrypted,cases=rows,
  acceptance_claim="snapshot only; not package or numerical re-audit",
  writes_to_original_evidence=False,new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0)
if __name__=="__main__":
 result=snapshot()
 print(json.dumps(result,ensure_ascii=False,sort_keys=True))
