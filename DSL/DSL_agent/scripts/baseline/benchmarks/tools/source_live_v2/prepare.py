"""Freeze the completed current source capture as a pending live proposal."""
import pathlib,json,sys
from scope import ROOT,BASE,HERE,PROPOSAL,CASE_DATA,CASES,PROVIDER,BUDGET,read,sha,digest,require,verify
sys.path.insert(0,str(BASE))
OUT=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-source-scope-refresh-v1")
def main():
 require(not PROPOSAL.exists(),"fresh_proposal")
 report=read(OUT/"report.json");identity=read(OUT/"identity.json")
 require(report["binding"]==digest({k:v for k,v in report.items()if k!="binding"})and report["status"]=="prepared"and
         report["prepared"]==report["planned"]==36 and report["failed"]==0 and report["preparation_tasks_closed"]is True,"completed_source_preparation")
 require(report["source_stage_identity_sha256"]==identity["manifest_sha256"]and
         report["unique_generation_requests"]==36,"unique_current_generation_contexts")
 rows=[]
 for row in report["rows"]:
  require(row["id"]==CASES[len(rows)]and row["source_captured"]and row["independent_reference_verified"]and
          row["preparation_task_final_status"]=="cancelled","closed_original_source_case")
  item={k:v for k,v in row.items()if k not in ["status","source_captured","independent_reference_verified","preparation_task_final_status"]}
  item.update(baseline_task_id=row["preparation_task_id"],baseline_task_is_closed=True,new_live_task_required=True,
              original_application_accuracy_certified=False)
  rows.append(item)
 value=dict(format="poseidon-source-expansions-live-proposal-v1",revision="r1349",approved=False,executed=False,
  development_checkout=str(ROOT),branch="lhy-agent-dsl",platform="aarch64-linux",
  source_stage_identity=identity,identity_sha256=identity["manifest_sha256"],
  case_inventory_sha256=sha(CASE_DATA),preparation_origin="36 independently source-captured cases; no candidate DSL generated",
  baseline_report=str(OUT/"report.json"),baseline_report_sha256=sha(OUT/"report.json"),baseline_report_binding=report["binding"],
  cases=rows,provider=PROVIDER,budget=BUDGET,
  validation=["source_capture_and_frozen_request_match","real_agent_generation","true_compile","artifact_integrity",
              "poseidon_cpu_encrypted_numerical","package_reload","new_inputs","cache_reuse"],
  disclosure=["public mathematical model","public weights or descriptors","layout","DSL rules","controlled feedback"],
  excluded_disclosure=["rule DSL answers","test arrays","reference outputs","credentials","source provenance internals"],
  previous_unknown_http_retried=False,provider_calls=0,compiler_calls=0,encrypted_calls=0,
  pending_outside_scope=["larger graph/layout families","application accuracy","bootstrap","ciphertext handoff","CUDA hardware","whole networks"],
  provider_documentation=["https://api-docs.deepseek.com/api/create-chat-completion/","https://api-docs.deepseek.com/guides/thinking_mode/"],
  preparation_tools={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ["scope.py","prepare.py"]})
 value["binding"]=digest(value);verify(value)
 with PROPOSAL.open("x")as f:json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write("\n")
 with(HERE/"preparation-report.json").open("x")as f:json.dump(dict(status="prepared",cases=36,proposal_binding=value["binding"],proposal_sha256=sha(PROPOSAL),
  source_captures=0,compiler_calls=0,encrypted_calls=0,provider_calls=0,http_attempts=0,previous_preparation_budget_reused=False),f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(dict(status="prepared",cases=36,binding=value["binding"],max_public_bytes=max(c["public_context_bytes"]for c in rows),
  provider_calls=0,approved=False)))
if __name__=="__main__":main()
