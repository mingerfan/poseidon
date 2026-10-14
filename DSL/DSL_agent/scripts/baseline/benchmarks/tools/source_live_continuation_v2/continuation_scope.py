"""Seven never-started original cases, preserving both interrupted campaign ledgers."""
import copy,importlib.util,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
PREVIOUS_HERE=HERE.parent/"source_live_continuation_v1"
_spec=importlib.util.spec_from_file_location("retired_first_continuation_scope",PREVIOUS_HERE/"continuation_scope.py")
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
original=previous.original;legacy=previous.legacy;OLDHERE=previous.OLDHERE
require=original.require;canonical=original.canonical;digest=original.digest;sha=original.sha;read=original.read
request_read=original.request_read;owned_lock=original.owned_lock
PROPOSAL=original.PROPOSAL;OPTIONS=original.OPTIONS;PROFILE=original.PROFILE;WORK=original.WORK;RESULTS=original.RESULTS
OLD_RUN=previous.RUN;RUN=RESULTS/"agent-source-expansions-live-continuation-v2"
STOP=ROOT/"docs/baseline/agent-source-expansions-second-stop-r1357.json"
STOP_SHA="b4c6bcfbd3d5434c92a83bc25fc119c3e2b7e08f291a3dac4d332fd2d9ebd635"
MANIFEST=HERE/"runner-manifest.json";AUTHORIZATION=original.AUTHORIZATION;EXECUTION_APPROVAL=previous.EXECUTION_APPROVAL
TOOL_NAMES=("continuation_scope.py","run.py","test_continuation.py","controls.py")
FIRST_STARTED=("einsum_view_mutation","einsum_partition_composition","chunk_branches","chunk_projection",
 "chunk_public_weight","unbind_fusion","unbind_mutation","unbind_sequence","diagonal_output_write",
 "diagonal_source_write","diagonal_contraction","trace_scalar_factor","scalar_index_write","three_scalar_factors",
 "scalar_function_merge","scalar_shared_write","scalar_source_snapshot","scalar_repeated_stack","scalar_singleton_view",
 "scalar_public_weights","tensor_square_division","linear_fourth_power_division","scalar_shared_power_division")
STARTED=previous.STARTED+FIRST_STARTED
CASES=tuple(c for c in original.CASES if c not in STARTED)
PRIOR=dict(generations=34,http_attempts=35,calls_resolved=False)
def binding(v):
 require(v.get("binding")==digest({k:x for k,x in v.items()if k!="binding"}),"continuation_record_binding")
def tools():
 return {str((HERE/n).relative_to(ROOT)):sha(original.regular(HERE/n))for n in TOOL_NAMES}
def select_cases(cases,started):
 require(type(cases)is list and tuple(c["id"]for c in cases)==original.CASES,"original_case_inventory")
 require(type(started)in(tuple,list)and tuple(started)==STARTED,"retired_started_case_inventory")
 selected=[copy.deepcopy(c)for c in cases if c["id"]not in started]
 require(tuple(c["id"]for c in selected)==CASES and len(selected)==7,"never_started_cases_only")
 return selected
def counts(current,prior,budget):
 require(set(prior)==set(PRIOR)and type(prior["generations"])is int and type(prior["http_attempts"])is int
  and prior==PRIOR,"fixed_prior_call_reservations")
 require(set(current)==set(PRIOR)and type(current["generations"])is int and type(current["http_attempts"])is int
  and current["generations"]>=0 and current["http_attempts"]>=0 and type(current["calls_resolved"])is bool,
  "continuation_call_counts")
 a=current["generations"]+prior["generations"];b=current["http_attempts"]+prior["http_attempts"]
 require(a<=budget["max_generations"]and b<=budget["max_http_attempts"],"cumulative_original_call_ceiling")
 return dict(**current,cumulative_generations=a,cumulative_http_attempts=b,prior_generations=34,prior_http_attempts=35,
  prior_calls_resolved=False,all_campaign_calls_resolved=False)
def registered_states():
 result=[]
 for name in previous.CASES:
  folder=OLD_RUN/name;allowed=set();marker=folder/"task-id.json";cache=folder/"cache.json"
  if marker.exists():allowed.add(read(marker)["task_id"])
  if cache.exists()and read(cache).get("status")=="passed":allowed.add(read(cache)["result"]["task_id"])
  require(all(type(t)is str and len(t)==32 and all(c in "0123456789abcdef"for c in t)for t in allowed),
   "registered_task_identity")
  actual={p.parent.name:p for p in(folder/"tasks").glob("*/state.json")}
  require(set(actual)==allowed,"registered_call_state_inventory")
  for task in sorted(allowed):
   p=actual[task];require(p.is_file()and p==p.resolve()and not p.is_symlink(),"registered_call_state_path")
   state=read(p);require(state["task_id"]==task,"registered_state_identity");result.append(state)
 return result
def retired_snapshot():
 previous.verify()
 require(sha(original.regular(STOP))==STOP_SHA,"retired_audit_bytes_changed")
 audit=read(STOP);binding(audit)
 require(audit["source_scope_verified"]is True and audit["stale_report_preserved"]is True,
  "reviewed_retired_status")
 require(audit["completed_cases"]==list(FIRST_STARTED[:-1])and audit["started_cases"]==list(FIRST_STARTED)
  and audit["never_started_cases"]==list(CASES),"reviewed_retired_case_inventory")
 require(all(sha(original.regular(p))==h for p,h in audit["protected_evidence_sha256"].items()),"retired_evidence_changed")
 plan=read(OLD_RUN/"plan.json");binding(plan)
 require(plan["binding"]==audit["plan_binding"]and not legacy.same_process(plan["owner"]),"retired_owner_must_be_terminal")
 rows=[read(OLD_RUN/c/"case-report.json")for c in FIRST_STARTED[:-1]]
 require(all(r["case"]==c and r["status"]=="passed"and r["compiled"]is True and r["encrypted_execution"]is True
  and all(r[p]["status"]=="passed"for p in("new_inputs","repeat","cache"))
  for c,r in zip(FIRST_STARTED[:-1],rows)),"retired_case_status")
 require(not(OLD_RUN/FIRST_STARTED[-1]/"case-report.json").exists(),"interrupted_case_receipt_changed")
 current=original.totals(registered_states())
 require(current==dict(generations=24,http_attempts=24,calls_resolved=False),"retired_call_accounting_changed")
 previous_counts=previous.counts(current,previous.retired_snapshot()["counts"],original.BUDGET)
 require((previous_counts["cumulative_generations"],previous_counts["cumulative_http_attempts"])==(34,35),
  "both_retired_ledgers_retained")
 require(audit["observed_status"]["generations"]==24 and audit["observed_status"]["http_attempts"]==24
  and audit["overall_campaign"]==dict(planned=36,passed=27,failed=1,interrupted=1,never_started=7,
   cumulative_generations=34,cumulative_http_attempts=35,calls_resolved=False),"reviewed_retired_status")
 for name in CASES:
  require(not(OLD_RUN/name).exists()and not(original.RUN/name).exists(),"case_already_started_in_retired_run")
 return dict(plan=plan,counts=dict(PRIOR),audit_binding=audit["binding"],audit_sha256=STOP_SHA,
  evidence=audit["protected_evidence_sha256"],started=list(STARTED),remaining=list(CASES))
def prior_context():
 retired=retired_snapshot();plan=retired["plan"]
 return dict(created=plan["created"],deadline_wall=plan["deadline_wall"],initial_result_directories=plan["initial_result_directories"],
  historical_run_binding=plan["binding"],counter_offset=retired["counts"],
  parked_http_attempts=[dict(case="einsum_public_weight",index=1,status="unknown"),
   dict(case="scalar_shared_power_division",index=0,status="unknown")],
  original_budget_kept=True,continuation_reason="user-reported VM shutdown/restart; original never-started cases only")
def check_manifest(value=None):
 value=read(MANIFEST)if value is None else value
 require(type(value)is dict and set(value)=={"format","proposal_binding","proposal_sha256","tool_hashes","retired_audit_sha256","binding"},
  "continuation_manifest_fields")
 binding(value)
 require(value["format"]=="poseidon-original-live-continuation-runner-v2"and value["proposal_binding"]==original.PROPOSAL_BINDING
  and value["proposal_sha256"]==original.PROPOSAL_SHA and value["tool_hashes"]==tools()
  and value["retired_audit_sha256"]==STOP_SHA,"continuation_tool_manifest")
 return value
def projected(original_view,manifest):
 view=copy.deepcopy(original_view);view["cases"]=select_cases(original_view["cases"],STARTED)
 view["tool_hashes"]={**view["tool_hashes"],**previous.check_manifest()["tool_hashes"],**manifest["tool_hashes"]}
 view["continuation_manifest_sha256"]=sha(MANIFEST)
 return view
def verify(value=None):
 original_view=original.verify(value);retired_snapshot();manifest=check_manifest()
 require(time.time()<read(OLD_RUN/"plan.json")["deadline_wall"],"original_batch_deadline_exhausted")
 return projected(original_view,manifest)
def authority(receipt,proposal,proposal_sha):
 original_view=original.verify();original.authority(receipt,original_view,proposal_sha)
 require(receipt==read(EXECUTION_APPROVAL)and canonical(proposal)==canonical(projected(original_view,check_manifest())),
  "original_human_approval_subset")
 retired_snapshot()
def totals(states):
 return counts(original.totals(states),retired_snapshot()["counts"],original.BUDGET)
def terminal(rows,states):
 require(len(rows)<=len(CASES)and len({r["case"]for r in rows})==len(rows)and
  all(r["case"]in CASES and r["status"]in("passed","failed")for r in rows),"continuation_case_scope")
 data=totals(states);passed=sum(r["status"]=="passed"for r in rows)
 return dict(status="complete"if len(rows)==len(CASES)and data["calls_resolved"]else"stopped",
  result_scope="original_never_started_after_second_restart",passed=passed,failed=len(rows)-passed,not_run=len(CASES)-len(rows),
  historical_passed=27,historical_failed=1,historical_interrupted=1,overall_campaign_passed=27+passed,
  overall_campaign_failed=1+len(rows)-passed,overall_campaign_interrupted=1,overall_campaign_not_run=len(CASES)-len(rows),
  overall_campaign_status="unresolved_prior_http",**data)
