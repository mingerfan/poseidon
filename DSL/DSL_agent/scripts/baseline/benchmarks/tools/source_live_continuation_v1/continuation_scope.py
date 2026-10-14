"""Continuation of never-started cases under the original human case/budget approval."""
import copy,importlib.util,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
OLDHERE=HERE.parent/"source_live_v2";sys.path.insert(0,str(OLDHERE))
import runner_scope as original
_spec=importlib.util.spec_from_file_location("retired_live_runtime",OLDHERE/"run.py")
legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(legacy)
require=original.require;canonical=original.canonical;digest=original.digest;sha=original.sha;read=original.read
request_read=original.request_read;owned_lock=original.owned_lock
PROPOSAL=original.PROPOSAL;OPTIONS=original.OPTIONS;PROFILE=original.PROFILE;WORK=original.WORK;RESULTS=original.RESULTS
OLD_RUN=original.RUN;RUN=RESULTS/"agent-source-expansions-live-continuation-v1"
STOP=ROOT/"docs/baseline/agent-source-expansions-live-stop-r1351.json"
STOP_SHA="7c79ab9fbf8c30f9b29533887f56860a279453f524ee965cef233f89565af0a2"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=original.AUTHORIZATION
EXECUTION_APPROVAL=ROOT/"docs/baseline/agent-source-expansions-execution-approval-r1349.json"
TOOL_NAMES=("continuation_scope.py","run.py","test_continuation.py","controls.py")
STARTED=("einsum_matrix","einsum_heads","einsum_contraction_broadcast","einsum_three","einsum_four","einsum_public_weight")
CASES=tuple(c for c in original.CASES if c not in STARTED)
def binding(v):
 require(v.get("binding")==digest({k:x for k,x in v.items()if k!="binding"}),"continuation_record_binding")
def tools():
 return {str((HERE/n).relative_to(ROOT)):sha(original.regular(HERE/n))for n in TOOL_NAMES}
def select_cases(cases,started):
 require(type(cases)is list and tuple(c["id"]for c in cases)==original.CASES,"original_case_inventory")
 require(type(started)in(tuple,list)and tuple(started)==STARTED,"retired_started_case_inventory")
 selected=[copy.deepcopy(c)for c in cases if c["id"]not in started]
 require(tuple(c["id"]for c in selected)==CASES and len(selected)==30,"never_started_cases_only")
 return selected
def counts(current,prior,budget):
 require(set(prior)=={"generations","http_attempts","calls_resolved"}and
  type(prior["generations"])is int and type(prior["http_attempts"])is int and
  prior==dict(generations=10,http_attempts=11,calls_resolved=False),"fixed_prior_call_reservations")
 require(set(current)==set(prior)and type(current["generations"])is int and type(current["http_attempts"])is int and
  current["generations"]>=0 and current["http_attempts"]>=0 and type(current["calls_resolved"])is bool,"continuation_call_counts")
 a=current["generations"]+prior["generations"];b=current["http_attempts"]+prior["http_attempts"]
 require(a<=budget["max_generations"]and b<=budget["max_http_attempts"],"cumulative_original_call_ceiling")
 return dict(**current,cumulative_generations=a,cumulative_http_attempts=b,
  prior_generations=prior["generations"],prior_http_attempts=prior["http_attempts"],
  prior_calls_resolved=False,all_campaign_calls_resolved=False)
def retired_snapshot():
 require(sha(original.regular(STOP))==STOP_SHA,"retired_audit_bytes_changed")
 audit=read(STOP);binding(audit)
 require(audit["observed_status"]==dict(status="stopped",passed=5,failed=1,not_run=30,generations=10,http_attempts=11,calls_resolved=False)
  and audit["source_scope_verified"]is True and audit["stale_report_preserved"]is True,"reviewed_retired_status")
 require(all(sha(original.regular(p))==h for p,h in audit["protected_evidence_sha256"].items()),"retired_evidence_changed")
 plan=read(OLD_RUN/"plan.json");binding(plan)
 require(plan["binding"]==audit["plan_binding"]and not legacy.same_process(plan["owner"]),"retired_owner_must_be_terminal")
 rows=[read(OLD_RUN/c/"case-report.json")for c in STARTED]
 require(tuple(r["case"]for r in rows)==STARTED and
  all(r["status"]=="passed"for r in rows[:5])and rows[-1]["failure_layer"]=="provider","retired_case_status")
 states=legacy.states(OLD_RUN);prior=original.totals(states)
 require(prior==dict(generations=10,http_attempts=11,calls_resolved=False),"retired_call_accounting_changed")
 for name in CASES:
  require(not (OLD_RUN/name).exists(),"case_already_started_in_original_run")
 return dict(plan=plan,counts=prior,audit_binding=audit["binding"],audit_sha256=STOP_SHA,
  evidence=audit["protected_evidence_sha256"],started=list(STARTED),remaining=list(CASES))
def prior_context():
 retired=retired_snapshot();plan=retired["plan"]
 return dict(created=plan["created"],deadline_wall=plan["deadline_wall"],
  initial_result_directories=plan["initial_result_directories"],historical_run_binding=plan["binding"],
  counter_offset=retired["counts"],parked_http_attempts=[dict(case="einsum_public_weight",index=1,status="unknown")],
  original_budget_kept=True,continuation_reason="guest-restart; only original never-started cases")
def check_manifest(value=None):
 value=read(MANIFEST)if value is None else value
 require(type(value)is dict and set(value)=={"format","proposal_binding","proposal_sha256","tool_hashes","retired_audit_sha256","binding"},"continuation_manifest_fields")
 binding(value)
 require(value["format"]=="poseidon-original-live-continuation-runner-v1"and
  value["proposal_binding"]==original.PROPOSAL_BINDING and value["proposal_sha256"]==original.PROPOSAL_SHA and
  value["tool_hashes"]==tools()and value["retired_audit_sha256"]==STOP_SHA,
  "continuation_tool_manifest")
 return value
def projected(original_view,manifest):
 view=copy.deepcopy(original_view)
 view["cases"]=select_cases(view["cases"],STARTED)
 view["tool_hashes"]={**view["tool_hashes"],**manifest["tool_hashes"]}
 view["continuation_manifest_sha256"]=sha(MANIFEST)
 return view
def verify(value=None):
 original_view=original.verify(value);retired_snapshot();manifest=check_manifest()
 require(time.time()<read(OLD_RUN/"plan.json")["deadline_wall"],"original_batch_deadline_exhausted")
 return projected(original_view,manifest)
def authority(receipt,proposal,proposal_sha):
 # The original human approval remains the only paid-call authority. No new approval is manufactured.
 original_view=original.verify()
 original.authority(receipt,original_view,proposal_sha)
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
  result_scope="original_never_started_cases",passed=passed,failed=len(rows)-passed,not_run=len(CASES)-len(rows),
  historical_passed=5,historical_failed=1,overall_campaign_passed=5+passed,
  overall_campaign_failed=1+len(rows)-passed,overall_campaign_not_run=len(CASES)-len(rows),
  overall_campaign_status="unresolved_prior_http",**data)
