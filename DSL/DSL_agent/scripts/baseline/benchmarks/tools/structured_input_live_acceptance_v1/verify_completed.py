"""Verify closed campaign records; no API, compilation or encrypted execution."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"structured_input_live_v1"))
import scope,run
data=scope.read(scope.ROOT/"docs/baseline/agent-structured-input-live-acceptance-r1393.json")
scope.check_binding(data)
proposal=scope.verify()
plan=scope.read(scope.RUN/"plan.json")
run.no_active()
scope.require(not run.same_process(plan["owner"]),"completed_owner_still_live")
scope.authority(scope.read(plan["approval_record"]),proposal,scope.sha(scope.PROPOSAL))
scope.require(data["proposal_binding"]==proposal["binding"] and data["plan_binding"]==plan["binding"],"accepted_campaign_binding")
scope.require(data["source_stage_identity"]==proposal["source_stage_identity"],"accepted_stage_identity")
scope.require(data["status"]=="passed" and data["planned"]==data["passed"]==12 and
              data["failed"]==data["skipped"]==data["not_run"]==0,"accepted_scope")
calls=scope.totals(run.states(scope.RUN))
scope.require(calls["calls_resolved"] and (calls["generations"],calls["http_attempts"])==
              (data["generations"],data["http_attempts"]),"closed_call_ledger")
scope.require(data["compiled_candidate_attempts"]<=scope.BOUNDS["max_compiler_calls"] and
              data["encrypted_execution_calls"]<=scope.BOUNDS["max_encrypted_calls"],"closed_native_budget")
scope.require(data["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"fixed_tolerance")
for path,want in data["evidence_hashes"].items():
    scope.require(scope.sha(scope.regular(path))==want,"retained_evidence_changed")
print(json.dumps(dict(status="passed",evidence_files=len(data["evidence_hashes"]),binding=data["binding"],
 source_and_dependencies_current=True,new_api_calls=0,new_compilations=0,new_encrypted_calls=0)))
