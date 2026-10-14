"""Verify current identities and retained acceptance bytes without a new run."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(BASE),str(HERE.parent/"public_arithmetic_live_v1")]
import scope,run,json,time
def main():
    run.no_active()
    proposal=scope.verify(scope.read(scope.PROPOSAL))
    plan=scope.read(scope.RUN/"plan.json")
    scope.require(type(plan)is dict and set(plan)==run.PLAN_FIELDS,"terminal_plan_fields")
    scope.check_binding(plan)
    expected=run.public_plan(proposal)
    scope.require(all(plan[k]==v for k,v in expected.items() if k!="approved") and plan["approved"]is True,
                  "terminal_preserves_approved_plan")
    scope.require(run.finite_number(plan["created"]) and run.finite_number(plan["deadline_wall"]) and
                  plan["deadline_wall"]==plan["created"]+proposal["budget"]["batch_wall_seconds"] and
                  plan["created"]<=time.time(),"terminal_original_deadline")
    scope.require(run.safe_inventory(plan["initial_result_directories"]),"terminal_initial_inventory")
    run.approval_path(Path(plan["approval_record"]))
    owner=plan["owner"]
    scope.require(type(owner)is dict and set(owner)=={"pid","start"} and
                  all(type(v)is int and v>0 for v in owner.values()) and not run.same_process(owner),
                  "terminal_owner_has_exited")
    claim=scope.read(run.LOCK)
    scope.require(claim==dict(format="poseidon-live-owner-lock-v2",owner=owner,plan_binding=plan["binding"],
        plan_sha256=scope.sha(scope.RUN/"plan.json"),created=plan["created"],deadline_wall=plan["deadline_wall"]),
        "terminal_retained_owner_binding")
    scope.authority(scope.read(plan["approval_record"]),proposal,scope.sha(scope.PROPOSAL))
    scope.require(not run.same_process(plan["owner"]),"completed_owner_required")
    final=scope.read(ROOT/"docs/baseline/agent-public-arithmetic-live-acceptance-r1388.json")
    scope.check_binding(final)
    scope.require(final["source_stage_identity"]==proposal["source_stage_identity"] and
                  final["source_stage_counts"]==proposal["source_stage_counts"],"current_identity_match")
    raw=scope.read(scope.RUN/"report.json")
    scope.require(raw["status"]=="complete" and final["status"]=="passed" and
                  final["planned"]==final["passed"]==len(scope.CASES) and
                  final["failed"]==final["skipped"]==final["not_run"]==0,"acceptance_denominator")
    calls=scope.totals(run.states(scope.RUN))
    scope.require(calls["calls_resolved"] and calls["generations"]==final["generations"]==15 and
                  calls["http_attempts"]==final["http_attempts"]==15,"terminal_call_ledger")
    for name,expected in final["evidence_hashes"].items():
        path=Path(name)
        scope.require(path.is_file() and not path.is_symlink() and scope.sha(path)==expected,"retained_evidence_hash")
    scope.require([row["case"] for row in final["rows"]]==list(scope.CASES),"case_order")
    print(json.dumps(dict(status="complete_verified",planned=final["planned"],passed=final["passed"],
        checked_evidence_files=len(final["evidence_hashes"]),binding=final["binding"],
        new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0)))
if __name__=="__main__":main()
