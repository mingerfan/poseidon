"""Freeze one known-outcome retest; read-only production inputs, no Nix/API."""
import copy,contextlib,importlib.util,io,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(HERE))
import scope
def write(path,value):
 with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main():
 scope.require(not scope.PROPOSAL.exists()and not scope.MANIFEST.exists(),"fresh_retest_proposal")
 parent=HERE.parent/"spatial_source_live_v1"
 spec=importlib.util.spec_from_file_location("qualified_parent_scope",parent/"scope.py")
 prior_scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_scope)
 with contextlib.redirect_stdout(io.StringIO()):prior_scope.verify()
 old=prior_scope.read(prior_scope.PROPOSAL)
 import run
 previous=prior_scope.read(prior_scope.RUN/"report.json");previous_plan=prior_scope.read(prior_scope.RUN/"plan.json")
 scope.require(previous["status"]=="complete"and previous["calls_resolved"]and
  not run.same_process(previous_plan["owner"]),"verified_previous_campaign_closed")
 case=next(c for c in old["cases"]if c["id"]==scope.CASES[0])
 directory=prior_scope.RUN/case["id"];closed=scope.read(directory/"case-report.json")
 task=directory/"tasks"/closed["task_id"];state=scope.read(task/"state.json")
 scope.require(closed["status"]=="failed"and closed["failure_layer"]=="provider"and
  state["status"]=="service_error"and prior_scope.totals([state])["calls_resolved"],"known_terminal_original_case")
 q=scope.read(case["request_file"]);source_meta=scope.read(Path(case["qualification"]["source_task_directory"])/"task.json")
 scope.require(q==source_meta["request"]and source_meta["identity"]==old["source_stage_identity"],"same_qualified_request_reference")
 proposal=copy.deepcopy(old)
 proposal.pop("binding")
 proposal.update(format="poseidon-spatial-source-retest-v2-proposal-v1",status="frozen_plan",cases=[copy.deepcopy(case)],
  provider=scope.PROVIDER,bounds=scope.BOUNDS,preparation_result="exact-current-frozen-source-reference-reused",
  preparation_seconds=0,old_paid_campaigns_closed=True,
  execution_requires_no_active_project_batch=True,retest_of_known_terminal_case=True,
  original_proposal_binding=old["binding"],original_run=str(prior_scope.RUN),
  source_qualification_scope="Same frozen source/reference and SDK from r1407; current application recaptures before the new paid call.",
  protocol_evidence=dict(local_config="generation_protocol.py",checked_date="2026-10-12",
   chat_completion="https://api-docs.deepseek.com/api/create-chat-completion/",
   documented_models=["deepseek-flash","deepseek-v4-pro"],documented_max_tokens=393216,
   requested_max_tokens=384000,reasoning_effort="high",stream=True,
   verification="Official documentation rechecked before this retest freeze"))
 paths=(prior_scope.PROPOSAL,directory/"case-report.json",task/"state.json",task/"provider-transport.json",
  ROOT/"docs/baseline/spatial-source-public-affine-diagnosis-r1412.json",
  ROOT/"docs/baseline/current-plan-test-authorization-2026-10-11.json",
  prior_scope.RUN/"report.json",prior_scope.RUN/"plan.json",
  ROOT/"docs/baseline/agent-spatial-source-live-acceptance-r1408.json")
 proposal["qualification_records"]={str(path):scope.sha(path)for path in paths}
 proposal["development_prestart_rejection"]=dict(
  previous_proposal=str(ROOT/"docs/baseline/agent-spatial-source-retest-proposal-r1413.json"),
  previous_proposal_sha256=scope.sha(ROOT/"docs/baseline/agent-spatial-source-retest-proposal-r1413.json"),
  log=str(HERE.parent/"spatial_source_continuation_v1/bilinear_retest.log"),
  log_sha256=scope.sha(HERE.parent/"spatial_source_continuation_v1/bilinear_retest.log"),
  reason="own timeout launcher misclassified before root creation",
  previous_root_absent=not(Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-spatial-source-retest-v1").exists()),
  previous_new_api_calls=0,previous_new_compiles=0,previous_new_encrypted_calls=0)
 scope.require(proposal["development_prestart_rejection"]["previous_root_absent"],"no_unknown_prior_retest")

 proposal["prior_failure_evidence"]=dict(terminal_layer="provider",source_math="six public affine coefficient mismatches",
  transport="transport_invalid_stream: stream_limit",prior_generation_reserved=state["generations"],
  prior_http_attempts=state["http_attempts"],outcomes_known=True,failed_http_usage_unknown=True,
  private_answers_or_manual_source_sent=False)
 proposal["binding"]=scope.digest(proposal);write(scope.PROPOSAL,proposal)
 source=(HERE/"scope.py").read_text().replace("FROZEN_BINDING",proposal["binding"]).replace("FROZEN_SHA",scope.sha(scope.PROPOSAL))
 (HERE/"scope.py").write_text(source)
 importlib.reload(scope)
 manifest=dict(format="poseidon-spatial-source-retest-v2-runner-v1",proposal_binding=proposal["binding"],
  proposal_sha256=scope.sha(scope.PROPOSAL),tool_hashes=scope.tools())
 manifest["binding"]=scope.digest(manifest);write(scope.MANIFEST,manifest)
 with contextlib.redirect_stdout(io.StringIO()):scope.verify()
 print(json.dumps(dict(status="frozen_not_executed",case=scope.CASES[0],proposal_binding=proposal["binding"],
  source_stage_identity=proposal["source_stage_identity"]["manifest_sha256"],new_generations=0,new_http_attempts=0,
  new_source_capture=0,new_compiles=0,new_encrypted_calls=0)))
 return 0
if __name__=="__main__":sys.exit(main())
