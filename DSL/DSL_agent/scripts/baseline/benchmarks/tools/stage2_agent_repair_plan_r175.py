"""Exact user-approved fifty-case explicit-v6 campaign; no credentials here."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,PAID,LIMITS,identity
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources as base_tools_sources
NAME="stage2-agent-repair-r175"
AUTH_NAME="stage2-agent-repair-authorization-r175.json"
REVIEW=ROOT/"docs/baseline/stage2-remaining-live-proposal-r174.json"
REVIEW_BINDING="f0d4f7eeb594e69c4c4cfc288a5c4e8a1b189c607580c3e6280c8139bd3d73ac"
USER="批准这50项有界重测"
BUDGET=dict(maximum_generations=200,maximum_http_attempts=800,max_repairs=3,provider_retries=3,
 api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,per_shard_retained_mib=4096,
 total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200)
def tools_sources():
 return dict(base_tools_sources(),**{"scripts/stage2_agent_repair_report_r176.py":sha(ROOT/"scripts/stage2_agent_repair_report_r176.py"),"scripts/stage2_agent_repair_job_r175.py":sha(ROOT/"scripts/stage2_agent_repair_job_r175.py")})
def authorization():
 p=document(REVIEW)
 if p["binding"]!=REVIEW_BINDING:raise ValueError("Reviewed proposal changed")
 return sealed(dict(format="poseidon-agent-repair-user-instruction-r175",user_instruction=USER,
  question_reply="call_sZjXgUR1fvXyeIIhe0PWSi5T/0",proposal=str(REVIEW),proposal_sha256=sha(REVIEW),
  proposal_binding=p["binding"],ids=[r["id"] for r in p["cases"]],paid_configuration=PAID,**BUDGET,
  no_explicit_currency_cap=True,timeout_retries_may_duplicate_billing=True,
  old_authorizations_not_reused=True,automatic_restart=False))
def build(proposal,auth_path,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from unified_graph_contract import validate_request
 from component_contract import reconstruct_request
 from deepseek_provider import public_request
 if proposal!=REVIEW or auth_path!=RESULTS/AUTH_NAME or output!=RESULTS/NAME:raise ValueError("Exact paths required")
 p=document(proposal);auth=document(auth_path)
 if p["binding"]!=REVIEW_BINDING or auth!=authorization():raise ValueError("Authorization scope/budget changed")
 sources=runtime_sources()
 if sources!=p["source_hashes"]:raise ValueError("Reviewed runtime changed")
 manifest=document(Path(p["request_manifest"]))
 if manifest["binding"]!=p["request_manifest_binding"] or manifest["source_hashes"]!=sources:raise ValueError("Request manifest changed")
 reportpath=ROOT/"docs/baseline/stage2-remaining-repairs-r174.json";report=document(reportpath)
 if report["binding"]!=p["report_binding"] or report["offline_repair_passed"]!=50 or report["offline_repair_failed"]!=0:raise ValueError("Offline repair evidence")
 for f,h in report["parents"].items():
  if sha(Path(f))!=h:raise ValueError("Offline repair evidence changed")
 before=json.loads((RESULTS/"stage2-remaining-r174/before.json").read_text())
 if len(before["compiler"])!=93 or not all(sha(Path(f))==h for f,h in before["compiler"].items()):raise ValueError("Compiler changed")
 proofpath=RESULTS/"stage2-agent-repair-native-r175/report.json";proof=document(proofpath)
 if proof["source_hashes"]!=sources or proof["passed"]!=2 or not proof["two_native_slots_observed"] or proof["min_available_mib"]<1024:raise ValueError("Native concurrency proof")
 if proof["runner_sha256"]!=sha(Path(__file__).with_name("probe_repair_native_r175.py")):raise ValueError("Native probe source changed")
 for f,h in proof["parents"].items():
  if sha(Path(f))!=h:raise ValueError("Native evidence changed")
 oldplanpath=RESULTS/"stage2-agent-repair-r172/plan.json";oldplan=document(oldplanpath)
 previous={s["id"]:s for r in oldplan["shards"] for s in r["plan"]["cases"]}
 prepared={r["id"]:r for r in manifest["requests"]};specs=[]
 if len(p["cases"])!=50 or len({r["id"] for r in p["cases"]})!=50:raise ValueError("Exact fifty-case scope")
 if set(r["id"] for r in p["cases"])!=set(r["id"] for r in report["rows"]):raise ValueError("Scope not qualified")
 for item in p["cases"]:
  s=dict(previous[item["id"]]);q=prepared[item["id"]]["request"];old=s["request"]
  if q["request_id"]!=item["new_request_id"] or old["request_id"]!=item["old_request_id"]:raise ValueError("Request identity")
  if digest(q["model"])!=item["model_sha256"] or digest(s["model"])!=item["model_sha256"]:raise ValueError("Model identity")
  if reconstruct_request(q)!=q or public_request(q)!=q:raise ValueError("Public contract")
  ignore={"request_id","generation_guidance"}
  if {k:v for k,v in old.items() if k not in ignore}!={k:v for k,v in q.items() if k not in ignore}:raise ValueError("Non-guidance task change")
  args=list(s["candidate_arguments"]);i=args.index("--unified-guidance");args[i+1]="explicit-v6"
  s.update(request=q,request_id=q["request_id"],candidate_arguments=args,
   original_request_id=old["request_id"],original_status="failed",retry_reason="r174 offline-qualified typed construction repair",
   manual_repair_source_sent=False,new_generation=True)
  s["evaluation_identity"]=identity(s,sources);specs.append(s)
 proposals=tools_sources();shards=[]
 for i in range(10):
  cases=specs[i::10];assert len(cases)==5
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=cases,source_hashes=sources,
   proposal_files=proposals,paid_configuration=PAID,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=20,maximum_http_attempts=80),
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
   shared_scheduling=dict(api_workers=10,native_workers=2,version=1),
   authorization_binding=auth["binding"],old_claims_preserved=True,automatic_retry=False))
  if len(json.dumps(shard).encode())>8*1024**2:raise ValueError("Shard payload")
  shards.append(dict(plan=shard,output=str(RESULTS/(NAME+"-shard-%02d"%i)),audit=str(RESULTS/(NAME+"-shard-%02d-audit"%i))))
 parents={str(f):sha(f) for f in (proposal,Path(p["request_manifest"]),reportpath,oldplanpath)}
 return sealed(dict(format="poseidon-agent-repair-queue-r175",proposal=str(proposal),authorization=str(auth_path),
  proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=proposals,
  parents=parents,compiler_guard=before["compiler"],native_probe=str(proofpath),native_probe_sha256=sha(proofpath),
  prior_seconds=0,claim_registry=str(RESULTS/"stage2-agent-repair-claims-r175"),shards=shards,planned=50,
  maximum_generations=200,maximum_http_attempts=800,max_wall_seconds=43200,max_retained_mib=32768,
  min_free_mib=4096,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,automatic_restart=False,
  generation_guidance="explicit-v6",initial_generation="fresh; no manual/rule answer sent",
  deferred_compiler_tasks=34,source_answer_substitution=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen repair plan changed")
if __name__=="__main__":
 from workspace_paths import RESULTS
 a=authorization();exclusive_json(RESULTS/AUTH_NAME,a);print(a["binding"])
