"""Exact authorized nineteen-case v7 retest; manual sources never enter live specs."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,PAID,LIMITS,identity
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources as base_tools_sources
NAME="stage2-agent-repair-r178"
AUTH_NAME="stage2-agent-repair-authorization-r178.json"
REVIEW=ROOT/"docs/baseline/stage2-noncompiler-live-proposal-r177.json"
REVIEW_BINDING="599aa3cef40c049e618c1d835bd9aa488663972d5bd3c998eb301676ec19a553"
BUDGET=dict(maximum_generations=76,maximum_http_attempts=304,max_repairs=3,provider_retries=3,
 api_workers=4,native_workers=1,compile_jobs=2,link_jobs=1,per_shard_retained_mib=4096,
 total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200)
def tools_sources():
 return dict(base_tools_sources(),**{f:sha(ROOT/f) for f in
  ("scripts/stage2_agent_repair_report_r179.py","scripts/stage2_agent_repair_job_r178.py")})
def authorization():
 p=document(REVIEW)
 if p["binding"]!=REVIEW_BINDING:raise ValueError("Reviewed proposal changed")
 return sealed(dict(format="poseidon-agent-repair-user-instruction-r178",
  user_instruction="批准这 19 项有界重测",question_reply="call_4enF9n6s1zqUqVeje9MHJnVm/0",
  proposal=str(REVIEW),proposal_sha256=sha(REVIEW),proposal_binding=p["binding"],
  ids=[r["id"] for r in p["cases"]],paid_configuration=PAID,**BUDGET,
  no_explicit_currency_cap=True,timeout_retries_may_duplicate_billing=True,
  old_authorizations_not_reused=True,automatic_restart=False))
def build(proposal,auth_path,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from component_contract import reconstruct_request
 from deepseek_provider import public_request
 if proposal!=REVIEW or auth_path!=RESULTS/AUTH_NAME or output!=RESULTS/NAME:raise ValueError("Exact paths required")
 p=document(proposal);auth=document(auth_path)
 if p["binding"]!=REVIEW_BINDING or auth!=authorization():raise ValueError("Authorization changed")
 sources=runtime_sources()
 if sources!=p["source_hashes"]:raise ValueError("Reviewed runtime changed")
 manifestpath=Path(p["request_manifest"])
 if sha(manifestpath)!=p["request_manifest_sha256"]:raise ValueError("Prepared requests changed")
 manifest=json.loads(manifestpath.read_text());prepared={r["id"]:r for r in manifest}
 reportpath=RESULTS/"stage2-noncompiler-r177/execution/report.json";report=document(reportpath)
 if report["binding"]!=p["offline_acceptance_binding"] or report["source_hashes"]!=sources:raise ValueError("Offline binding")
 if len(report["rows"])!=19 or not report.get("completed") or any(r["result"]["status"]!="passed" for r in report["rows"]):raise ValueError("Offline qualification")
 for r in report["rows"]:
  if sha(Path(r["result"]["evidence"])/"report.json")!=r["report_sha256"]:raise ValueError("Offline evidence changed")
 before=json.loads((RESULTS/"stage2-noncompiler-r177/before.json").read_text())
 if len(before["compiler"])!=93 or not all(sha(Path(f))==h for f,h in before["compiler"].items()):raise ValueError("Compiler changed")
 unitpath=RESULTS/"stage2-noncompiler-r177/unit.json";unit=json.loads(unitpath.read_text())
 if unit["source_hashes"]!=sources or any(unit[k] for k in ("failures","errors","skipped")):raise ValueError("Unit gate")
 previous={}
 indexpath=RESULTS/"stage2-agent-campaign-v7-r130/index.json";idx=document(indexpath)
 for ref in idx["shards"]:
  sp=indexpath.parent/ref["file"]
  if sha(sp)!=ref["sha256"]:raise ValueError("Old shard changed")
  previous.update({s["id"]:s for s in document(sp)["cases"]})
 oldplanpath=RESULTS/"stage2-agent-repair-r175/plan.json"
 previous.update({s["id"]:s for r in document(oldplanpath)["shards"] for s in r["plan"]["cases"]})
 if len(p["cases"])!=19 or set(prepared)!=set(r["id"] for r in p["cases"]):raise ValueError("Scope")
 specs=[]
 for item in p["cases"]:
  s=dict(previous[item["id"]]);q=prepared[item["id"]]["request"];old=s["request"]
  if q["request_id"]!=item["request_id"] or old["request_id"]!=item["original_request_id"]:raise ValueError("Request identity")
  if reconstruct_request(q)!=q or public_request(q)!=q:raise ValueError("Public contract")
  ignore={"request_id","generation_guidance"}
  if {k:v for k,v in old.items() if k not in ignore}!={k:v for k,v in q.items() if k not in ignore}:raise ValueError("Non-guidance task change")
  args=list(s["candidate_arguments"]);args[args.index("--unified-guidance")+1]="explicit-v7"
  s.update(request=q,request_id=q["request_id"],candidate_arguments=args,
   original_request_id=old["request_id"],original_status="failed",retry_reason="r177 offline-qualified exact task",
   manual_repair_source_sent=False,new_generation=True)
  s["evaluation_identity"]=identity(s,sources);specs.append(s)
 proposals=tools_sources();shards=[]
 for i in range(4):
  cases=specs[i::4]
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=cases,source_hashes=sources,
   proposal_files=proposals,paid_configuration=PAID,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,
    maximum_generations=len(cases)*4,maximum_http_attempts=len(cases)*16),
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
   shared_scheduling=dict(api_workers=4,native_workers=1,version=1),
   authorization_binding=auth["binding"],old_claims_preserved=True,automatic_retry=False))
  if len(json.dumps(shard).encode())>8*1024**2:raise ValueError("Shard payload")
  shards.append(dict(plan=shard,output=str(RESULTS/(NAME+"-shard-%02d"%i)),audit=str(RESULTS/(NAME+"-shard-%02d-audit"%i))))
 return sealed(dict(format="poseidon-agent-repair-queue-r178",proposal=str(proposal),authorization=str(auth_path),
  proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=proposals,
  parents={str(f):sha(f) for f in (proposal,manifestpath,reportpath,oldplanpath,unitpath)},
  compiler_guard=before["compiler"],native_probe=str(unitpath),native_probe_sha256=sha(unitpath),
  native_policy="Reserve existing native slot 1 with flock; only slot 0 remains available",
  prior_seconds=0,claim_registry=str(RESULTS/"stage2-agent-repair-claims-r178"),shards=shards,planned=19,
  maximum_generations=76,maximum_http_attempts=304,max_wall_seconds=43200,max_retained_mib=32768,
  min_free_mib=4096,api_workers=4,native_workers=1,compile_jobs=2,link_jobs=1,automatic_restart=False,
  generation_guidance="explicit-v7",initial_generation="fresh; no manual/rule answer sent",
  deferred_compiler_tasks=26,source_answer_substitution=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen plan changed")
if __name__=="__main__":
 from workspace_paths import RESULTS
 a=authorization();exclusive_json(RESULTS/AUTH_NAME,a);print(a["binding"])
