"""Freeze one newly authorized 80-case retry; no credentials, provider or old claim reuse."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,PAID,LIMITS,identity
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources
NAME="stage2-agent-repair-r172"
AUTH_NAME="stage2-agent-repair-authorization-r172.json"
REVIEW=ROOT/"docs/baseline/validation-adapter-acceptance-r171.json"
REVIEW_BINDING="fed62e9ed790a40506e6fe09b85368b8d2a113e267c474903fc4bd86898eb667"
USER="那么启动真实agent测试进行修复"
BUDGET=dict(maximum_generations=320,maximum_http_attempts=1280,max_repairs=3,provider_retries=3,
 api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,per_shard_retained_mib=4096,
 total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200)
def scope(review):
 rows=[r for r in review["retained_cases"] if r.get("encrypted_replay",{}).get("result",{}).get("status")!="passed"]
 if len(rows)!=80 or len({r["id"] for r in rows})!=80:raise ValueError("Exact failed cohort required")
 return sorted(rows,key=lambda r:("encrypted_replay" not in r,r["id"]))
def authorization():
 from workspace_paths import RESULTS
 review=document(REVIEW)
 if review["binding"]!=REVIEW_BINDING:raise ValueError("Reviewed failure cohort changed")
 return sealed(dict(format="poseidon-agent-repair-user-instruction-r172",user_instruction=USER,
  interpretation="One bounded fresh-generation and repair evaluation of the 80 noncompiler failures discussed immediately before this instruction; 33 compiler cases deferred. Bounds inherit existing settings and are capped for this cohort, not unlimited authorization.",
  proposal=str(REVIEW),proposal_sha256=sha(REVIEW),proposal_binding=REVIEW_BINDING,
  ids=[r["id"] for r in scope(review)],paid_configuration=PAID,**BUDGET,
  no_explicit_currency_cap=True,timeout_retries_may_duplicate_billing=True,
  old_authorizations_not_reused=True,automatic_restart=False))
def build(proposal,auth_path,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from unified_graph_contract import prepare,validate_request
 from deepseek_provider import public_request
 if proposal!=REVIEW or auth_path!=RESULTS/AUTH_NAME or output!=RESULTS/NAME:raise ValueError("Exact paths required")
 review=document(proposal);auth=document(auth_path)
 if auth!=authorization():raise ValueError("Authorization scope/budget changed")
 sources=runtime_sources()
 if sources!=review["source_hashes"]:raise ValueError("Reviewed runtime changed")
 guard=document(REVIEW)["qualification_api_regression"]["compiler_files_unchanged"]
 before=json.loads((RESULTS/"validation-adapter-r167/before.json").read_text())
 if guard!=93 or not all(sha(Path(p))==h for p,h in before["compiler"].items()):raise ValueError("Compiler changed")
 probe_path=RESULTS/"stage2-agent-repair-native-r172/report.json";probe=document(probe_path)
 if probe["source_hashes"]!=sources or probe["passed"]!=2 or not probe["two_native_slots_observed"] or probe["min_available_mib"]<1024:raise ValueError("Native concurrency proof")
 for f,h in probe["parents"].items():
  if sha(Path(f))!=h:raise ValueError("Native evidence changed")
 index_path=RESULTS/"stage2-agent-campaign-v7-r130/index.json";index=document(index_path);original={}
 for ref in index["shards"]:
  path=index_path.parent/ref["file"]
  if path.name!=ref["file"] or sha(path)!=ref["sha256"]:raise ValueError("Original plan changed")
  for row in document(path)["cases"]:original[row["id"]]=row
 parents={str(proposal):sha(proposal),str(index_path):sha(index_path)}
 prior_path=ROOT/"docs/baseline/stage2-noncompiler-repairs-r160.json"
 prior=json.loads(prior_path.read_text());old_rows={r["id"]:r for r in prior["remaining_static_cases"]}
 parents[str(prior_path)]=sha(prior_path)
 specs=[]
 for row in scope(review):
  s=dict(original[row["id"]]);old=s["request"];record=old_rows[row["id"]]
  response=Path(record["original_response"])
  if sha(response)!=record["original_response_sha256"]:raise ValueError("Original answer changed")
  previous=json.loads((response.parent.parent/"request.json").read_text())
  if digest(previous["model"])!=s["model_sha256"] or digest(s["model"])!=s["model_sha256"]:raise ValueError("Model identity changed")
  parents[str(response)]=sha(response);parents[str(response.parent.parent/"request.json")]=sha(response.parent.parent/"request.json")
  request=prepare(s["model"],old["compiler_profile_sha256"],old["compiler_configuration"],s.get("exercise"),
   construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"),
   helper_profile=s.get("helper_profile"),helper_exercise=s.get("required_helpers"),chunk_period=s.get("chunk_period"),
   generation_guidance="explicit-v5")
  validate_request(request);assert public_request(request)==request
  # Only explanatory guidance changes; mathematical task, weights, layout,
  # required constructs and compiler configuration must stay byte-identical.
  ignore={"generation_guidance","request_id"}
  if {k:v for k,v in request.items() if k not in ignore}!={k:v for k,v in previous.items() if k not in ignore}:raise ValueError("Non-guidance task change")
  args=list(s["candidate_arguments"]);i=args.index("--unified-guidance");args[i+1]="explicit-v5"
  s.update(request=request,request_id=request["request_id"],candidate_arguments=args,
   original_request_id=previous["request_id"],original_response_sha256=sha(response),
   original_source_sha256=record["source_sha256"],original_status="failed",
   retry_reason=row["next_action"],manual_repair_source_sent=False,new_generation=True)
  s["evaluation_identity"]=identity(s,sources);specs.append(s)
 proposals=tools_sources();shards=[]
 for i in range(10):
  cases=specs[i::10];n=len(cases)
  assert n==8
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=cases,source_hashes=sources,
   proposal_files=proposals,paid_configuration=PAID,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=4*n,maximum_http_attempts=16*n),
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
   shared_scheduling=dict(api_workers=10,native_workers=2,version=1),
   authorization_binding=auth["binding"],old_claims_preserved=True,automatic_retry=False))
  if len(json.dumps(shard).encode())>8*1024**2:raise ValueError("Shard size")
  shards.append(dict(plan=shard,output=str(RESULTS/(NAME+"-shard-%02d"%i)),audit=str(RESULTS/(NAME+"-shard-%02d-audit"%i))))
 return sealed(dict(format="poseidon-agent-repair-queue-r172",proposal=str(proposal),authorization=str(auth_path),
  proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=proposals,
  parents=parents,compiler_guard=before["compiler"],native_probe=str(probe_path),native_probe_sha256=sha(probe_path),
  prior_seconds=0,claim_registry=str(RESULTS/"stage2-agent-repair-claims-r172"),shards=shards,planned=80,
  maximum_generations=320,maximum_http_attempts=1280,max_wall_seconds=43200,max_retained_mib=32768,
  min_free_mib=4096,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,automatic_restart=False,
  generation_guidance="explicit-v5",initial_generation="fresh; old responses and manual DSL are not sent",
  deferred_compiler_tasks=33,source_answer_substitution=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen repair plan changed")
if __name__=="__main__":
 from workspace_paths import RESULTS
 a=authorization();exclusive_json(RESULTS/AUTH_NAME,a);print(a["binding"])
