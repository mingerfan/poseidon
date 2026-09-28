"""One bounded retry of the three r154 provider failures, explicitly requested."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,PAID,LIMITS,identity
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources
IDS=("construct_053_1","construct_054_1","construct_127_2")
SOURCE="f6493518f4bf63c257fb7eab9f9a782a89cf5cf4a16c9006f607976f2c58b69b"
NAME="stage2-provider-retry-r155"
USER="重做3个provider的，然后对其他问题进行修复：处理构造攻陷不足和类型混淆问题，定位数值比较失败和透明密文错误，然后检查scale相关的问题"

def build(proposal,authorization,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from unified_graph_contract import validate_request,prepare
 if proposal!=RESULTS/"stage2-targeted-retry-result-r154.json" or authorization!=RESULTS/"stage2-provider-authorization-r155.json" or output!=RESULTS/NAME:raise ValueError("Exact scope paths")
 previous=document(proposal);auth=document(authorization)
 if previous["binding"]!="8703ea196463276c10f02c29212a243eb34da8d699015816d9fa7e0d8ca6a510":raise ValueError("Prior result identity")
 if auth!=sealed(dict(format="poseidon-provider-retry-authorization-r155",user_instruction=USER,ids=list(IDS),parent_sha256=sha(proposal),
  maximum_generations=12,maximum_http_attempts=48,api_workers=3,native_workers=2,max_wall_seconds=43200,no_currency_cap=True,duplicate_billing_possible=True)):raise ValueError("Authorization scope")
 sources=runtime_sources()
 if digest(sources)!=SOURCE:raise ValueError("Frozen retry source changed")
 rows={x["id"]:x for x in previous["rows"]}
 oldplan=document(RESULTS/"stage2-targeted-retry-r153/plan.json")
 specs={s["id"]:s for ref in oldplan["shards"] for s in ref["plan"]["cases"]}
 probe_path=RESULTS/"stage2-recovery-native-probe-r147/report.json";probe=document(probe_path)
 if probe["source_hashes"]!=sources or probe["passed"]!=2 or not probe["two_native_slots_observed"]:raise ValueError("Native concurrency proof")
 for f,h in probe["parents"].items():
  if sha(Path(f))!=h:raise ValueError("Native evidence changed")
 shards=[];proposals=tools_sources()
 for i,name in enumerate(IDS):
  row=rows[name]
  if row["status"]!="failed" or row["terminal_failure_layer"]!="provider":raise ValueError("Not a provider failure")
  rep=Path(row["evidence"])/"report.json"
  if sha(rep)!=row["report_sha256"]:raise ValueError("Prior report changed")
  s=dict(specs[name]);old=s["request"];validate_request(old)
  request=prepare(s["model"],old["compiler_profile_sha256"],old["compiler_configuration"],s.get("exercise"),
   construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"),
   helper_profile=s.get("helper_profile"),helper_exercise=s.get("required_helpers"),chunk_period=s.get("chunk_period"),generation_guidance="explicit-v4")
  if request!=old:raise ValueError("Request changed")
  s.update(prior_retry_evidence=row["evidence"],prior_retry_report_sha256=row["report_sha256"])
  s["evaluation_identity"]=identity(s,sources)
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=[s],source_hashes=sources,
   proposal_files=proposals,paid_configuration=PAID,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=4,maximum_http_attempts=16),
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),shared_scheduling=dict(api_workers=3,native_workers=2,version=1),
   authorization_binding=auth["binding"],old_claims_preserved=True,automatic_retry=False))
  shards.append(dict(plan=shard,output=str(RESULTS/(NAME+"-shard-%02d"%i)),audit=str(RESULTS/(NAME+"-shard-%02d-audit"%i))))
 return sealed(dict(format="poseidon-provider-retry-queue-r155",proposal=str(proposal),authorization=str(authorization),
  proposal_sha256=sha(proposal),authorization_sha256=sha(authorization),source_hashes=sources,proposal_files=proposals,
  native_probe=str(probe_path),native_probe_sha256=sha(probe_path),prior_seconds=0,claim_registry=str(RESULTS/"stage2-provider-retry-claims-r155"),
  shards=shards,planned=3,maximum_generations=12,maximum_http_attempts=48,max_wall_seconds=43200,
  max_retained_mib=32768,min_free_mib=4096,api_workers=3,native_workers=2,compile_jobs=2,link_jobs=1,
  automatic_restart=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen provider plan changed")
 return plan
if __name__=="__main__":
 from workspace_paths import RESULTS
 p=RESULTS/"stage2-targeted-retry-result-r154.json"
 a=sealed(dict(format="poseidon-provider-retry-authorization-r155",user_instruction=USER,ids=list(IDS),parent_sha256=sha(p),
  maximum_generations=12,maximum_http_attempts=48,api_workers=3,native_workers=2,max_wall_seconds=43200,no_currency_cap=True,duplicate_billing_possible=True))
 exclusive_json(RESULTS/"stage2-provider-authorization-r155.json",a)
 print(a["binding"])
