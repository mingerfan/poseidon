"""User-authorized remaining-28 retest after SEAL gate correction, with finite budgets."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,LIMITS
from stage2_agent_pilot_plan import PAID as ORIGINAL_PAID,sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources as base_tools_sources
NAME="stage2-agent-repair-r189"
AUTH_NAME="stage2-agent-repair-authorization-r187.json"
REVIEW=ROOT/"docs/baseline/stage2-gate-live-proposal-r186.json"
PAID=dict(ORIGINAL_PAID)
BUDGET=dict(maximum_generations=60,maximum_http_attempts=240,api_workers=4,native_workers=1,
 compile_jobs=2,link_jobs=1,per_shard_retained_mib=4096,total_retained_mib=32768,min_free_mib=4096,max_wall_seconds=43200)
def tools_sources():
 return dict(base_tools_sources(),**{f:sha(ROOT/f) for f in
  ("scripts/stage2_agent_repair_report_r190.py","scripts/stage2_agent_repair_job_r189.py",
   "scripts/baseline/seal_keys/artifact_parameters.cpp","scripts/baseline/seal_keys/artifact_scale_probe.cpp",
   "scripts/baseline/seal_keys/CMakeLists.txt")})
def authorization():
 p=document(REVIEW)
 return sealed(dict(format="poseidon-agent-repair-user-instruction-r187",
  user_instruction="上述修正完以后，继续解决剩余的所有失败项，并允许启动真实agent测试",
  authorization_scope="Current explicit user instruction; finite subset of earlier operational limits, not an old authorization replay.",
  proposal=str(REVIEW),proposal_sha256=sha(REVIEW),proposal_binding=p["binding"],
  ids=[r["id"] for r in p["cases"]],paid_configuration=PAID,**BUDGET,
  no_explicit_currency_cap=True,timeout_retries_may_duplicate_billing=True,
  old_authorizations_not_reused=True,automatic_restart=False))
def build(proposal,auth_path,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from seal_cpu_golden import KEY_BUILD
 from component_contract import reconstruct_request
 from deepseek_provider import public_request
 if proposal!=REVIEW or auth_path!=RESULTS/AUTH_NAME or output!=RESULTS/NAME:raise ValueError("Exact paths required")
 p=document(proposal);auth=document(auth_path)
 if auth!=authorization():raise ValueError("Authorization changed")
 sources=runtime_sources()
 if sources!=p["source_hashes"]:raise ValueError("Reviewed runtime changed")
 manifestpath=RESULTS/"seal-gate-r186/prepared.json"
 if sha(manifestpath)!=p["prepared_sha256"]:raise ValueError("Requests changed")
 manifest=json.loads(manifestpath.read_text())
 if manifest["source_hashes"]!=sources:raise ValueError("Prepared source drift")
 cases=manifest["cases"]
 tracker=document(ROOT/"docs/baseline/stage2-noncompiler-closure-r180.json")
 ids={r["id"] for r in tracker["rows"] if r["agent_status"]!="passed"}
 if len(cases)!=28 or {s["id"] for s in cases}!=ids:raise ValueError("Remaining task scope changed")
 unitpath=RESULTS/"seal-gate-r186/unit.json";unit=json.loads(unitpath.read_text())
 if unit["source_hashes"]!=sources or any(unit[k] for k in ("failures","errors","skipped")):raise ValueError("Unit gate")
 proofpath=RESULTS/"seal-gate-r186/prelive-qualification.json";proof=document(proofpath)
 if proof["source_hashes"]!=sources or len(proof["manual"])!=2 or any(r["result"]["status"]!="passed" for r in proof["manual"]):raise ValueError("Manual feasibility gate")
 before=json.loads((RESULTS/"seal-gate-r181/before.json").read_text())
 if len(before["compiler"])!=93 or not all(sha(Path(f))==h for f,h in before["compiler"].items()):raise ValueError("Compiler changed")
 for s in cases:
  if reconstruct_request(s["request"])!=s["request"] or public_request(s["request"])!=s["request"]:raise ValueError("Public request mismatch")
  if s["manual_repair_source_sent"] is not False:raise ValueError("Source-answer substitution")
 groups=[[s for s in cases if s["max_repairs"]==3]]
 rest=[s for s in cases if s["max_repairs"]==1]
 groups += [rest[i::3] for i in range(3)]
 if [len(g) for g in groups]!=[2,9,9,8]:raise ValueError("Shard scope")
 recovery_path=ROOT/"docs/baseline/stage2-precall-recovery-r189.json"
 recovery=document(recovery_path)
 if recovery["generations"] or recovery["http_attempts"] or recovery["uncertain_calls"]:raise ValueError("Unreconciled prior calls")
 if not all(sha(Path(f))==h for f,h in recovery["parents"].items()):raise ValueError("Prior evidence drift")
 proposals=tools_sources();shards=[]
 for i,group in enumerate(groups):
  paid=dict(PAID,max_repairs=3 if i==0 else 1);specs=[]
  for source in group:
   s=dict(source);q=s["request"]
   s["evaluation_identity"]=digest(dict(task_id=s["id"],model_sha256=s["model_sha256"],request_id=s["request_id"],
      runtime_source_sha256=digest(sources),paid_configuration=paid))
   if s["candidate_arguments"][s["candidate_arguments"].index("--max-repairs")+1]!=str(paid["max_repairs"]):raise ValueError("Repair cap mismatch")
   specs.append(s)
  maximum=len(specs)*(paid["max_repairs"]+1)
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=specs,source_hashes=sources,
   proposal_files=proposals,paid_configuration=paid,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=maximum,maximum_http_attempts=maximum*4),
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
   shared_scheduling=dict(api_workers=4,native_workers=1,version=1),
   authorization_binding=auth["binding"],old_claims_preserved=True,automatic_retry=False))
  if len(json.dumps(shard).encode())>8*1024**2:raise ValueError("Shard payload")
  shards.append(dict(plan=shard,output=str(RESULTS/(NAME+"-shard-%02d"%i)),audit=str(RESULTS/(NAME+"-shard-%02d-audit"%i))))
 checker=KEY_BUILD/"libseal_artifact_parameters.so"
 return sealed(dict(format="poseidon-agent-repair-queue-r189",proposal=str(proposal),authorization=str(auth_path),
  proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=proposals,
  parents={str(f):sha(f) for f in (proposal,manifestpath,proofpath,unitpath,recovery_path)},
  compiler_guard=before["compiler"],checker_guard={str(checker):sha(checker)},
  native_probe=str(proofpath),native_probe_sha256=sha(proofpath),
  native_policy="Reserve existing native slot 1; only slot 0 available",
  prior_seconds=recovery["prior_seconds"],claim_registry=str(RESULTS/"stage2-agent-repair-claims-r189"),shards=shards,planned=28,
  maximum_generations=60,maximum_http_attempts=240,max_wall_seconds=43200,max_retained_mib=32768,
  min_free_mib=4096,api_workers=4,native_workers=1,compile_jobs=2,link_jobs=1,automatic_restart=False,
  generation_guidance="explicit-v8",initial_generation="fresh; no manual/rule answer sent",
  deferred_compiler_tasks=0,source_answer_substitution=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen plan changed")
