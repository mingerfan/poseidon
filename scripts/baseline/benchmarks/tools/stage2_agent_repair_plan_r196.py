"""Use only SiLU0113's two unspent generations from the current user authorization."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,LIMITS
from stage2_agent_pilot_plan import PAID as ORIGINAL_PAID,sha,check_binding
from benchmark_graph import digest
from campaign_live_state import sealed
from semantic_benchmark_execution import runtime_sources
from stage2_targeted_retry_plan import document,tools_sources as base_tools_sources
from stage2_agent_repair_plan_r187 import authorization as original_authorization
NAME="stage2-agent-repair-r196"
AUTH_NAME="stage2-agent-repair-authorization-r187.json"
REVIEW=ROOT/"docs/baseline/stage2-gate-live-proposal-r186.json"
PAID=dict(ORIGINAL_PAID,max_repairs=1)
def tools_sources():
 return dict(base_tools_sources(),**{f:sha(ROOT/f) for f in (
  "scripts/stage2_agent_repair_job_r196.py","scripts/stage2_agent_repair_report_r197.py",
  "scripts/baseline/seal_keys/artifact_parameters.cpp","scripts/baseline/seal_keys/artifact_scale_probe.cpp",
  "scripts/baseline/seal_keys/CMakeLists.txt")})
def build(proposal,auth_path,output):
 from workspace_paths import RESULTS as R
 from hecate_python_env import VENV
 from seal_cpu_golden import KEY_BUILD
 from component_contract import reconstruct_request
 from deepseek_provider import public_request
 if proposal!=REVIEW or auth_path!=R/AUTH_NAME or output!=R/NAME:raise ValueError("Exact continuation paths")
 p=document(proposal);auth=document(auth_path)
 if auth!=original_authorization():raise ValueError("Original authorization changed")
 sources=runtime_sources()
 delta=sorted(k for k in set(sources)|set(p["source_hashes"]) if sources.get(k)!=p["source_hashes"].get(k))
 if delta!=["scripts/baseline/decorated_functions.py","scripts/baseline/test_native_diagnostics.py"]:raise ValueError("Unexpected runtime change")
 paths=[R/"seal-gate-r195/prepared.json",R/"seal-gate-r195/unit.json",R/"seal-gate-r195/prelive-qualification.json",
        ROOT/"docs/baseline/stage2-recovery-r192.json",ROOT/"docs/baseline/stage2-agent-repair-result-r194.json",
        R/"stage2-agent-repair-r193/report.json",R/"stage2-agent-repair-r187/job-status.json"]
 prepared,unit,proof,recovery,finished,queue,first=[json.loads(f.read_text()) for f in paths]
 for value in (proof,recovery,finished,queue):check_binding(value)
 if any(v["source_hashes"]!=sources for v in (prepared,unit,proof)):raise ValueError("Current offline source binding")
 if unit["run"]!=167 or any(unit[k] for k in ("failures","errors","skipped")):raise ValueError("Current unit acceptance")
 if len(proof["manual"])!=2 or any(r["result"]["status"]!="passed" for r in proof["manual"]):raise ValueError("Current FHE proof")
 if not queue["queue_finished"] or queue["failure"]:raise ValueError("Prior queue is not terminal")
 prior_gen=recovery["prior_generations"]+finished["totals"]["generations"]
 prior_http=recovery["prior_http_attempts"]+finished["totals"]["http_attempts"]
 old=next(r for r in recovery["rows"] if r["id"]=="free_bench_helper_0113")
 if prior_gen!=55 or prior_http!=55 or old["generations"]!=2 or old["http_attempts"]!=2 or old["status"]!="interrupted":raise ValueError("Prior usage changed")
 s=dict(next(s for s in prepared["cases"] if s["id"]==old["id"]))
 allowed=next(s for s in p["cases"] if s["id"]==old["id"])
 for key in ("model_sha256","request_id"):
  if s[key]!=allowed[key]:raise ValueError("Approved case/request changed")
 if reconstruct_request(s["request"])!=s["request"] or public_request(s["request"])!=s["request"]:raise ValueError("Public request")
 s["max_repairs"]=1;s["candidate_arguments"]=list(s["candidate_arguments"])
 s["candidate_arguments"][s["candidate_arguments"].index("--max-repairs")+1]="1"
 s["evaluation_identity"]=digest(dict(task_id=s["id"],model_sha256=s["model_sha256"],request_id=s["request_id"],
  runtime_source_sha256=digest(sources),paid_configuration=PAID))
 before=json.loads((R/"seal-gate-r181/before.json").read_text())
 if len(before["compiler"])!=93 or not all(sha(Path(f))==h for f,h in before["compiler"].items()):raise ValueError("Compiler changed")
 proposal_files=tools_sources()
 shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=[s],source_hashes=sources,
  proposal_files=proposal_files,paid_configuration=PAID,
  limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=2,maximum_http_attempts=8),
  executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
  shared_scheduling=dict(api_workers=1,native_workers=1,version=1),authorization_binding=auth["binding"],
  old_claims_preserved=True,automatic_retry=False))
 checker=KEY_BUILD/"libseal_artifact_parameters.so"
 return sealed(dict(format="poseidon-silu-unspent-quota-r196",proposal=str(proposal),authorization=str(auth_path),
  proposal_sha256=sha(proposal),authorization_sha256=sha(auth_path),source_hashes=sources,proposal_files=proposal_files,
  parents={str(f):sha(f) for f in [proposal,auth_path,*paths]},compiler_guard=before["compiler"],
  checker_guard={str(checker):sha(checker)},native_probe=str(paths[2]),native_probe_sha256=sha(paths[2]),
  prior_seconds=queue["seconds"],deadline_epoch=first["started"]+43200,
  prior_generations=prior_gen,prior_http_attempts=prior_http,
  case_generations_used=2,case_original_generation_cap=4,original_generation_cap=60,original_http_cap=240,
  in_flight_old_call_still_counted=True,manual_recovery_decision="Fresh generation within two remaining per-case slots; never replay an old HTTP identity or erase its uncertain receipt.",
  claim_registry=str(R/"stage2-agent-repair-claims-r196"),shards=[dict(plan=shard,output=str(R/(NAME+"-shard-00")),audit=str(R/(NAME+"-shard-00-audit")))],
  planned=1,maximum_generations=2,maximum_http_attempts=8,max_wall_seconds=43200,max_retained_mib=32768,
  min_free_mib=4096,api_workers=1,native_workers=1,compile_jobs=2,link_jobs=1,automatic_restart=False,
  source_change=delta,source_answer_substitution=False))
def verify(plan,output):
 check_binding(plan)
 if plan!=build(Path(plan["proposal"]),Path(plan["authorization"]),output):raise ValueError("Frozen continuation changed")
