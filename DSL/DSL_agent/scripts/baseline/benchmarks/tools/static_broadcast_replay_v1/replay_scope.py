"""Bounded replay scope. Current packages keep their own source identity."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
NORMAL=HERE.parent/"static_broadcast_live_v1"
sys.path.insert(0,str(BASE));sys.path.insert(0,str(NORMAL))
import scope as origin_scope
import run as supervision
from planning import aliases,digest,require
read=origin_scope.read;sha=origin_scope.sha
WORK=supervision.WORK;RESULTS=supervision.RESULTS
RUN=RESULTS/"agent-static-broadcast-replay-v1"
PROPOSAL=ROOT/"docs/baseline/agent-static-broadcast-replay-proposal-r1430.json"
AUDIT=ROOT/"docs/baseline/agent-static-broadcast-live-acceptance-r1429.json"
BOUNDS=dict(batch_wall_seconds=900,case_wall_seconds=180,max_compiler_calls=3,max_encrypted_calls=9,
 max_scripted_attempts=3,paid_generations=0,http_attempts=0,new_result_bytes=64*1024**2,
 global_result_bytes=32*1024**3,minimum_free_bytes=4*1024**3,compile_jobs=2,link_jobs=1,
 native_concurrency=1,native_address_bytes=4*1024**3,native_cpu_seconds=150)
TOOL_NAMES=("replay_scope.py","freeze_replay.py","replay.py","planning.py","exact_source.py","test_controls.py")
def tool_hashes():
 return {str(HERE/n):sha(HERE/n)for n in TOOL_NAMES}
def bound(value):
 require(type(value)is dict and value.get("binding")==digest({k:v for k,v in value.items()if k!="binding"}),"replay_self_binding")
def validate(p,origin):
 bound(p)
 require(p["format"]=="poseidon-static-broadcast-replay-proposal-v1"and p["status"]=="frozen_plan",
  "replay_proposal_format")
 require(p["origin_proposal_binding"]==origin["binding"]and p["bounds"]==BOUNDS and
  all(type(x)is int for x in p["bounds"].values()),"replay_proposal_scope")
 require(p["stage_identity"]==origin["source_stage_identity"]and p["paid_generations"]==p["http_attempts"]==0,
  "replay_unpaid_identity")
 require(p["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4),"replay_frozen_numeric_tolerance")
 wanted=aliases(origin)
 require([c["id"]for c in p["cases"]]==[c["id"]for c in wanted],"replay_complete_source_list")
 for c,w in zip(p["cases"],wanted):
  for key in ("id","context","representative_case","request_id","model_sha256","mathematical_request_sha256","source"):
   require(c[key]==w[key],"replay_exact_source_case")
  require(c["status"]in("ready","blocked_origin_failure"),"replay_declared_origin_status")
  if c["status"]=="ready":
   require(c["origin"]["agent_generated"]is True and c["origin"]["manual_substitution"]is False,
    "replay_genuine_source")
 require(p["new_agent_generation"]is False and p["cache_cross_source_reuse"]is False,"replay_honest_boundary")
 return p
def checked_origin():
 original=read(origin_scope.regular(origin_scope.PROPOSAL))
 origin_scope.verify(original)
 bound(original)
 return original
def verify():
 origin=checked_origin();p=read(PROPOSAL);validate(p,origin)
 require(p["tool_hashes"]==tool_hashes(),"replay_tool_changes")
 for path,want in p["evidence_hashes"].items():
  path=origin_scope.regular(path);require(sha(path)==want,"replay_frozen_evidence_changed")
 return p
def terminal(rows,cases):
 require(len({r["id"]for r in rows})==len(rows)and set(r["id"]for r in rows)<={c["id"]for c in cases},
  "replay_terminal_row_scope")
 require(all(r["status"]in("passed","failed","blocked_origin_failure")for r in rows),"replay_terminal_status")
 counts={key:sum(r["status"]==key for r in rows)for key in("passed","failed","blocked_origin_failure")}
 counts.update(planned=len(cases),not_run=len(cases)-len(rows),paid_generations=0,http_attempts=0)
 counts["benchmark_status"]="completed_with_failures"if counts["failed"]or counts["blocked_origin_failure"]else(
  "passed"if counts["not_run"]==0 else"stopped")
 return counts
