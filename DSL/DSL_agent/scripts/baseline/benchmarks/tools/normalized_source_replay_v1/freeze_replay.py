"""Freeze only closed genuine Agent origins; no provider or native execution."""
import copy,json,sys
from pathlib import Path
from replay_scope import *
def write(path,value):
 with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main():
 supervision.no_active();require(not PROPOSAL.exists()and not RUN.exists(),"replay_fresh_scope")
 origin=checked_origin();report=read(origin_scope.RUN/"report.json");plan=read(origin_scope.RUN/"plan.json")
 require(report["status"]=="complete"and report["calls_resolved"]is True and
  report["proposal_binding"]==origin["binding"]and report["binding"]==plan["binding"]and
  not supervision.same_process(plan["owner"]),"replay_origin_campaign_closed")
 audit=read(AUDIT);bound(audit)
 require(audit["audit_status"]=="passed"and audit["not_run"]==0 and audit["calls_resolved"]and
  audit["proposal_binding"]==origin["binding"],"replay_origin_campaign_audited")
 rows={r["case"]:r for r in report["rows"]};cases=aliases(origin);files={}
 for path in (origin_scope.PROPOSAL,origin_scope.RUN/"report.json",origin_scope.RUN/"plan.json",AUDIT):
  files[str(path)]=sha(path)
 from generation_receipt import verify as verify_generation
 from candidate_contract import strict_json
 for c in cases:
  old=rows[c["representative_case"]]
  if old["status"]!="passed":
   c.update(status="blocked_origin_failure",origin_failure=old.get("result",{}).get("failure"));continue
  task=origin_scope.RUN/old["case"]/"tasks"/old["task_id"]
  state=read(task/"state.json");raw=task/("response-"+str(state["generations"]-1)+".json")
  answer=strict_json(read(raw)["raw"]);receipt=task/"generation-provenance.json"
  verified=verify_generation(read(receipt),read(c["source"]["request_file"]),answer["hecate_source"],
   generation_identity_sha256=origin["source_stage_identity"]["generation"]["sha256"])
  package=Path(old["package"])
  require((package/"candidate.py").read_text()==answer["hecate_source"]and read(package/"manifest.json")["agent_generated"]is True,
   "replay_unchanged_genuine_origin")
  c.update(status="ready",origin=dict(task_directory=str(task),response_file=str(raw),
   candidate_file=str(package/"candidate.py"),receipt_file=str(receipt),receipt_binding=verified["receipt_binding"],
   generation_identity_sha256=origin["source_stage_identity"]["generation"]["sha256"],
   candidate_sha256=sha(package/"candidate.py"),agent_generated=True,manual_substitution=False))
  for path in (task/"state.json",task/"task.json",raw,receipt,package/"candidate.py",package/"manifest.json"):
   files[str(path)]=sha(path)
 auth=ROOT/"docs/baseline/current-plan-test-authorization-2026-10-11.json";authority=read(auth);bound(authority)
 require(authority["approved"]is True and authority["human_message"]=="目前执行的计划内后续所有的离线和真实测试全部批准同意",
  "replay_existing_human_authority")
 files[str(auth)]=sha(auth)
 p=dict(format="poseidon-normalized-source-replay-proposal-v1",status="frozen_plan",
  origin_proposal_binding=origin["binding"],stage_identity=origin["source_stage_identity"],
  cases=cases,bounds=BOUNDS,paid_generations=0,http_attempts=0,new_agent_generation=False,
  cache_cross_source_reuse=False,tool_hashes=tool_hashes(),evidence_hashes=files,human_authority=str(auth),
  numeric_tolerance=dict(atol=1e-5,rtol=1e-4),backend_modified=False,parameter_modified=False,network_modified=False)
 p["binding"]=digest(p);validate(p,origin);write(PROPOSAL,p);verify()
 print(json.dumps(dict(binding=p["binding"],planned=25,ready=sum(c["status"]=="ready"for c in cases),
  blocked=sum(c["status"]!="ready"for c in cases),paid_generations=0,http_attempts=0)))
if __name__=="__main__":main()
