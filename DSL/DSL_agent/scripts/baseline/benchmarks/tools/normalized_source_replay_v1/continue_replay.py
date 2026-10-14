"""Wait for the verified live origin and its audit, then bounded offline replay."""
import json,subprocess,time
from pathlib import Path
from replay_scope import *
STATUS=HERE/"continuation-status.json"
def status(phase,**extra):
 v=dict(format="poseidon-normalized-source-replay-continuation-v1",phase=phase,paid_calls=0,
  compiler_calls_from_watcher=0,encrypted_calls_from_watcher=0,
  watcher_sha256=sha(Path(__file__)),**extra)
 v["binding"]=digest(v)
 temp=STATUS.with_suffix(".tmp");temp.write_bytes(origin_scope.canonical(v));temp.replace(STATUS)
def main():
 require(not STATUS.exists(),"fresh_replay_continuation")
 original=read(origin_scope.RUN/"plan.json");owner=original["owner"]
 require(original["proposal_binding"]==origin_scope.verify()["binding"],"verified_origin_scope")
 deadline=time.monotonic()+900
 status("waiting_verified_origin",original_owner=owner,original_plan_binding=original["binding"],
  wait_seconds=900,scope="26 closed real generation requests followed by 25 unchanged-source offline replays")
 while time.monotonic()<deadline:
  report=read(origin_scope.RUN/"report.json")
  if supervision.same_process(owner):
   time.sleep(5);continue
  require(report["status"]=="complete"and report["calls_resolved"]is True,"origin_stopped_or_unresolved")
  if not AUDIT.exists():time.sleep(5);continue
  audit=read(AUDIT);bound(audit);require(audit["audit_status"]=="passed","origin_audit_failed")
  try:supervision.no_active()
  except ValueError as error:
   require(str(error)=="active_project_execution","unexpected_idle_guard");time.sleep(5);continue
  break
 else:raise ValueError("origin_wait_ceiling_no_restart")
 status("freezing_origin_receipts",original_owner=owner)
 with(HERE/"freeze-first.log").open("x")as log:
  r=subprocess.run(["timeout","-k","5s","90s","python3","-B",str(HERE/"freeze_replay.py")],
   stdout=log,stderr=subprocess.STDOUT,timeout=100)
 require(r.returncode==0,"replay_freeze_failed")
 p=verify();status("offline_replay_running",proposal_binding=p["binding"],planned=25,bounds=BOUNDS)
 with(HERE/"offline-first.log").open("x")as log:
  r=subprocess.run(["timeout","-k","30s","2760s","python3","-B",str(HERE/"replay.py"),"--execute"],
   stdout=log,stderr=subprocess.STDOUT,timeout=2800)
 status("offline_replay_terminal",return_code=r.returncode,proposal_binding=p["binding"])
 require((RUN/"report.json").exists(),"replay_terminal_report_missing")
 report=read(RUN/"report.json")
 if report["status"]!="complete":raise ValueError("replay_stopped_preserved")
 with(HERE/"audit-first.log").open("x")as log:
  a=subprocess.run(["timeout","-k","10s","360s","python3","-B",str(HERE/"audit.py")],
   stdout=log,stderr=subprocess.STDOUT,timeout=375)
 status("complete"if a.returncode==0 else"audit_failed",replay_return_code=r.returncode,
  audit_return_code=a.returncode,proposal_binding=p["binding"])
 require(a.returncode==0,"replay_audit_failed")
 print(json.dumps(dict(status="complete",new_agent_generations=0,http_attempts=0,report=str(RUN/"report.json"))))
if __name__=="__main__":
 require(not STATUS.exists(),"fresh_replay_continuation")
 try:main()
 except Exception as error:
  status("stopped",code=type(error).__name__,diagnostic=str(error)[:300])
  raise
