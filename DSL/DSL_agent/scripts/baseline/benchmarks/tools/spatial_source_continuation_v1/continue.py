"""Finite approved continuation; never restarts the original campaign."""
import argparse,contextlib,fcntl,io,json,pathlib,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(HERE.parent/"spatial_source_live_v1"))
import scope,run
PLAN=HERE/"plan.json";STATUS=HERE/"status.json";EVENTS=HERE/"events.jsonl";OWNED=False
def require(value,code):
 if not value:raise ValueError(code)
def read(path):return scope.read(path)
def verified():
 plan=read(PLAN);scope.check_binding(plan)
 require(plan["source_root"]==str(ROOT)and plan["stages"]==
  ["wait_for_verified_main_terminal","audit_original_34_cases","fresh_bilinear_retest","audit_retest"],"continuation_scope")
 require(plan["maximum_wait_seconds"]==3600 and plan["retest_seconds"]==14460 and plan["audit_seconds"]==330,"continuation_bounds")
 for path,want in plan["frozen_files"].items():
  require(scope.sha(scope.regular(path))==want,"continuation_frozen_file")
 authority=read(plan["human_authorization"]);scope.check_binding(authority)
 require(authority["approved"]is True and authority["human_source"]=="direct_user_message"and
  authority["binding"]==plan["human_authorization_binding"],"continuation_human_authority")
 with contextlib.redirect_stdout(io.StringIO()):scope.verify()
 main=read(pathlib.Path(plan["main_root"])/"plan.json")
 require(main["binding"]==plan["main_plan_binding"]and main["owner"]==plan["main_owner"]and
  main["proposal_binding"]==plan["main_proposal_binding"],"continuation_original_owner")
 return plan
def status(stage,**fields):
 data=dict(stage=stage,time=time.time(),owner=run.process_identity(__import__("os").getpid()),**fields)
 with EVENTS.open("a")as f:f.write(json.dumps(data,sort_keys=True,allow_nan=False)+"\n")
 data["binding"]=scope.digest(data)
 tmp=HERE/"status-writing.json";tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n");tmp.replace(STATUS)
 return data
def idle():
 code="import sys;sys.path.insert(0,"+repr(str(HERE.parent/"spatial_source_retest_v1"))+");import run;run.no_active()"
 result=subprocess.run(["python3","-B","-c",code],capture_output=True,text=True,timeout=10)
 require(result.returncode==0,"continuation_active_project_execution")
def execute(command,label,seconds):
 log=HERE/(label+".log");require(not log.exists(),"preserve_continuation_execution_log")
 status(label,command=command,deadline_seconds=seconds)
 with log.open("x")as f:
  result=subprocess.run(["timeout","-k","10s",str(seconds)+"s",*command],stdout=f,stderr=subprocess.STDOUT,timeout=seconds+20)
 status(label+"_terminal",exit_code=result.returncode,log=str(log),log_sha256=scope.sha(log))
 return result.returncode
def audit(plan,campaign,path):
 code=execute(["python3","-B",str(HERE.parent/"spatial_source_live_acceptance_v1/audit.py"),
  "--campaign",campaign],campaign+"_audit",plan["audit_seconds"])
 require(code==0,"continuation_audit_failed")
 value=read(path);scope.check_binding(value)
 require(value["audit_status"]=="passed"and value["calls_resolved"]is True,"continuation_audit_integrity")
 for observed,want in value["evidence_hashes"].items():
  require(scope.sha(scope.regular(observed))==want,"continuation_audit_evidence")
 return value
def watch():
 global OWNED
 with (HERE/"owner.lock").open("a+")as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(not STATUS.exists(),"continuation_single_launch")
  OWNED=True
  plan=verified();started=time.monotonic();status("waiting_for_primary",plan_binding=plan["binding"])
  while run.same_process(plan["main_owner"]):
   require(time.monotonic()-started<plan["maximum_wait_seconds"],"continuation_wait_boundary")
   verified();time.sleep(15)
  verified()
  main=read(pathlib.Path(plan["main_root"])/"report.json")
  require(main["binding"]==plan["main_plan_binding"]and main["status"]=="complete"and
   main["planned"]==34 and main["passed"]+main["failed"]==34 and main["not_run"]==0 and
   main["calls_resolved"]is True,"continuation_main_terminal")
  require(scope.totals(run.states(scope.RUN))["calls_resolved"],"continuation_ledger_terminal")
  idle();status("primary_terminal",passed=main["passed"],failed=main["failed"],not_run=0)
  first=audit(plan,"spatial",plan["original_acceptance"])
  require(first["proposal_binding"]==plan["main_proposal_binding"],"continuation_primary_audit_identity")
  verified();idle()
  require(not pathlib.Path(plan["retest_root"]).exists(),"continuation_fresh_retest_root")
  code=execute(["python3","-B",str(HERE.parent/"spatial_source_retest_v1/run.py"),
   "--execute",plan["retest_root"],"--approval-record",plan["retest_approval"]],"bilinear_retest",plan["retest_seconds"])
  require(code in(0,1),"continuation_retest_exit")
  report=read(pathlib.Path(plan["retest_root"])/"report.json")
  require(report["status"]=="complete"and report["planned"]==1 and report["not_run"]==0 and
   report["calls_resolved"]is True,"continuation_retest_terminal")
  idle();last=audit(plan,"bilinear-retest",plan["retest_acceptance"])
  require(last["proposal_binding"]==plan["retest_proposal_binding"],"continuation_retest_audit_identity")
  status("complete",original_passed=first["passed"],original_failed=first["failed"],
   retest_passed=last["passed"],retest_failed=last["failed"],generations=last["generations"],
   http_attempts=last["http_attempts"],original_failure_preserved=True,manual_substitution=False)
  return 0
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument("--plan",action="store_true");mode.add_argument("--watch",action="store_true")
 args=parser.parse_args()
 if args.plan:
  value=verified();print(json.dumps(dict(status="verified_not_launched",binding=value["binding"],
   original_owner_live=run.same_process(value["main_owner"]),stages=value["stages"],new_api_calls=0)))
  return 0
 try:return watch()
 except BaseException as error:
  if OWNED:status("stopped",error=type(error).__name__,reason=str(error),automatic_retry=False)
  raise
if __name__=="__main__":raise SystemExit(main())
