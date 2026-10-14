"""Freeze only the serial dispatcher, let its live candidate finish, then stop.
The original queue budget remains enforced by its parent. No candidate is signalled.
"""
import argparse,json,os,signal,time
from pathlib import Path
from stage2_agent_campaign import ROOT
from stop_queue_after_audit import argv,matches
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed
from run_stage2_guidance_retest import evidence_path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--batch",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--resume-observation",type=Path);a=p.parse_args()
 from workspace_paths import RESULTS
 if a.output.exists():raise ValueError("Fresh output")
 dispatch=[];candidates=[]
 for d in Path("/proc").iterdir():
  if not d.name.isdigit():continue
  args=argv(d.name)
  if matches(args,"run_stage2_agent_campaign_v4.py","--output",a.batch) and b"--inside" in args:dispatch.append((int(d.name),args))
  if any(x.endswith(b"/run_candidate.py") for x in args) and b"--case" in args:
   case=Path(args[args.index(b"--case")+1].decode())
   if case.parent.parent==a.batch:candidates.append((int(d.name),args,case))
 if a.resume_observation is not None:
  prior=strict_file(a.resume_observation,131072);check_binding(prior)
  if prior["batch"]!=str(a.batch) or candidates:raise ValueError("Unexpected active candidate during terminal handoff")
  if len(dispatch)!=1 or dispatch[0][0]!=prior["dispatcher_pid"]:raise ValueError("Dispatcher changed")
  childargs=[x.encode() for x in prior["candidate_command"]]
  if argv(prior["candidate_pid"]):raise ValueError("Terminal observation requires exited candidate")
  state=(Path("/proc")/str(prior["dispatcher_pid"])/"stat").read_text().rsplit(")",1)[1].split()[0]
  if state!="T":raise ValueError("Dispatcher must remain frozen")
  case=Path(prior["candidate_command"][prior["candidate_command"].index("--case")+1])
  candidates=[(prior["candidate_pid"],childargs,case)]
 if len(dispatch)!=1 or len(candidates)!=1:raise ValueError("One actual dispatcher and candidate required")
 pid,expected=dispatch[0];child,childargs,case=candidates[0]
 a.output.mkdir();start=time.monotonic();paused=False
 record=dict(runner_sha256=sha(Path(__file__)),batch=str(a.batch),dispatcher_pid=pid,candidate_pid=child,candidate_command=[x.decode() for x in childargs],
  no_candidate_signal=True,new_paid_calls=0,reason="User requested parallel dispatch without interrupting in-flight generation")
 try:
  if argv(pid)!=expected or (argv(child)!=childargs and a.resume_observation is None):raise ValueError("Process identity changed")
  os.kill(pid,signal.SIGSTOP);paused=True;dump(a.output/"before-wait.json",sealed(record))
  original=strict_file(a.batch/"plan.json",8*1024**2);check_binding(original)
  # Resume the original dispatcher before its deadline so its unchanged
  # watchdog, not this handoff observer, owns cancellation and key cleanup.
  proc_stat=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
  deadline_monotonic=int(proc_stat[19])/os.sysconf("SC_CLK_TCK")+original["limits"]["max_wall_seconds"]-10
  while argv(child):
   if time.monotonic()>=deadline_monotonic:raise TimeoutError("Return control to original cumulative budget watchdog")

   if time.monotonic()-start>3600:raise TimeoutError("Original parent budget must resolve live candidate")
   if not argv(pid):raise ValueError("Original dispatcher terminated by parent; inspect original budget failure")
   time.sleep(.2)
  folder=evidence_path(case.parent/"run.log",RESULTS)
  if folder is None:raise ValueError("No terminal evidence")
  report=strict_file(folder/"report.json",8*1024**2)
  if report["status"] not in ("passed","repair_budget_exhausted","provider_failed"):raise ValueError("Nonterminal or infrastructure failure")
  record.update(evidence=str(folder),candidate_report_sha256=sha(folder/"report.json"),candidate_status=report["status"],
   seconds_waited=time.monotonic()-start,no_active_candidate_at_signal=True)
  dump(a.output/"before-signal.json",sealed(record))
  if argv(pid)!=expected:raise ValueError("Dispatcher changed")
  os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT);paused=False
  deadline=time.monotonic()+30
  while argv(pid) and time.monotonic()<deadline:time.sleep(.2)
  if argv(pid):raise TimeoutError("Dispatcher did not exit")
  terminal=strict_file(a.batch/"report.json",8*1024**2);check_binding(terminal)
  record.update(status="stopped_at_terminal_candidate_boundary",batch_report_sha256=sha(a.batch/"report.json"),
   cumulative_seconds=terminal["seconds"],requires_terminal_reconciliation=True)
  dump(a.output/"report.json",sealed(record));print(json.dumps(record),flush=True)
 finally:
  if paused:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
if __name__=="__main__":main()
