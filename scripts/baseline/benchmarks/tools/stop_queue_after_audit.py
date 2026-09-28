"""Stop a serial scheduler only while its finished shard audit is its last child.
Never signal a candidate or restart a provider call. Retain all original reports.
"""
import argparse,json,os,signal,time
from pathlib import Path
from stage2_agent_campaign import ROOT
from benchmark_runner import strict_file,dump
from stage2_agent_pilot_plan import check_binding,sha
from campaign_live_state import sealed

def argv(pid):
 try:
  raw=(Path("/proc")/str(pid)/"cmdline").read_bytes().rstrip(bytes([0]))
  return raw.split(bytes([0])) if raw else []
 except (OSError,ProcessLookupError):return []
def matches(args,script,flag,value):
 return any(x.endswith(("/"+script).encode()) for x in args) and flag.encode() in args and args[args.index(flag.encode())+1]==str(value).encode()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--queue",type=Path,required=True)
 p.add_argument("--batch",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 from workspace_paths import RESULTS
 if any(x.is_symlink() or x.resolve().parent!=RESULTS.resolve() for x in (a.queue,a.batch,a.output)):raise ValueError("Direct result paths")
 if a.output.exists():raise ValueError("Fresh handoff record")
 a.output.mkdir();start=time.monotonic();pid=None;paused=False
 record=dict(queue=str(a.queue),batch=str(a.batch),reason="User requested bounded parallel shard dispatch",
  no_candidate_signal=True,new_paid_calls=0)
 try:
  while time.monotonic()-start<3600:
   if (a.queue/"report.json").exists():
    raise ValueError("Queue already stopped; inspect without restarting")
   queues=[];audits=[]
   for d in Path("/proc").iterdir():
    if not d.name.isdigit():continue
    args=argv(d.name)
    if matches(args,"campaign_queue_v2.py","--output",a.queue) and b"--inside" in args:queues.append((int(d.name),args))
    if matches(args,"audit_stage2_campaign_shard_v2.py","--batch",a.batch) and b"--inside" in args:audits.append(int(d.name))
   if len(queues)!=1:raise ValueError("Expected one verified serial scheduler")
   if audits:
    pid,expected=queues[0];os.kill(pid,signal.SIGSTOP);paused=True
    if argv(pid)!=expected:raise ValueError("Scheduler identity changed")
    audit_pid=audits[0];deadline=time.monotonic()+300
    while argv(audit_pid) and time.monotonic()<deadline:time.sleep(.2)
    if argv(audit_pid):raise TimeoutError("Audit did not finish at safe boundary")
    report=Path(str(a.batch)+"-audit")/"report.json"
    value=strict_file(report,16*1024**2);check_binding(value)
    terminal=strict_file(a.batch/"report.json",8*1024**2);check_binding(terminal)
    if terminal["failure"] or terminal["not_run_or_interrupted"] or value["statuses"].get("audit_failed"):
     raise ValueError("Unsafe terminal audit")
    for d in Path("/proc").iterdir():
     if d.name.isdigit() and any(x.endswith(b"/run_candidate.py") for x in argv(d.name)):
      raise ValueError("Unexpected candidate at audit boundary")
    record.update(scheduler_pid=pid,audit=str(report),audit_sha256=sha(report),
     plan_binding=terminal["plan_binding"],statuses=value["statuses"],no_active_candidate_at_signal=True)
    dump(a.output/"before-signal.json",sealed(record))
    os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT);paused=False
    deadline=time.monotonic()+30
    while argv(pid) and time.monotonic()<deadline:time.sleep(.2)
    if argv(pid):raise TimeoutError("Scheduler did not exit")
    record.update(status="stopped_at_completed_audit_boundary",seconds=time.monotonic()-start)
    dump(a.output/"report.json",sealed(record));print(json.dumps(record),flush=True);return 0
   time.sleep(.2)
  raise TimeoutError("No completed-audit boundary within observation budget")
 finally:
  if paused and pid is not None:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
if __name__=="__main__":raise SystemExit(main())
