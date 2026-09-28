"""Read-only bounded process/resource sampling; sampled peaks are not full-run peaks."""
import argparse,json,os,time
from pathlib import Path
from stage2_agent_campaign import ROOT
from stage2_agent_pilot_plan import sha,check_binding
from campaign_live_state import sealed
from benchmark_runner import strict_file,dump
from run_stage2_guidance_retest import evidence_path,size_bytes

def stat(pid):
 p=Path("/proc")/str(pid)
 try:
  fields=(p/"stat").read_text().rsplit(")",1)[1].split()
  status=dict(line.split(":",1) for line in (p/"status").read_text().splitlines() if ":" in line)
  return dict(pid=pid,ppid=int(fields[1]),start_ticks=int(fields[19]),
   rss_kib=int(status.get("VmRSS","0 kB").split()[0]),
   cpu_ticks=int(fields[11])+int(fields[12]))
 except (FileNotFoundError,ProcessLookupError):return None

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--batch",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--seconds",type=int,default=600);a=p.parse_args()
 from workspace_paths import RESULTS
 if not 1<=a.seconds<=3600:raise ValueError("Observation bound")
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result file")
 plan=strict_file(a.batch/"plan.json",8*1024**2);check_binding(plan);root=None
 for f in a.batch.glob("*/process.json"):
  candidate=stat(strict_file(f,131072)["pid"])
  if candidate is None:continue
  parent=stat(candidate["ppid"])
  if parent is None:continue
  command=(Path("/proc")/str(parent["pid"])/"cmdline").read_bytes().rstrip(b"\0").split(b"\0")
  allowed=[str(ROOT/"scripts/baseline/benchmarks/tools"/n).encode() for n in ("run_stage2_agent_campaign_v2.py","run_stage2_agent_campaign_v3.py")]
  if any(x in command for x in allowed) and str(a.batch).encode() in command:root=parent;break
 if root is None:raise ValueError("No verified active runner")
 start=time.monotonic();records=[];reason="observation_time_bound"
 while time.monotonic()-start<a.seconds:
  current=stat(root["pid"])
  if current is None or current["start_ticks"]!=root["start_ticks"]:reason="verified_runner_exited";break
  pending=[root["pid"]];seen=set();processes=[]
  while pending:
   pid=pending.pop()
   if pid in seen:continue
   seen.add(pid);item=stat(pid)
   if item is None:continue
   processes.append(item)
   try:pending.extend(int(x) for x in (Path("/proc")/str(pid)/"task"/str(pid)/"children").read_text().split())
   except (FileNotFoundError,ProcessLookupError):pass
  mem=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
  paths=[a.batch]
  for log in a.batch.glob("*/run.log"):
   folder=evidence_path(log,RESULTS)
   if folder:paths.append(folder)
  disk=os.statvfs(RESULTS)
  records.append(dict(elapsed_seconds=time.monotonic()-start,descendant_processes=len(processes),
   summed_rss_kib=sum(x["rss_kib"] for x in processes),available_memory_kib=int(mem["MemAvailable"].split()[0]),
   disk_available_bytes=disk.f_bavail*disk.f_frsize,observed_result_bytes=size_bytes(paths)))
  time.sleep(2)
 report=sealed(dict(format="poseidon-sampled-resources-v1",plan_binding=plan["binding"],runner=root,
  sampler_sha256=sha(Path(__file__)),sample_interval_seconds=2,samples=len(records),stop_reason=reason,
  seconds=time.monotonic()-start,records=records,
  peak_sampled_summed_rss_kib=max((x["summed_rss_kib"] for x in records),default=0),
  minimum_sampled_available_memory_kib=min((x["available_memory_kib"] for x in records),default=None),
  max_sampled_result_bytes=max((x["observed_result_bytes"] for x in records),default=0),
  limitation="Partial process-tree samples; shared pages may be counted twice; transients between samples and earlier run peaks are not observed.",
  paid_calls=0,encrypted_executions=0))
 dump(a.output,report);print(json.dumps({k:v for k,v in report.items() if k!="records"}))
 return 0
if __name__=="__main__":raise SystemExit(main())
