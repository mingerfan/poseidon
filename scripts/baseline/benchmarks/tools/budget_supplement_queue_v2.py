"""Only the explicitly approved 47 unlaunched tasks, preserving six original clocks."""
from retained_artifact_usage import size_bytes

import argparse,fcntl,json,os,shlex,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed,exclusive_json
from semantic_benchmark_execution import runtime_sources
from campaign_queue_v2 import checked_plan
from run_stage2_guidance_retest import stop_child,evidence_path

def available_mib():
 values=dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
 return int(values["MemAvailable"].split()[0])//1024


def shard_stop_kind(terminal,ref):
 if terminal["plan_binding"]!=ref["binding"]:raise ValueError("Shard identity")
 if terminal["failure"]=="TimeoutError: Cumulative wall budget exhausted":
  if terminal["seconds"]<7200:raise ValueError("Unproven cumulative time stop")
  return "audited_budget_stopped"
 if terminal["failure"] or terminal["not_run_or_interrupted"]:raise ValueError("Shard stopped or uncertain")
 return "audited_terminal"

def verify(plan):
 from workspace_paths import RESULTS
 from approved_budget_extension import verify_extension
 check_binding(plan)
 if plan["runner_sha256"]!=sha(Path(__file__)) or plan["source_hashes"]!=runtime_sources():raise ValueError("Supplement source drift")
 policy=dict(api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,max_retained_mib=8192,min_free_mib=4096,
  admission_available_mib=3072,stop_available_mib=512)
 if any(plan.get(k)!=v for k,v in policy.items()):raise ValueError("Resource policy drift")
 if len(plan["shards"])!=6:raise ValueError("Six approved shards required")
 for name,h in plan["handoff_parents"].items():
  p=Path(name)
  if p.is_symlink() or not any(p.resolve().is_relative_to(b.resolve()) for b in (ROOT,RESULTS)) or sha(p)!=h:
   raise ValueError("Bound preparation changed")
 totals=dict(tasks=0,generations=0,http=0,seconds=0);seen=set();paths=set()
 for ref in plan["shards"]:
  p=Path(ref["plan"]);s=checked_plan(p);prepared=verify_extension(s)
  if sha(p)!=ref["sha256"] or s["binding"]!=ref["binding"]:raise ValueError("Shard plan drift")
  n=len(s["cases"]);number=prepared["shard"]
  if number in seen:raise ValueError("Duplicate authorized shard")
  seen.add(number)
  if ref["tasks"]!=n or ref["remaining_seconds"]!=7200-s["budget_extension"]["seconds_already_used"] or ref["max_wall_seconds"]!=7200:
   raise ValueError("Remaining budget reset")
  if s["limits"]["concurrency"]!=1:raise ValueError("Per-shard serial execution required")
  for key in ("output","audit"):
   q=Path(ref[key])
   if q.is_symlink() or q.resolve().parent!=RESULTS.resolve() or str(q) in paths:raise ValueError("Output boundary")
   paths.add(str(q))
  totals["tasks"]+=n;totals["generations"]+=4*n;totals["http"]+=16*n;totals["seconds"]+=ref["remaining_seconds"]+390
 if seen!={"002","003","007","009","010","011"}:raise ValueError("Unauthorized shard set")
 if (totals["tasks"],totals["generations"],totals["http"])!=(47,188,752):raise ValueError("Authorization ceiling changed")
 for key,total in (("planned_tasks","tasks"),("maximum_generations","generations"),("maximum_http_attempts","http"),("max_wall_seconds","seconds")):
  if plan[key]!=totals[total]:raise ValueError("Total budget mismatch")
 previous=strict_file(Path(plan["predecessor_queue_plan"]),1024**2);check_binding(previous)
 if previous["binding"]!=plan["predecessor_queue_binding"] or previous["source_hashes"]!=plan["source_hashes"]:raise ValueError("Predecessor mismatch")
 if plan["max_wall_seconds"]>previous["continuation"]["seconds_remaining"]-previous["max_wall_seconds"]:
  raise ValueError("Original aggregate time ceiling exceeded")
 for name in plan["inherited_artifact_roots"]:
  q=Path(name)
  if q.is_symlink() or q.resolve().parent!=RESULTS.resolve():raise ValueError("Inherited path boundary")

def previous_terminal(plan):
 previous_path=Path(plan["predecessor_queue_plan"]);previous=strict_file(previous_path,1024**2);check_binding(previous)
 report_path=previous_path.parent/"report.json"
 if not report_path.exists():raise ValueError("Predecessor still active; do not launch supplement")
 report=strict_file(report_path,1024**2);check_binding(report)
 if report["plan_binding"]!=previous["binding"] or report["failure"] or not report["queue_finished"]:
  raise ValueError("Predecessor needs reconciliation before supplement")
 if report["seconds"]+previous["continuation"]["seconds_used"]+plan["max_wall_seconds"]>previous["continuation"]["seconds_remaining"]+previous["continuation"]["seconds_used"]:
  raise ValueError("Original cumulative queue clock exceeded")
 return sealed(dict(plan=str(previous_path),plan_sha256=sha(previous_path),report=str(report_path),
  report_sha256=sha(report_path),seconds=report["seconds"],new_paid_calls=0))

def prepare(index_path,output,previous_path):
 from workspace_paths import RESULTS
 from approved_budget_extension import retained_paths
 if output.exists() or output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output")
 index=strict_file(index_path,1024**2);check_binding(index)
 previous=strict_file(previous_path,1024**2);check_binding(previous)
 refs=[];parents={str(index_path):sha(index_path),str(previous_path):sha(previous_path)}
 inherited=set(previous["inherited_artifact_roots"]);inherited.add(str(previous_path.parent))
 for ref in previous["shards"]:inherited.update([ref["output"],ref["audit"]])
 output.mkdir();plans=output/"plans";plans.mkdir()
 for row in index["shards"]:
  prep_path=index_path.parent/row["file"]
  if sha(prep_path)!=row["sha256"]:raise ValueError("Preparation drift")
  prep=strict_file(prep_path,8*1024**2);check_binding(prep)
  original=strict_file(Path(prep["original_plan"]),8*1024**2);check_binding(original)
  s=dict(original);s.pop("binding");s["cases"]=prep["cases"]
  n=len(s["cases"]);s["limits"]=dict(original["limits"],max_wall_seconds=7200,maximum_generations=4*n,maximum_http_attempts=16*n)
  s["budget_extension"]=dict(preparation=str(prep_path),preparation_sha256=sha(prep_path),seconds_already_used=prep["limits"]["original_seconds_used"])
  s["proposal_files"]=dict(original["proposal_files"])
  for name in ("approved_budget_extension.py","run_stage2_budget_supplement_v2.py","budget_supplement_queue_v2.py","retained_artifact_usage.py"):
   f=Path(__file__).with_name(name);s["proposal_files"][str(f.relative_to(ROOT))]=sha(f)
  s=sealed(s);p=plans/("shard-"+prep["shard"]+".json");dump(p,s)
  refs.append(dict(plan=str(p),sha256=sha(p),binding=s["binding"],tasks=n,max_wall_seconds=7200,
   remaining_seconds=7200-prep["limits"]["original_seconds_used"],
   output=str(RESULTS/(output.name+"-shard-"+prep["shard"])),audit=str(RESULTS/(output.name+"-shard-"+prep["shard"]+"-audit"))))
  inherited.update(str(p) for p in retained_paths(prep))
  for name,h in ((str(prep_path),sha(prep_path)),(prep["authorization"],prep["authorization_sha256"]),
   (prep["original_plan"],prep["original_plan_sha256"]),(prep["original_report"],prep["original_report_sha256"])):
   parents[name]=h
 plan=sealed(dict(format="poseidon-approved-budget-supplement-queue-v1",shards=refs,source_hashes=runtime_sources(),
  runner_sha256=sha(Path(__file__)),handoff_parents=parents,inherited_artifact_roots=sorted(inherited),
  predecessor_queue_plan=str(previous_path),predecessor_queue_binding=previous["binding"],
  planned_tasks=sum(r["tasks"] for r in refs),maximum_generations=188,maximum_http_attempts=752,
  max_wall_seconds=sum(r["remaining_seconds"]+390 for r in refs),max_retained_mib=8192,min_free_mib=4096,
  api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,admission_available_mib=3072,stop_available_mib=512,
  monetary_cap=None,automatic_restart=False,stage2_complete=False))
 verify(plan);dump(output/"plan.json",plan)
 print(json.dumps(dict(binding=plan["binding"],tasks=47,shards=6,max_wall_seconds=plan["max_wall_seconds"],new_paid_calls=0)))

def run(output,binding,probe):
 from hecate_python_env import VENV
 from workspace_paths import RESULTS
 from run_stage2_budget_supplement_v2 import validate_plan
 plan=strict_file(output/"plan.json",1024**2);verify(plan)
 previous_terminal(plan)
 if (output/"predecessor-terminal.json").exists():raise ValueError("Prior supplement dispatch exists")
 if binding!=plan["binding"]:raise ValueError("Exact live binding")
 proof=strict_file(probe,4*1024**2);check_binding(proof)
 if (proof["source_hashes"]!=plan["source_hashes"] or proof["passed"]!=2 or not proof["two_native_slots_observed"]
     or proof["min_available_mib"]<1024 or proof["new_paid_calls"]!=0):raise ValueError("Concurrent offline acceptance missing")
 for name,hsh in proof["parents"].items():
  if sha(Path(name))!=hsh:raise ValueError("Probe evidence changed")
 if proof["runner_sha256"]!=sha(Path(__file__).with_name("probe_native_parallel.py")):raise ValueError("Probe implementation changed")
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure Python")
 if (output/"report.json").exists() or (output/"launches").exists():raise ValueError("Preserve prior launch; reconcile explicitly")
 for ref in plan["shards"]:validate_plan(strict_file(Path(ref["plan"]),8*1024**2),ref["binding"])
 fd=os.open(RESULTS/"stage2-agent-paid-queue.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"a+") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  predecessor=previous_terminal(plan)
  exclusive_json(output/"predecessor-terminal.json",predecessor)
  launches=output/"launches";launches.mkdir();stop=threading.Event();guard=threading.Lock()
  active={};rows=[];started=time.monotonic();failure=None
  def execute(command,log,seconds):
   with log.open("x") as stream:
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    with guard:active[child.pid]=child
    start=time.monotonic()
    try:
     while child.poll() is None:
      if stop.is_set():raise RuntimeError("Parallel queue cancelled; retain uncertain calls")
      if time.monotonic()-start>=seconds:raise TimeoutError("Child time boundary")
      try:child.wait(timeout=1)
      except subprocess.TimeoutExpired:pass
     return child.returncode
    finally:
     stop_child(child)
     with guard:active.pop(child.pid,None)
  def worker(ref):
   while available_mib()<plan["admission_available_mib"]:
    if stop.wait(.5):raise RuntimeError("Admission cancelled")
   if stop.is_set():raise RuntimeError("Cancelled before claim")
   mark=launches/ref["binding"];mark.mkdir()
   command=[str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("run_stage2_budget_supplement_v2.py")),
    "--plan",ref["plan"],"--approve-live-binding",ref["binding"],"--output",ref["output"],
    "--parallel-queue-plan",str(output/"plan.json"),"--inside"]
   exclusive_json(mark/"launch.json",sealed(dict(command=command,queue_binding=binding,automatic_retry=False)))
   code=execute(command,mark/"run.log",ref["remaining_seconds"]+60)
   terminal=strict_file(Path(ref["output"])/"report.json",8*1024**2);check_binding(terminal)
   terminal_kind=shard_stop_kind(terminal,ref)
   command=[str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("audit_stage2_campaign_shard_v2.py")),
    "--batch",ref["output"],"--output",ref["audit"],"--inside"]
   acode=execute(command,mark/"audit.log",300)
   result=strict_file(Path(ref["audit"])/"report.json",16*1024**2);check_binding(result)
   if acode or result["plan_binding"]!=ref["binding"] or result["statuses"].get("audit_failed"):raise ValueError("Independent audit failed")
   return dict(binding=ref["binding"],status=terminal_kind,runner_exit=code,output=ref["output"],audit=ref["audit"],
    audit_sha256=sha(Path(ref["audit"])/"report.json"),statuses=result["statuses"],generations=result["generations"],http_attempts=result["http_attempts"])
  def boundary():
   if time.monotonic()-started>=plan["max_wall_seconds"]:raise TimeoutError("Queue total time boundary")
   if available_mib()<plan["stop_available_mib"]:raise ValueError("Available memory safety reserve")
   stat=os.statvfs(RESULTS)
   if stat.f_bavail*stat.f_frsize<plan["min_free_mib"]*1024**2:raise ValueError("Disk reserve")
   paths=[output,*map(Path,plan["inherited_artifact_roots"])]
   for ref in plan["shards"]:
    folder=Path(ref["output"]);paths.extend((folder,Path(ref["audit"])))
    for log in folder.glob("*/run.log"):
     f=evidence_path(log,RESULTS)
     if f:paths.append(f)
   for inherited in list(paths):
    for log in inherited.glob("*/run.log"):
     evidence=evidence_path(log,RESULTS)
     if evidence:paths.append(evidence)
   if size_bytes(paths)>plan["max_retained_mib"]*1024**2:raise ValueError("Aggregate artifact boundary")
   if runtime_sources()!=plan["source_hashes"] or sha(Path(__file__))!=plan["runner_sha256"]:raise ValueError("Parallel source drift")
  pool=ThreadPoolExecutor(max_workers=plan["api_workers"])
  try:
   futures={pool.submit(worker,r) for r in plan["shards"]}
   while futures:
    boundary();done,futures=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
    for f in done:
     row=f.result();rows.append(row);dump(output/"progress.json",rows);print(json.dumps(row),flush=True)
  except BaseException as e:
   failure=type(e).__name__+": "+str(e);stop.set()
   with guard:children=list(active.values())
   for child in children:stop_child(child)
   raise
  finally:
   stop.set();pool.shutdown(wait=True,cancel_futures=True)
   dump(output/"report.json",sealed(dict(format="poseidon-parallel-campaign-run-v1",plan_binding=binding,
    rows=rows,seconds=time.monotonic()-started,failure=failure,queue_finished=len(rows)==len(plan["shards"]),
    uncertain_calls_possible=bool(failure),stage2_complete=False)))
 return 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","run"))
 p.add_argument("--index",type=Path);p.add_argument("--previous-plan",type=Path);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--approve-live-binding");p.add_argument("--probe",type=Path);p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 if a.mode=="prepare":
  if a.index is None or a.previous_plan is None:p.error("Index and predecessor plan required")
  prepare(a.index,a.output,a.previous_plan);return 0
 plan=strict_file(a.output/"plan.json",1024**2);verify(plan)
 if a.approve_live_binding!=plan["binding"] or a.probe is None:p.error("Exact binding and offline native proof required")
 previous_terminal(plan)
 if not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=plan["max_wall_seconds"]+90)
 return run(a.output,a.approve_live_binding,a.probe)
if __name__=="__main__":raise SystemExit(main())
