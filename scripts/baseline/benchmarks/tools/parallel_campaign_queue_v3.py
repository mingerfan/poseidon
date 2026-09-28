"""Bounded parallel shard scheduler: ten candidate/API workers, two native slots.
One pinned Nix parent. Continue only never-started shards, keep cumulative budgets.
A proven per-shard time stop is audited and isolated; integrity/environment failures stop the queue.
"""
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

def verify(plan):
 check_binding(plan)
 verify_continuation(plan)
 if plan["runner_sha256"]!=sha(Path(__file__)) or runtime_sources()!=plan["source_hashes"]:raise ValueError("Parallel source drift")
 if sha(Path(plan["source_index"]))!=plan["source_index_sha256"]:raise ValueError("Index drift")
 policy=dict(api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,max_retained_mib=8192,min_free_mib=4096,
             admission_available_mib=3072,stop_available_mib=512)
 if any(plan.get(k)!=v for k,v in policy.items()):raise ValueError("Parallel resource policy drift")
 if not 1<=len(plan["shards"])<=43:raise ValueError("Shard count")
 refs=plan["shards"]
 if len({r["binding"] for r in refs})!=len(refs):raise ValueError("Duplicate shard")
 from workspace_paths import RESULTS
 paths=set();totals=dict(tasks=0,generations=0,http=0,seconds=0)
 for r in refs:
  p=Path(r["plan"]);s=checked_plan(p)
  if s.get("shared_scheduling")!={"api_workers":10,"native_workers":2,"version":1}:raise ValueError("Serial-only shard")
  if sha(p)!=r["sha256"] or s["binding"]!=r["binding"]:raise ValueError("Shard identity")
  n=len(s["cases"])
  if not 1<=n<=48 or s["limits"]["concurrency"]!=1 or r["tasks"]!=n:raise ValueError("Per-shard serial boundary")
  if r["max_wall_seconds"]!=s["limits"]["max_wall_seconds"]:raise ValueError("Shard time drift")
  for k in ("output","audit"):
   q=Path(r[k])
   if q.is_symlink() or q.resolve().parent!=RESULTS.resolve() or str(q) in paths:raise ValueError("Output boundary")
   paths.add(str(q))
  totals["tasks"]+=n;totals["generations"]+=s["limits"]["maximum_generations"];totals["http"]+=s["limits"]["maximum_http_attempts"];totals["seconds"]+=r["max_wall_seconds"]+390
 for name,key in (("planned_tasks","tasks"),("maximum_generations","generations"),("maximum_http_attempts","http"),("max_wall_seconds","seconds")):
  if plan[name]!=totals[key]:raise ValueError("Total budget drift")
 for name,h in plan["handoff_parents"].items():
  q=Path(name)
  if q.is_symlink() or not q.resolve().is_relative_to(RESULTS.resolve()) or sha(q)!=h:raise ValueError("Handoff evidence drift")
 for name in plan["inherited_artifact_roots"]:
  q=Path(name)
  if q.is_symlink() or q.resolve().parent!=RESULTS.resolve():raise ValueError("Inherited evidence boundary")

def shard_stop_kind(terminal,ref):
 if terminal["plan_binding"]!=ref["binding"]:raise ValueError("Shard identity")
 if terminal["failure"]=="TimeoutError: Cumulative wall budget exhausted":
  if terminal["seconds"]<ref["max_wall_seconds"]:raise ValueError("Unproven shard budget stop")
  return "audited_budget_stopped"
 if terminal["failure"] or terminal["not_run_or_interrupted"]:raise ValueError("Shard stopped or uncertain")
 return "audited_terminal"

def verify_continuation(plan):
 from workspace_paths import RESULTS
 c=plan["continuation"]
 old_path=Path(c["queue_plan"]);report_path=Path(c["queue_report"])
 for p in (old_path,report_path):
  if p.is_symlink() or not p.resolve().is_relative_to(RESULTS.resolve()):raise ValueError("Continuation path")
 if sha(old_path)!=c["queue_plan_sha256"] or sha(report_path)!=c["queue_report_sha256"]:raise ValueError("Prior queue changed")
 old=strict_file(old_path,1024**2);report=strict_file(report_path,1024**2)
 check_binding(old);check_binding(report)
 if report["plan_binding"]!=old["binding"] or not report["failure"] or report["queue_finished"]:
  raise ValueError("Stopped predecessor required")
 if old["source_hashes"]!=plan["source_hashes"]:raise ValueError("Prior source differs")
 if c["seconds_used"]!=report["seconds"] or c["seconds_remaining"]!=old["max_wall_seconds"]-report["seconds"]:
  raise ValueError("Cumulative queue budget reset")
 if plan["max_wall_seconds"]>c["seconds_remaining"]:raise ValueError("Remaining total budget exceeded")
 refs={r["binding"]:r for r in old["shards"]}
 if len({r["previous_binding"] for r in plan["shards"]})!=len(plan["shards"]):raise ValueError("Repeated predecessor shard")
 for r in plan["shards"]:
  prior=refs.get(r["previous_binding"])
  if prior is None:raise ValueError("Unknown predecessor shard")
  if Path(prior["output"]).exists() or Path(prior["audit"]).exists() or (old_path.parent/"launches"/prior["binding"]).exists():
   raise ValueError("Previously started shard cannot receive fresh budget")
  original=checked_plan(Path(prior["plan"]));current=checked_plan(Path(r["plan"]))
  if sha(Path(prior["plan"]))!=prior["sha256"]:raise ValueError("Prior shard changed")
  if any(current[k]!=original[k] for k in ("cases","limits","paid_configuration","source_hashes")):
   raise ValueError("Continuation changes requests or budgets")

def prepare(index_path,output,previous):
 from workspace_paths import RESULTS
 if output.exists() or output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output")
 old_path=previous/"plan.json";report_path=previous/"report.json"
 old=strict_file(old_path,1024**2);report=strict_file(report_path,1024**2)
 check_binding(old);check_binding(report)
 index=strict_file(index_path,16*1024**2);check_binding(index)
 new_by_cases={}
 for item in index["shards"]:
  p=index_path.parent/item["file"];v=checked_plan(p)
  if sha(p)!=item["sha256"]:raise ValueError("Index shard drift")
  new_by_cases[tuple(x["evaluation_identity"] for x in v["cases"])]=(p,v)
 refs=[];inherited=set(old["inherited_artifact_roots"]);inherited.add(str(previous))
 parents={str(old_path):sha(old_path),str(report_path):sha(report_path)}
 for prior in old["shards"]:
  p=Path(prior["plan"]);v=checked_plan(p)
  folder=Path(prior["output"]);audit=Path(prior["audit"])
  if folder.exists() or audit.exists() or (previous/"launches"/prior["binding"]).exists():
   for q in (folder,audit):
    if q.exists():inherited.add(str(q))
    if (q/"report.json").exists():parents[str(q/"report.json")]=sha(q/"report.json")
   for log in folder.glob("*/run.log"):
    ev=evidence_path(log,RESULTS)
    if ev:inherited.add(str(ev))
   continue
  new_path,spec=new_by_cases[tuple(x["evaluation_identity"] for x in v["cases"])]
  i=Path(prior["output"]).name.rsplit("-",1)[-1]
  refs.append(dict(plan=str(new_path),sha256=sha(new_path),binding=spec["binding"],tasks=len(spec["cases"]),
   previous_binding=prior["binding"],output=str(RESULTS/(output.name+"-shard-"+i)),
   audit=str(RESULTS/(output.name+"-shard-"+i+"-audit")),max_wall_seconds=spec["limits"]["max_wall_seconds"]))
 value=sealed(dict(format="poseidon-parallel-campaign-continuation-v2",shards=refs,source_hashes=runtime_sources(),
  source_index=str(index_path),source_index_sha256=sha(index_path),runner_sha256=sha(Path(__file__)),
  handoff_parents=parents,inherited_artifact_roots=sorted(inherited),
  continuation=dict(queue_plan=str(old_path),queue_report=str(report_path),queue_plan_sha256=sha(old_path),
   queue_report_sha256=sha(report_path),seconds_used=report["seconds"],seconds_remaining=old["max_wall_seconds"]-report["seconds"]),
  planned_tasks=sum(r["tasks"] for r in refs),maximum_generations=sum(checked_plan(Path(r["plan"]))["limits"]["maximum_generations"] for r in refs),
  maximum_http_attempts=sum(checked_plan(Path(r["plan"]))["limits"]["maximum_http_attempts"] for r in refs),
  max_wall_seconds=sum(r["max_wall_seconds"]+390 for r in refs),max_retained_mib=8192,min_free_mib=4096,
  api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,admission_available_mib=3072,stop_available_mib=512,
  monetary_cap=None,automatic_restart=False,stage2_complete=False))
 verify(value);output.mkdir();dump(output/"plan.json",value)
 print(json.dumps({k:value[k] for k in ("binding","planned_tasks","maximum_generations","maximum_http_attempts","max_wall_seconds","api_workers","native_workers")}))


def run(output,binding,probe):
 from hecate_python_env import VENV
 from workspace_paths import RESULTS
 from run_stage2_agent_campaign_v7 import validate_plan
 plan=strict_file(output/"plan.json",1024**2);verify(plan)
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
   command=[str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("run_stage2_agent_campaign_v7.py")),
    "--plan",ref["plan"],"--approve-live-binding",ref["binding"],"--output",ref["output"],
    "--parallel-queue-plan",str(output/"plan.json"),"--inside"]
   exclusive_json(mark/"launch.json",sealed(dict(command=command,queue_binding=binding,automatic_retry=False)))
   code=execute(command,mark/"run.log",ref["max_wall_seconds"]+60)
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
 p.add_argument("--index",type=Path);p.add_argument("--previous",type=Path);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--approve-live-binding");p.add_argument("--probe",type=Path);p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 if a.mode=="prepare":
  if a.index is None or a.previous is None:p.error("Index and stopped predecessor required")
  prepare(a.index,a.output,a.previous);return 0
 if a.probe is None:p.error("Concurrent offline proof required")
 plan=strict_file(a.output/"plan.json",1024**2);verify(plan)
 if a.approve_live_binding!=plan["binding"]:p.error("Exact binding")
 if not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=plan["max_wall_seconds"]+90)
 return run(a.output,a.approve_live_binding,a.probe)
if __name__=="__main__":raise SystemExit(main())
