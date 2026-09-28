"""Explicitly authorized recovery only; immutable old claims and separate results.

One pure Nix parent, four serial shards with one shared native slot, independent candidate sandboxes.
No automatic queue resume, claim reset, extra cases, or source-answer substitution.
"""
import argparse,fcntl,json,os,shlex,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE,PAID,LIMITS,identity
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed,exclusive_json,claim
from campaign_request_files import read_request
from campaign_candidate_failures import transparent_failure
from retained_artifact_usage import size_bytes
from run_stage2_guidance_retest import outcome,terminal_failure_layer,stop_child,evidence_path
from semantic_benchmark_execution import runtime_sources

from stage2_agent_repair_plan_r178 import build, verify, document, tools_sources

from contextlib import contextmanager
@contextmanager
def serial_native_reservation():
 from hecate_python_env import WORK
 import stat
 directory=WORK/"cache/agent-native-slots"
 directory.mkdir(mode=0o700,parents=True,exist_ok=True)
 info=directory.lstat()
 if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o022:
  raise ValueError("Unsafe native slot directory")
 fd=os.open(directory/"slot-1.lock",os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
 try:
  info=os.fstat(fd)
  if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1 or info.st_mode&0o022:
   raise ValueError("Unsafe native slot reservation")
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield
 finally:os.close(fd)

class ShardBudget(ValueError):
 pass

def available_mib():
 values=dict(l.split(":",1) for l in Path("/proc/meminfo").read_text().splitlines())
 return int(values["MemAvailable"].split()[0])//1024

def assert_idle():
 for path in Path("/proc").glob("[0-9]*/cmdline"):
  try:args=path.read_bytes().split(bytes([0]))
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(x.endswith(b"/run_candidate.py") or x.endswith(b"/run_stage2_remaining_campaign_v3.py") for x in args):
   raise ValueError("Existing candidate/shard process; preserve it")

def candidate_command(shard,spec,work):
 # The public launcher owns .env discovery and cleanup. In a pinned pure shell
 # it invokes inside() directly, avoiding nested Nix without bypassing credentials.
 return [shard["executable"],"-B",shard["entrypoint"],"--case",str(work/"model.json"),*spec["candidate_arguments"]]

def run(output,binding,check_only=False):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from platform_config import require_python_packages
 import numpy as np,torch
 require_python_packages(torch,np)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure Nix required")
 started=time.monotonic()
 plan=document(output/"plan.json");verify(plan,output)
 if binding!=plan["binding"]:raise ValueError("Exact execution binding")
 if check_only:
  print(json.dumps(dict(validated=True,binding=binding,tasks=19,shards=4,new_paid_calls=0)),flush=True);return 0
 if (output/"dispatch.json").exists() or (output/"report.json").exists():raise ValueError("Already dispatched; no automatic retry")
 fd=os.open(RESULTS/"stage2-agent-paid-queue.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 campaign_fd=os.open(RESULTS/"stage2-agent-campaign-execution.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"a+") as lock,os.fdopen(campaign_fd,"a+") as campaign_lock,serial_native_reservation():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert_idle()
  for ref in plan["shards"]:
   for k in ("output","audit"):
    if Path(ref[k]).exists():raise ValueError("Existing shard artifacts")
   for s in ref["plan"]["cases"]:
    if (Path(plan["claim_registry"])/(s["evaluation_identity"]+".json")).exists():
     raise ValueError("Targeted retry already claimed; no automatic second evaluation")
  if size_bytes([RESULTS])+4*1024**3>32*1024**3:raise ValueError("Insufficient aggregate candidate headroom")
  if available_mib()<3072:raise ValueError("Insufficient admission memory")
  stop=threading.Event();guard=threading.Lock();active={};rows=[];failure=None
  def boundary(storage=False):
   if plan["prior_seconds"]+time.monotonic()-started>=43200:raise TimeoutError("Targeted retry total time budget")
   if available_mib()<512:raise ValueError("Available memory reserve")
   stat=os.statvfs(RESULTS)
   if stat.f_bavail*stat.f_frsize<4096*1024**2:raise ValueError("Disk free reserve")
   if runtime_sources()!=plan["source_hashes"] or tools_sources()!=plan["proposal_files"]:raise ValueError("Runtime/tool source drift")
   if sha(Path(plan["proposal"]))!=plan["proposal_sha256"] or sha(Path(plan["authorization"]))!=plan["authorization_sha256"]:raise ValueError("Authorization drift")
   if storage and size_bytes([RESULTS])>=32768*1024**2:raise ValueError("Aggregate retained space budget")
   if sha(Path(plan["native_probe"]))!=plan["native_probe_sha256"]:raise ValueError("Native proof drift")
  boundary(True)
  exclusive_json(output/"dispatch.json",sealed(dict(plan_binding=binding,automatic_restart=False,status="dispatching")))
  def execute(command,log,seconds,local_boundary):
   if stop.is_set():raise RuntimeError("Recovery cancelled")
   with log.open("x") as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   with guard:active[child.pid]=child
   began=time.monotonic()
   try:
    while child.poll() is None:
     if stop.is_set():raise RuntimeError("Recovery cancelled; preserve uncertain calls")
     if time.monotonic()-began>=seconds:raise TimeoutError("Child hard timeout")
     local_boundary()
     try:child.wait(timeout=1)
     except subprocess.TimeoutExpired:pass
    local_boundary();return child.returncode
   finally:
    stop_child(child)
    with guard:active.pop(child.pid,None)
  def worker(ref):
   while available_mib()<3072:
    if stop.wait(.5):raise RuntimeError("Admission cancelled")
   if stop.is_set():raise RuntimeError("Cancelled before shard")
   out=Path(ref["output"]);out.mkdir();shard=ref["plan"];dump(out/"plan.json",shard)
   completed=[];folders=set();error=None;begin=time.monotonic();consecutive_service_failures=0
   def local_boundary():
    if stop.is_set():raise RuntimeError("Recovery cancelled")
    for log in out.glob("*/run.log"):
     folder=evidence_path(log,RESULTS)
     if folder:folders.add(folder)
    if size_bytes([out,*folders])>=4096*1024**2:raise ShardBudget("Shard retained space budget")
   try:
    for s in shard["cases"]:
     local_boundary()
     if size_bytes([out,*folders])+1024**3>4096*1024**2:raise ShardBudget("Shard next candidate headroom")
     acquired,_=claim(Path(plan["claim_registry"]),s,plan["source_hashes"],out,shard["binding"])
     if not acquired:raise ValueError("Existing recovery claim; never reset")
     work=out/s["id"];work.mkdir();dump(work/"model.json",s["model"]);dump(work/"request.expected.json",s["request"])
     cmd=candidate_command(shard,s,work)
     exclusive_json(work/"launch.json",sealed(dict(plan_binding=shard["binding"],evaluation_identity=s["evaluation_identity"],command=cmd,automatic_restart_allowed=False)))
     code=execute(cmd,work/"run.log",43200,local_boundary)
     folder=evidence_path(work/"run.log",RESULTS)
     if folder is None:raise ValueError("Missing candidate evidence")
     folders.add(folder);report=strict_file(folder/"report.json",8*1024**2)
     if report["status"]=="running":raise ValueError("Uncertain candidate report")
     read_request(folder/"request.json",s["request"])
     if digest(strict_file(folder/"model.json",131072))!=s["model_sha256"]:raise ValueError("Executed model changed")
     row=dict(id=s["id"],track=s["track"],evaluation_identity=s["evaluation_identity"],status=outcome(report,code),exit_code=code,
      evidence=str(folder),report_sha256=sha(folder/"report.json"),agent_calls=report["agent_calls"],
      generations=report.get("provider_metrics",{}).get("generation_attempts",0),failure_layer=terminal_failure_layer(report))
     rejection=transparent_failure(folder,report,PAID)
     if rejection:row["candidate_rejection"]=rejection
     completed.append(row);dump(out/"progress.json",completed)
     if report["status"]=="provider_failed":
      last=report.get("provider_metrics",{}).get("calls",[])[-1]
      row["provider_error"]=last.get("error")
      consecutive_service_failures+=1
      if last.get("error") in ("http_status_401","http_status_403","response_model_mismatch"):
       raise ValueError("Provider identity/access failure")
      if consecutive_service_failures>=3:
       raise ShardBudget("Three consecutive provider failures; stop this shard pending diagnosis")
     else:consecutive_service_failures=0
     if row["failure_layer"] in ("environment","harness_launch","key_setup","integrity") or (row["failure_layer"]=="seal_runtime" and not rejection):
      raise ValueError("Environment/integrity failure")
   except ShardBudget as e:error=type(e).__name__+": "+str(e)
   except BaseException as e:error=type(e).__name__+": "+str(e);raise
   finally:
    dump(out/"report.json",sealed(dict(format="poseidon-stage2-agent-campaign-run-v1",plan_binding=shard["binding"],rows=completed,
     not_run_or_interrupted=[s["id"] for s in shard["cases"] if s["id"] not in {r["id"] for r in completed}],
     seconds=time.monotonic()-begin,failure=error,independent_audit_complete=False,stage2_complete=False)))
   audit=Path(ref["audit"])
   cmd=[shard["executable"],"-B",str(Path(__file__).with_name("audit_stage2_campaign_shard_v2.py")),"--inside","--batch",str(out),"--output",str(audit)]
   code=execute(cmd,out/"independent-audit.log",300,lambda: None)
   a=document(audit/"report.json")
   if code or a["plan_binding"]!=shard["binding"] or a["statuses"].get("audit_failed"):raise ValueError("Independent audit failed")
   return dict(binding=shard["binding"],output=str(out),audit=str(audit),audit_sha256=sha(audit/"report.json"),
    statuses=a["statuses"],generations=a["generations"],http_attempts=a["http_attempts"],status="audited_budget_stopped" if error else "audited_terminal")
  pool=ThreadPoolExecutor(max_workers=4)
  try:
   pending={pool.submit(worker,ref) for ref in plan["shards"]};last_storage=0
   while pending:
    full=time.monotonic()-last_storage>=10;boundary(full)
    if full:last_storage=time.monotonic()
    done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
    for future in done:
     row=future.result();rows.append(row);dump(output/"progress.json",rows);print(json.dumps(row),flush=True)
   if not all(sha(Path(p))==h for p,h in plan["compiler_guard"].items()):raise ValueError("Compiler changed")
   if sum(r["generations"] for r in rows)>plan["maximum_generations"] or sum(r["http_attempts"] for r in rows)>plan["maximum_http_attempts"]:raise ValueError("Call budget")
  except BaseException as e:
   failure=type(e).__name__+": "+str(e);stop.set();raise
  finally:
   stop.set();pool.shutdown(wait=True,cancel_futures=True)
   dump(output/"report.json",sealed(dict(format="poseidon-agent-repair-queue-run-r178",plan_binding=binding,rows=rows,
    seconds=plan["prior_seconds"]+time.monotonic()-started,prior_seconds=plan["prior_seconds"],failure=failure,queue_finished=len(rows)==4,uncertain_calls_possible=bool(failure),stage2_complete=False)))
 return 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","check","run"))
 p.add_argument("--proposal",type=Path);p.add_argument("--authorization",type=Path)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--approve-live-binding");p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 if a.mode=="prepare" and not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=300)
 if a.mode=="prepare":
  if a.output.exists():raise ValueError("Fresh preparation only")
  if a.proposal is None or a.authorization is None:p.error("Approved proposal and authorization required")
  plan=build(a.proposal,a.authorization,a.output);a.output.mkdir();dump(a.output/"plan.json",plan)
  print(json.dumps(dict(binding=plan["binding"],tasks=19,shards=4,new_paid_calls=0)));return 0
 plan=document(a.output/"plan.json")
 if a.mode=="run" and a.approve_live_binding!=plan["binding"]:p.error("Exact binding required")
 if not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=43300 if a.mode=="run" else 300)
 return run(a.output,plan["binding"] if a.mode=="check" else a.approve_live_binding,a.mode=="check")
if __name__=="__main__":raise SystemExit(main())
