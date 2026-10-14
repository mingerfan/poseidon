"""Bounded sequential paid-shard queue. One pinned Nix environment; no implicit retry."""
import argparse,fcntl,json,os,shlex,subprocess,sys,time
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha,check_binding
from campaign_live_state import sealed,exclusive_json
from benchmark_runner import strict_file,dump
from run_stage2_guidance_retest import stop_child,size_bytes
from semantic_benchmark_execution import runtime_sources

def checked_plan(path):
 plan=strict_file(path,8*1024**2);check_binding(plan)
 if runtime_sources()!=plan["source_hashes"]:raise ValueError("Queue runtime drift")
 for name,hsh in plan["proposal_files"].items():
  if sha(ROOT/name)!=hsh:raise ValueError("Shard implementation drift")
 if "recovery" in plan:raise ValueError("Recovered shard requires explicit manual resume, not queue restart")
 return plan

def prepare(index_path,output,first,last):
 from workspace_paths import RESULTS
 if output.exists() or output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result queue required")
 index=strict_file(index_path,16*1024**2);check_binding(index)
 if not 0<=first<=last<len(index["shards"]):raise ValueError("Explicit inclusive shard range required")
 refs=[];total_tasks=total_generations=total_http=0;requests=0
 for position,ref in enumerate(index["shards"]):
  if not first<=position<=last:continue
  p=index_path.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(p)!=ref["sha256"]:raise ValueError("Queue shard changed")
  plan=checked_plan(p)
  if not 1<=len(plan["cases"])<=48 or plan["limits"]["concurrency"]!=1:raise ValueError("Serial <=48 task shards required")
  refs.append(dict(plan=str(p),sha256=sha(p),binding=plan["binding"],tasks=len(plan["cases"]),
   output=str(RESULTS/(output.name+"-shard-%03d"%position)),
   audit=str(RESULTS/(output.name+"-shard-%03d-audit"%position)),
   max_wall_seconds=plan["limits"]["max_wall_seconds"]))
  total_tasks+=len(plan["cases"]);total_generations+=plan["limits"]["maximum_generations"];total_http+=plan["limits"]["maximum_http_attempts"]
  requests+=sum(len(json.dumps(s["request"]).encode()) for s in plan["cases"])
 value=sealed(dict(format="poseidon-paid-sequential-queue-v1",shards=refs,
  source_hashes=runtime_sources(),source_index=str(index_path),source_index_sha256=sha(index_path),
  runner_sha256=sha(Path(__file__)),planned_tasks=total_tasks,
  maximum_generations=total_generations,maximum_http_attempts=total_http,
  max_wall_seconds=sum(r["max_wall_seconds"]+390 for r in refs),
  max_retained_mib=8192,min_free_mib=4096,monetary_cap=None,
  outbound="Frozen public models, constants, layouts, DSL rules, response format and constrained diagnostics",
  approximate_frozen_request_bytes=requests,concurrency=1,compile_jobs=2,link_jobs=1,
  automatic_candidate_retry=False,automatic_shard_restart=False,stage2_complete=False))
 output.mkdir();dump(output/"plan.json",value)
 print(json.dumps({k:value[k] for k in ("binding","planned_tasks","maximum_generations","maximum_http_attempts","max_wall_seconds","max_retained_mib","monetary_cap")}))
 return value

def verify_queue(plan):
 check_binding(plan)
 if plan["runner_sha256"]!=sha(Path(__file__)):raise ValueError("Queue runner drift")
 if runtime_sources()!=plan["source_hashes"]:raise ValueError("Queue runtime drift")
 if sha(Path(plan["source_index"]))!=plan["source_index_sha256"]:raise ValueError("Queue index changed")
 if plan["concurrency"]!=1 or plan["max_retained_mib"]!=8192 or plan["min_free_mib"]!=4096:raise ValueError("Queue resource policy changed")
 from workspace_paths import RESULTS
 if not 1<=len(plan["shards"])<=43:raise ValueError("Queue shard count")
 if len({r["binding"] for r in plan["shards"]})!=len(plan["shards"]):raise ValueError("Duplicate queue shard")
 totals=dict(tasks=0,generations=0,http=0,seconds=0)
 paths=set()
 for ref in plan["shards"]:
  path=Path(ref["plan"]);child=checked_plan(path)
  if sha(path)!=ref["sha256"] or child["binding"]!=ref["binding"]:raise ValueError("Queue plan drift")
  if ref["tasks"]!=len(child["cases"]) or ref["max_wall_seconds"]!=child["limits"]["max_wall_seconds"]:raise ValueError("Queue child budget drift")
  for key in ("output","audit"):
   q=Path(ref[key])
   if q.is_symlink() or q.resolve().parent!=RESULTS.resolve() or str(q) in paths:raise ValueError("Queue output boundary")
   paths.add(str(q))
  totals["tasks"]+=len(child["cases"]);totals["generations"]+=child["limits"]["maximum_generations"]
  totals["http"]+=child["limits"]["maximum_http_attempts"];totals["seconds"]+=child["limits"]["max_wall_seconds"]+390
 if (plan["planned_tasks"]!=totals["tasks"] or plan["maximum_generations"]!=totals["generations"]
     or plan["maximum_http_attempts"]!=totals["http"] or plan["max_wall_seconds"]!=totals["seconds"]):raise ValueError("Queue total budget drift")

def run(output,binding,resume):
 from hecate_python_env import VENV
 from workspace_paths import RESULTS
 if output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Direct queue output required")
 plan=strict_file(output/"plan.json",1024**2);verify_queue(plan)
 if binding!=plan["binding"]:raise ValueError("Exact approved queue binding required")
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned pure Python required")
 from run_stage2_agent_campaign_v3 import validate_plan
 for ref in plan["shards"]:validate_plan(strict_file(Path(ref["plan"]),8*1024**2),ref["binding"])
 fd=os.open(RESULTS/"stage2-agent-paid-queue.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=[];prior=0.
  if (output/"report.json").exists():
   if not resume:raise ValueError("Existing queue requires explicit resume")
   old=strict_file(output/"report.json",1024**2);check_binding(old)
   if old["plan_binding"]!=binding:raise ValueError("Queue resume binding")
   rows=old["rows"];prior=old["seconds"]
   import math
   if type(prior) not in (int,float) or not math.isfinite(prior) or prior<0:raise ValueError("Queue elapsed budget")
   if len({r["binding"] for r in rows})!=len(rows) or not {r["binding"] for r in rows}<={r["binding"] for r in plan["shards"]}:raise ValueError("Queue checkpoint identities")
   refs={r["binding"]:r for r in plan["shards"]}
   for row in rows:
    ref=refs[row["binding"]]
    if row["output"]!=ref["output"] or row["audit"]!=ref["audit"]:raise ValueError("Queue checkpoint location")
    audit_path=Path(row["audit"])/"report.json"
    if row["status"]!="audited_terminal" or audit_path.is_symlink() or sha(audit_path)!=row["audit_sha256"]:raise ValueError("Uncertain queue checkpoint")
    result=strict_file(audit_path,16*1024**2);check_binding(result)
    if result["plan_binding"]!=row["binding"] or result["statuses"]!=row["statuses"]:raise ValueError("Changed audited checkpoint")
    for name,hsh in result["parents"].items():
     q=Path(name)
     if q.is_symlink() or not q.resolve().is_relative_to(RESULTS.resolve()) or sha(q)!=hsh:raise ValueError("Queue audit evidence changed")
  elif resume:raise ValueError("No terminal queue checkpoint")
  if prior>=plan["max_wall_seconds"]:raise TimeoutError("Queue cumulative wall budget exhausted")
  done={x["binding"] for x in rows}
  for ref in plan["shards"]:
   if ref["binding"] not in done and ((output/ref["binding"]).exists() or Path(ref["output"]).exists()):raise ValueError("Uncertain launched shard: inspect before any restart")
  started=time.monotonic();child=None;failure=None
  def boundary():
   if prior+time.monotonic()-started>=plan["max_wall_seconds"]:raise TimeoutError("Queue cumulative wall budget")
   stat=os.statvfs(RESULTS)
   if stat.f_bavail*stat.f_frsize<plan["min_free_mib"]*1024**2:raise ValueError("Queue free disk reserve")
   paths=[output]
   for ref in plan["shards"]:
    b=Path(ref["output"]);paths.extend([b,Path(ref["audit"])])
    progress=b/"progress.json"
    if progress.exists():
     paths.extend(Path(x["evidence"]) for x in strict_file(progress,8*1024**2) if "evidence" in x)
    for log in b.glob("*/run.log"):
     from run_stage2_guidance_retest import evidence_path
     folder=evidence_path(log,RESULTS)
     if folder:paths.append(folder)
   if size_bytes(paths)>plan["max_retained_mib"]*1024**2:raise ValueError("Queue retained artifact budget")
   if runtime_sources()!=plan["source_hashes"] or sha(Path(__file__))!=plan["runner_sha256"]:raise ValueError("Queue source drift")
  def execute(command,log,seconds):
   nonlocal child
   with log.open("x") as f:
    child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic()
    while child.poll() is None:
     boundary()
     if time.monotonic()-start>=seconds:raise TimeoutError("Queue child time boundary")
     try:child.wait(timeout=1)
     except subprocess.TimeoutExpired:pass
    return child.returncode
  try:
   for ref in plan["shards"]:
    if ref["binding"] in done:continue
    boundary()
    marker=output/ref["binding"];marker.mkdir()
    command=[str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("run_stage2_agent_campaign_v3.py")),
     "--plan",ref["plan"],"--approve-live-binding",ref["binding"],"--output",ref["output"],"--inside"]
    exclusive_json(marker/"launch.json",sealed(dict(command=command,queue_binding=binding,automatic_retry=False)))
    code=execute(command,marker/"run.log",ref["max_wall_seconds"]+60)
    terminal=strict_file(Path(ref["output"])/"report.json",8*1024**2);check_binding(terminal)
    if terminal["plan_binding"]!=ref["binding"] or terminal.get("failure") or terminal["not_run_or_interrupted"]:raise ValueError("Shard stopped or uncertain; retain queue")
    audit=[str(VENV/"bin/python"),"-B",str(Path(__file__).with_name("audit_stage2_campaign_shard_v2.py")),
     "--batch",ref["output"],"--output",ref["audit"],"--inside"]
    audit_code=execute(audit,marker/"audit.log",300)
    result=strict_file(Path(ref["audit"])/"report.json",16*1024**2);check_binding(result)
    if audit_code or result["statuses"].get("audit_failed") or result["plan_binding"]!=ref["binding"]:raise ValueError("Independent audit failed")
    row=dict(binding=ref["binding"],status="audited_terminal",runner_exit=code,output=ref["output"],audit=ref["audit"],
     audit_sha256=sha(Path(ref["audit"])/"report.json"),statuses=result["statuses"],generations=result["generations"],http_attempts=result["http_attempts"])
    rows.append(row);dump(output/"progress.json",rows)
    print(json.dumps(row),flush=True)
  except BaseException as error:
   failure=type(error).__name__+": "+str(error)
   if child:stop_child(child)
   raise
  finally:
   if child:stop_child(child)
   report=sealed(dict(format="poseidon-paid-queue-run-v1",plan_binding=binding,rows=rows,
    seconds=prior+time.monotonic()-started,failure=failure,queue_finished=len(rows)==len(plan["shards"]),
    stage2_complete=False,uncertain_calls_possible=bool(failure)))
   dump(output/"report.json",report)
 return 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","run"))
 p.add_argument("--index",type=Path);p.add_argument("--first-shard",type=int);p.add_argument("--last-shard",type=int)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--approve-live-binding");p.add_argument("--resume",action="store_true")
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 if a.mode=="prepare":
  if a.index is None or a.first_shard is None or a.last_shard is None:p.error("Explicit index/range required")
  prepare(a.index,a.output,a.first_shard,a.last_shard);return 0
 plan=strict_file(a.output/"plan.json",1024**2);verify_queue(plan)
 if not a.approve_live_binding or a.approve_live_binding!=plan["binding"]:p.error("Exact binding required")
 if not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=plan["max_wall_seconds"]+90)
 return run(a.output,a.approve_live_binding,a.resume)
if __name__=="__main__":raise SystemExit(main())
