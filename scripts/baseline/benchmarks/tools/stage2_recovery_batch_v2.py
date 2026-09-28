"""Explicitly authorized recovery only; immutable old claims and separate results.

One pure Nix parent, up to ten serial shards, independent candidate sandboxes.
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

PROPOSAL_BINDING="9da74b49ef701a8f329e3c4fc3eee04de0c6f47704de459afff5bb88aca5aa14"
AUTHORIZATION_BINDING="030c94d81f8bd626b2034f0b5b49f3dd98ecccd0a42196d9ee97a25ce6b78b2a"
FORMAT="poseidon-authorized-recovery-queue-v2"

class ShardBudget(ValueError):
 pass

def tools_sources():
 return {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob("*.py")}

def document(path,limit=16*1024**2):
 p=Path(path)
 if p.is_symlink() or p.name==".env":raise ValueError("Unsafe document")
 v=strict_file(p,limit);check_binding(v);return v

def approved(proposal,authorization):
 p=document(proposal);a=document(authorization)
 if p["binding"]!=PROPOSAL_BINDING or a["binding"]!=AUTHORIZATION_BINDING:raise ValueError("Unapproved recovery scope")
 if a["proposal"]!=str(Path(proposal).resolve()) or a["proposal_sha256"]!=sha(Path(proposal)) or a["proposal_binding"]!=p["binding"]:raise ValueError("Authorization identity")
 if p["planned"]!=54 or len(p["cases"])!=54 or len({c["id"] for c in p["cases"]})!=54:raise ValueError("Recovery task count")
 for key,value in dict(maximum_generations=216,maximum_http_attempts=864,api_workers=10,native_workers=2,
   compile_jobs=2,link_jobs=1,max_repairs=3,provider_retries=3,per_shard_retained_mib=4096,
   total_retained_mib=32768,min_free_mib=4096,additional_campaign_wall_seconds=43200).items():
  if p[key]!=value:raise ValueError("Authorization budget changed")
 if runtime_sources()!=p["source_hashes"]:raise ValueError("Approved runtime drift")
 return p,a

def build(proposal,authorization,output):
 from workspace_paths import RESULTS
 from hecate_python_env import VENV
 from unified_graph_contract import validate_request,prepare
 p,a=approved(proposal,authorization)
 if output.is_symlink() or output.resolve().parent!=RESULTS.resolve():raise ValueError("Direct result output")
 old_index=RESULTS/"stage2-agent-campaign-v7-r130/index.json";idx=document(old_index)
 original={}
 for ref in idx["shards"]:
  f=old_index.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(f)!=ref["sha256"]:raise ValueError("Historical plan hash")
  for s in document(f)["cases"]:original[s["id"]]=s
 sources=runtime_sources();proposals=tools_sources();specs=[]
 startup_path=RESULTS/"stage2-recovery-startup-proof-r148.json";startup=document(startup_path)
 if startup["binding"]!="85ff18cb2f45dc2850ee32037843b8da3173bab3fca15fd060d88e2991af6f0b" or startup["api_calls"]!=0 or not startup["all_children_terminal"]:raise ValueError("Unproven startup recovery")
 for n,h in startup["parents"].items():
  if sha(Path(n))!=h:raise ValueError("Startup evidence changed")

 probe_path=RESULTS/"stage2-recovery-native-probe-r147/report.json";probe=document(probe_path)
 if probe["source_hashes"]!=sources or probe["passed"]!=2 or not probe["two_native_slots_observed"] or probe["min_available_mib"]<1024 or probe["new_paid_calls"]!=0:
  raise ValueError("Concurrent native proof missing")
 if probe["runner_sha256"]!=sha(Path(__file__).with_name("probe_recovery_native.py")):raise ValueError("Native proof runner drift")
 for name,h in probe["parents"].items():
  if sha(Path(name))!=h:raise ValueError("Native proof evidence changed")

 # Focused diagnostics first, then historical interruptions; no extra successful controls.
 ordered=sorted(p["cases"],key=lambda c:(c["guidance"]!="explicit-v4",c["id"]))
 for c in ordered:
  s=dict(original[c["id"]]);r=c["request"];validate_request(r);old=s["request"]
  if c["previous_request_id"]!=old["request_id"]:raise ValueError("Original request identity")
  if c["guidance"]=="explicit-v3" and r!=old:raise ValueError("Recovery request changed")
  kw=dict(construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"),
   helper_profile=s.get("helper_profile"),helper_exercise=s.get("required_helpers"),chunk_period=s.get("chunk_period"),
   generation_guidance=c["guidance"])
  if prepare(s["model"],old["compiler_profile_sha256"],old["compiler_configuration"],s.get("exercise"),**kw)!=r:
   raise ValueError("Proposed request no longer reproducible")
  args=list(s["candidate_arguments"]);pos=args.index("--unified-guidance")
  if args[pos+1]!="explicit-v3":raise ValueError("Frozen argument version")
  args[pos+1]=c["guidance"]
  s.update(request=r,request_id=r["request_id"],candidate_arguments=args,
   original_evaluation_identity=s["evaluation_identity"],original_status=c["prior_status"],
   original_runtime_source_sha256=idx["runtime_source_sha256"])
  s["evaluation_identity"]=identity(s,sources);specs.append(s)
 shards=[]
 for i in range(10):
  cases=specs[i::10];n=len(cases)
  shard=sealed(dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=cases,source_hashes=sources,
   proposal_files=proposals,paid_configuration=PAID,
   limits=dict(LIMITS,max_wall_seconds=43200,max_retained_mib=4096,maximum_generations=4*n,maximum_http_attempts=16*n),
   authorization_binding=a["binding"],recovery_proposal_binding=p["binding"],
   executable=str(VENV/"bin/python"),entrypoint=str(BASE/"run_candidate.py"),
   shared_scheduling=dict(api_workers=10,native_workers=2,version=1),
   old_claims_preserved=True,automatic_retry=False,stage2_complete=False))
  shards.append(dict(plan=shard,output=str(RESULTS/(output.name+"-shard-%02d"%i)),audit=str(RESULTS/(output.name+"-shard-%02d-audit"%i))))
 return sealed(dict(format=FORMAT,proposal=str(Path(proposal).resolve()),authorization=str(Path(authorization).resolve()),
  proposal_sha256=sha(Path(proposal)),authorization_sha256=sha(Path(authorization)),source_hashes=sources,proposal_files=proposals,
  startup_proof=str(startup_path),startup_proof_sha256=sha(startup_path),prior_seconds=startup["prior_seconds"],
  native_probe=str(probe_path),native_probe_sha256=sha(probe_path),
  shards=shards,planned=54,maximum_generations=216,maximum_http_attempts=864,max_wall_seconds=43200,
  max_retained_mib=32768,min_free_mib=4096,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,
  admission_available_mib=3072,stop_available_mib=512,automatic_restart=False,stage2_complete=False))

def verify(plan,output):
 check_binding(plan)
 expected=build(Path(plan["proposal"]),Path(plan["authorization"]),output)
 if plan!=expected:raise ValueError("Frozen recovery plan changed")
 return plan

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
  print(json.dumps(dict(validated=True,binding=binding,tasks=54,shards=10,new_paid_calls=0)),flush=True);return 0
 if (output/"dispatch.json").exists() or (output/"report.json").exists():raise ValueError("Already dispatched; no automatic retry")
 fd=os.open(RESULTS/"stage2-agent-paid-queue.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 campaign_fd=os.open(RESULTS/"stage2-agent-campaign-execution.lock",os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"a+") as lock,os.fdopen(campaign_fd,"a+") as campaign_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert_idle()
  for ref in plan["shards"]:
   for k in ("output","audit"):
    if Path(ref[k]).exists():raise ValueError("Existing shard artifacts")
   for s in ref["plan"]["cases"]:
    if (RESULTS/"stage2-recovery-execution-claims-r147"/(s["evaluation_identity"]+".json")).exists():raise ValueError("Recovery claim already exists")
    oldclaim=RESULTS/"stage2-agent-evaluation-claims"/(s["evaluation_identity"]+".json")
    startup=document(Path(plan["startup_proof"]))
    proven={r["evaluation_identity"]:r for r in startup["rows"]}
    if oldclaim.exists() and (s["evaluation_identity"] not in proven or str(oldclaim)!=proven[s["evaluation_identity"]]["claim"]):raise ValueError("Unproven earlier claim")
  if size_bytes([RESULTS])+10*1024**3>32*1024**3:raise ValueError("Insufficient aggregate candidate headroom")
  if available_mib()<3072:raise ValueError("Insufficient admission memory")
  stop=threading.Event();guard=threading.Lock();active={};rows=[];failure=None
  def boundary(storage=False):
   if plan["prior_seconds"]+time.monotonic()-started>=43200:raise TimeoutError("Recovery total time budget")
   if available_mib()<512:raise ValueError("Available memory reserve")
   stat=os.statvfs(RESULTS)
   if stat.f_bavail*stat.f_frsize<4096*1024**2:raise ValueError("Disk free reserve")
   if runtime_sources()!=plan["source_hashes"] or tools_sources()!=plan["proposal_files"]:raise ValueError("Runtime/tool source drift")
   if sha(Path(plan["proposal"]))!=plan["proposal_sha256"] or sha(Path(plan["authorization"]))!=plan["authorization_sha256"]:raise ValueError("Authorization drift")
   if storage and size_bytes([RESULTS])>=32768*1024**2:raise ValueError("Aggregate retained space budget")
   if sha(Path(plan["native_probe"]))!=plan["native_probe_sha256"]:raise ValueError("Native proof drift")
   if sha(Path(plan["startup_proof"]))!=plan["startup_proof_sha256"]:raise ValueError("Startup proof drift")
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
   completed=[];folders=set();error=None;begin=time.monotonic()
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
     acquired,_=claim(RESULTS/"stage2-recovery-execution-claims-r147",s,plan["source_hashes"],out,shard["binding"])
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
  pool=ThreadPoolExecutor(max_workers=10)
  try:
   pending={pool.submit(worker,ref) for ref in plan["shards"]};last_storage=0
   while pending:
    full=time.monotonic()-last_storage>=10;boundary(full)
    if full:last_storage=time.monotonic()
    done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
    for future in done:
     row=future.result();rows.append(row);dump(output/"progress.json",rows);print(json.dumps(row),flush=True)
   if sum(r["generations"] for r in rows)>216 or sum(r["http_attempts"] for r in rows)>864:raise ValueError("Call budget")
  except BaseException as e:
   failure=type(e).__name__+": "+str(e);stop.set();raise
  finally:
   stop.set();pool.shutdown(wait=True,cancel_futures=True)
   dump(output/"report.json",sealed(dict(format="poseidon-recovery-queue-run-v1",plan_binding=binding,rows=rows,
    seconds=plan["prior_seconds"]+time.monotonic()-started,prior_seconds=plan["prior_seconds"],failure=failure,queue_finished=len(rows)==10,uncertain_calls_possible=bool(failure),stage2_complete=False)))
 return 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("prepare","check","run"))
 p.add_argument("--proposal",type=Path);p.add_argument("--authorization",type=Path)
 p.add_argument("--output",type=Path,required=True);p.add_argument("--approve-live-binding");p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 if a.mode=="prepare":
  if a.output.exists():raise ValueError("Fresh preparation only")
  if a.proposal is None or a.authorization is None:p.error("Approved proposal and authorization required")
  plan=build(a.proposal,a.authorization,a.output);a.output.mkdir();dump(a.output/"plan.json",plan)
  print(json.dumps(dict(binding=plan["binding"],tasks=54,shards=10,new_paid_calls=0)));return 0
 plan=document(a.output/"plan.json")
 if a.mode=="run" and a.approve_live_binding!=plan["binding"]:p.error("Exact binding required")
 if not a.inside:
  from hecate_python_env import VENV,enter_nix
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=43300 if a.mode=="run" else 300)
 return run(a.output,plan["binding"] if a.mode=="check" else a.approve_live_binding,a.mode=="check")
if __name__=="__main__":raise SystemExit(main())
