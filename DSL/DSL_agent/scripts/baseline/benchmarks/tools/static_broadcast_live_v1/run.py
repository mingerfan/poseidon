"""Frozen normalized source requests through the existing Application. Explicit approval; no auto-resume."""
import argparse,ctypes,fcntl,importlib.util,json,math,os,shlex,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
from scope import ROOT,PROPOSAL,CASES,canonical,digest,sha,read,require,verify,authority,totals,terminal,RUN,OPTIONS,PROFILE,tools,owned_lock
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
HELPER=Path(__file__).resolve().parent/"package_checks.py"

# Fixed live execution contract; capture producer files are frozen separately.
LOCK = RESULTS/"agent-static-broadcast-live-v1.lock"
PUBLIC_FIELDS={"format","proposal_binding","proposal_sha256","cases","provider","budget",
 "tool_hashes","approved","provider_calls","validation"}
PLAN_FIELDS=PUBLIC_FIELDS|{"approval_record","approval_sha256","created","deadline_wall",
 "initial_result_directories","owner","binding"}
CASE_FIELDS={"format","case","plan_binding","started_wall","deadline_wall",
 "initial_result_directories","binding"}
def finite_number(value):
 return type(value) in (int,float) and math.isfinite(value)
def safe_inventory(value):
 return type(value) is list and all(type(n) is str and n not in ("",".","..") and Path(n).name==n for n in value) and len(value)==len(set(value))
def public_plan(proposal):
 return dict(format="poseidon-source-live-tool-plan-v1",proposal_binding=proposal["binding"],
  proposal_sha256=sha(PROPOSAL),cases=list(CASES),provider=proposal["provider"],budget=proposal["budget"],
  tool_hashes=tool_hashes(),approved=False,provider_calls=0,
  validation=["source_capture","numerical_qualification","new_input_execution","package_cache_reuse"])
def process_info(pid):
 try:
  fields=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
  return dict(pid=int(pid),parent=int(fields[1]),group=int(fields[2]),start=int(fields[19]),state=fields[0])
 except (OSError,ValueError,IndexError):return None
def process_identity(pid):
 info=process_info(pid);require(info is not None,"execution_owner_missing")
 return {key:info[key] for key in ("pid","start")}
def same_process(identity):
 info=process_info(identity["pid"])
 return info is not None and info["start"]==identity["start"] and info["state"] not in ("Z","X")
def owner_current(owner):
 require(type(owner) is dict and set(owner)=={"pid","start"} and
  all(type(v) is int and v>0 for v in owner.values()) and same_process(owner),"execution_owner_identity")
 pid=os.getpid();seen=set()
 while pid not in seen:
  if pid==owner["pid"]:break
  seen.add(pid);info=process_info(pid)
  require(info is not None and info["parent"]>0,"execution_owner_ancestry")
  pid=info["parent"]
 else:raise ValueError("execution_owner_ancestry")
 require(pid==owner["pid"],"execution_owner_ancestry")
 stat=LOCK.stat();device=f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}:{stat.st_ino}"
 held=False
 for line in Path("/proc/locks").read_text().splitlines():
  fields=line.split()
  if len(fields)>=6 and fields[1:4]==["FLOCK","ADVISORY","WRITE"]:
   held=held or (fields[4]==str(owner["pid"]) and
    tuple(int(x,16) for x in fields[5].split(":")[:2])==
    (os.major(stat.st_dev),os.minor(stat.st_dev)) and int(fields[5].split(":")[2])==stat.st_ino)
 require(held,"execution_owner_lock")
def approval_path(path):
 path=Path(path)
 require(path.is_absolute() and path==path.resolve() and path.is_file() and
  not any(p.is_symlink() for p in (path,*path.parents)) and
  (path.is_relative_to(ROOT) or path.is_relative_to(RESULTS)) and
  path.suffix==".json" and not any(n.startswith(".") for n in path.parts),"approval_record_path")
 return path
def bind_lock(stream,plan):
 payload=dict(format="poseidon-live-owner-lock-v2",owner=plan["owner"],plan_binding=plan["binding"],
  plan_sha256=sha(RUN/"plan.json"),created=plan["created"],deadline_wall=plan["deadline_wall"])
 stream.seek(0);stream.truncate();stream.write(canonical(payload).decode());stream.flush();os.fsync(stream.fileno())
def validate_plan(plan,proposal):
 require(type(plan) is dict and set(plan)==PLAN_FIELDS,"execution_plan_fields")
 require(plan["binding"]==digest({k:v for k,v in plan.items() if k!="binding"}),"execution_plan_binding")
 expected=public_plan(proposal)
 require(all(plan[k]==v for k,v in expected.items() if k!="approved") and plan["approved"] is True,
  "execution_preserves_approved_plan")
 require(finite_number(plan["created"]) and finite_number(plan["deadline_wall"]) and
  plan["deadline_wall"]==plan["created"]+proposal["budget"]["batch_wall_seconds"] and
  plan["created"]<=time.time(),"original_batch_deadline")
 require(safe_inventory(plan["initial_result_directories"]),"execution_initial_inventory")
 approval_path(plan["approval_record"])
 owner_current(plan["owner"])
 claim=read(LOCK)
 require(claim==dict(format="poseidon-live-owner-lock-v2",owner=plan["owner"],plan_binding=plan["binding"],
  plan_sha256=sha(RUN/"plan.json"),created=plan["created"],deadline_wall=plan["deadline_wall"]) and
  read(RUN/"plan.json")==plan,"execution_owner_plan_binding")
def case_boundary(output,name,plan,proposal):
 require(name in CASES,"reviewed_cases")
 folder=Path(output)/name;value=read(folder/"case-boundary.json")
 require(type(value) is dict and set(value)==CASE_FIELDS and value["format"]=="poseidon-live-case-boundary-v2" and
  value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}) and
  value["case"]==name and value["plan_binding"]==plan["binding"],"case_boundary_identity")
 require(finite_number(value["started_wall"]) and finite_number(value["deadline_wall"]) and
  plan["created"]<=value["started_wall"]<=time.time() and
  value["deadline_wall"]==min(plan["deadline_wall"],value["started_wall"]+proposal["budget"]["case_wall_seconds"]),
  "original_case_deadline")
 require(safe_inventory(value["initial_result_directories"]) and RUN.name in value["initial_result_directories"],
  "case_initial_inventory")
 marker=folder/"case-started.json"
 if marker.exists():
  started=read(marker)
  require(started.get("case")==name and started.get("plan_binding")==plan["binding"] and
   started.get("case_boundary_binding")==value["binding"] and
   started.get("binding")==digest({k:v for k,v in started.items() if k!="binding"}),"case_started_identity")
 return value
def enable_subreaper():
 libc=ctypes.CDLL(None,use_errno=True)
 require(libc.prctl(36,1,0,0,0)==0,"process_supervision_unavailable")
class OwnedTree:
 """Linux-only owned descendants, including setsid/double-fork and PID reuse."""
 def __init__(self):
  enable_subreaper();self.root=os.getpid();self.owned={};self.stop_event=threading.Event()
  infos=self.snapshot();self.baseline={p for p in infos if self.descends(p,self.root,infos)}
  self.thread=None
 @staticmethod
 def snapshot():
  return {int(p.name):info for p in Path("/proc").iterdir() if p.name.isdigit()
   for info in (process_info(int(p.name)),) if info is not None}
 @staticmethod
 def descends(pid,root,infos):
  seen=set()
  while pid in infos and pid not in seen:
   seen.add(pid);pid=infos[pid]["parent"]
   if pid==root:return True
  return False
 def capture(self):
  infos=self.snapshot()
  for pid,info in infos.items():
   if pid not in self.baseline and pid!=self.root and self.descends(pid,self.root,infos):
    self.owned.setdefault(pid,info["start"])
  return infos
 def start(self):
  def watch():
   while not self.stop_event.wait(.05):self.capture()
  self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
  return self
 def living(self):
  infos=self.capture()
  return [pid for pid,start in self.owned.items() if pid in infos and
   infos[pid]["start"]==start and infos[pid]["state"] not in ("Z","X")]
 def signal(self,sig):
  for pid in self.living():
   info=process_info(pid)
   if info is not None and info["start"]==self.owned[pid]:
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
 def finish(self,exclude_reap=()):
  self.stop_event.set()
  if self.thread is not None:self.thread.join(timeout=2)
  had_background=bool(self.living())
  for sig,seconds in ((signal.SIGTERM,.4),(signal.SIGKILL,2.)):
   end=time.monotonic()+seconds
   while self.living() and time.monotonic()<end:
    self.signal(sig);time.sleep(.02)
  require(not self.living(),"owned_background_survived")
  for pid,start in tuple(self.owned.items()):
   info=process_info(pid)
   if pid not in exclude_reap and info is not None and info["start"]==start and info["state"]=="Z":
    try:os.waitpid(pid,os.WNOHANG)
    except ChildProcessError:pass
  return had_background
def checked_receipt(folder,name,plan,case):
 path=Path(folder)/"case-report.json"
 require(path.is_file(),"missing_case_receipt")
 row=read(path);marker=read(Path(folder)/"case-started.json")
 require(type(row) is dict and row.get("format")=="poseidon-source-live-case-report-v2" and
  row.get("case")==name and row.get("request_id")==case["request_id"] and
  row.get("plan_binding")==plan["binding"] and row.get("case_started_binding")==marker["binding"] and
  row.get("status") in ("passed","failed") and row.get("new_agent_generation") is True,"case_receipt_identity")
 return row
def interrupted(*_):
 raise InterruptedError("execution_interrupted")

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_bytes(canonical(value));os.chmod(tmp,0o600);tmp.replace(path)
def tool_hashes():
 return {str(ROOT/p):h for p,h in tools().items()}
def preview():
 return public_plan(verify(read(PROPOSAL)))
def measure(path):
 return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(path)],text=True,timeout=30).split()[0])
  for arg in ("--apparent-size","--count-links"))
def active_project_command(arguments):
 # Supervisors remain active between individual candidate/HTTP workers.
 # Match only project-owned execution paths; read-only --plan is allowed.
 args=tuple(os.fsdecode(x) for x in arguments if x)
 paths={str(Path(__file__).resolve()),str(BASE/"candidate_worker.py"),
  str(BASE/"run_candidate.py"),str(BASE/"deepseek_http_worker.py"),str(BASE/"candidate_trace.py")}
 executing=any(flag in args for flag in ("--execute","--inside","--case"))
 for argument in args:
  path=Path(argument)
  if argument in paths and (argument!=str(Path(__file__).resolve()) or executing):return True
  if path.name in ("hecate-opt","poseidon-agent-cpu","poseidon_agent_cpu"):return True
  if executing and path.is_absolute() and path.is_relative_to(BASE/"benchmarks/tools") and path.suffix==".py":return True
 return False
def own_timeout_command(arguments,entry_argv):
 args=tuple(os.fsdecode(x)for x in arguments if x)
 entry=tuple(os.fsdecode(x)for x in entry_argv)
 if not entry or not args or Path(args[0]).name!="timeout":return False
 try:location=Path(entry[0]).resolve()
 except OSError:return False
 if not location.is_relative_to(BASE/"benchmarks/tools"):return False
 if len(args)<len(entry)+4 or args[-len(entry):]!=entry:return False
 if args[-len(entry)-1]!="-B":return False
 if Path(args[-len(entry)-2]).name not in("python3",Path(sys.executable).name):return False
 return True
def ancestor_identities():
 identities={};initial=os.getppid();pid=initial;seen=set()
 while pid>1 and pid not in seen:
  seen.add(pid);info=process_info(pid)
  if info is None or info["state"]in("Z","X"):return {}
  identities[pid]=(info["start"],info["parent"]);pid=info["parent"]
 if os.getppid()!=initial:return {}
 for pid,(start,parent)in identities.items():
  info=process_info(pid)
  if info is None or (info["start"],info["parent"])!=(start,parent):return {}
 return identities
def no_active():
 ancestors=ancestor_identities()
 for p in Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=(p/"cmdline").read_bytes().split(b"\0")
  except OSError:continue
  if int(p.name)in ancestors and own_timeout_command(a,sys.argv):
   info=process_info(int(p.name))
   try:executable=(p/"exe").resolve(strict=True)
   except OSError:executable=None
   trusted_timeout=executable is not None and executable.name=="timeout"and(
    executable==Path("/usr/bin/timeout").resolve()or executable.is_relative_to(Path("/nix/store")))
   if trusted_timeout and info is not None and (info["start"],info["parent"])==ancestors[int(p.name)]:continue
  require(not active_project_command(a),"active_project_execution")
def execute(output,approval):
 require(approval is not None,"approval_record_required")
 require(Path(output)==RUN,"fixed_campaign_root")
 require(Path(output).is_absolute() and Path(output)==Path(output).resolve() and
  Path(output).parent==RESULTS and not Path(output).exists(),"fresh_campaign_root")
 approval=approval_path(approval)
 proposal=verify(read(PROPOSAL));authority(read(approval),proposal,sha(PROPOSAL));no_active()
 with owned_lock(LOCK) as lock_stream:
  public=preview();initial=sorted(p.name for p in RESULTS.iterdir())
  require(measure(RESULTS)<=proposal["budget"]["global_result_bytes"] and
   shutil.disk_usage(WORK).free>=proposal["budget"]["reserve_bytes"],"artifact_or_disk_boundary")
  Path(output).mkdir(mode=0o700)
  created=time.time()
  plan=dict(public,approved=True,approval_record=str(approval),owner=process_identity(os.getpid()),
   approval_sha256=sha(approval),created=created,initial_result_directories=initial,
   deadline_wall=created+proposal["budget"]["batch_wall_seconds"])
  plan["binding"]=digest(plan);write(Path(output)/"plan.json",plan);bind_lock(lock_stream,plan)
  sys.path.insert(0,str(BASE))
  from hecate_python_env import VENV,enter_nix
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
   [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",str(output)])
  tree=OwnedTree().start()
  try:
   code=enter_nix(command,seconds=max(1,int(plan["deadline_wall"]-time.time())),keep_env=(),environment="poseidon-cpu")
  finally:
   background=tree.finish()
  require(not background,"nix_background_exit")
  return code

def boundary(output,plan,case=None):
 require(Path(output)==RUN,"fixed_campaign_root")
 proposal=verify(read(PROPOSAL));validate_plan(plan,proposal)
 authority(read(plan["approval_record"]),proposal,sha(PROPOSAL))
 require(sha(plan["approval_record"])==plan["approval_sha256"],"approval_record_changed")
 require(time.time()<plan["deadline_wall"],"batch_time_boundary")
 require(shutil.disk_usage(WORK).free>=proposal["budget"]["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=proposal["budget"]["global_result_bytes"],"global_artifact_boundary")
 added=[p for p in RESULTS.iterdir() if p.name not in plan["initial_result_directories"]]
 require(sum(measure(p) for p in added)<=proposal["budget"]["new_batch_bytes"],"batch_artifact_boundary")
 if case is not None:
  supervisor_current(output,plan)
  limit=case_boundary(output,case,plan,proposal)
  added=[p for p in RESULTS.iterdir() if p.name not in limit["initial_result_directories"]]
  require(measure(Path(output)/case)+sum(measure(p) for p in added)<=
   proposal["budget"]["per_case_bytes"],"case_artifact_boundary")
  require(time.time()<limit["deadline_wall"],"case_time_boundary")
 # Expensive source/du checks must not permit a post after the deadline.
 require(time.time()<plan["deadline_wall"],"batch_time_boundary")
 return proposal
def states(output):
 result=[]
 for name in CASES:
  folder=Path(output)/name
  allowed=set();marker=folder/"task-id.json"
  if marker.exists():
   task=read(marker)["task_id"]
   require(type(task) is str and len(task)==32 and all(c in "0123456789abcdef" for c in task),"registered_task_identity")
   allowed.add(task)
  cache=folder/"cache.json"
  if cache.exists() and read(cache).get("status")=="passed":
   task=read(cache)["result"]["task_id"]
   require(type(task) is str and len(task)==32 and all(c in "0123456789abcdef" for c in task),"registered_cache_identity")
   allowed.add(task)
  actual={p.parent.name:p for p in (folder/"tasks").glob("*/state.json")}
  require(set(actual)==allowed,"registered_call_state_inventory")
  for task in sorted(allowed):
   p=actual[task]
   require(p.is_file() and p==p.resolve() and not p.is_symlink(),"registered_call_state_path")
   result.append(read(p))
 return result
def save(output,plan,rows,status=None):
 try:data=terminal(rows,states(output))
 except (ValueError,FileNotFoundError,KeyError):
  data=dict(status="stopped",passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]!="passed" for r in rows),
   not_run=len(CASES)-len(rows),generations=None,http_attempts=None,calls_resolved=False,accounting_integrity_unknown=True)
 if status is not None and not data.get("accounting_integrity_unknown"):data["status"]=status
 write(Path(output)/"report.json",dict(format="poseidon-source-live-report-v1",binding=plan["binding"],
  proposal_binding=plan["proposal_binding"],planned=len(CASES),rows=rows,cost_usd=None,
  new_agent_generation=True,automatic_restart=False,**data))
 return data
def reserve_execution(output,plan):
 value=dict(format="poseidon-live-supervisor-claim-v2",binding=plan["binding"],pid=os.getpid(),
  process_start=str(process_identity(os.getpid())["start"]),started_wall=time.time())
 with (Path(output)/"execution-claim.json").open("xb") as stream:
  os.chmod(stream.name,0o600);stream.write(canonical(value))
def supervisor_current(output,plan):
 claim=read(Path(output)/"execution-claim.json")
 require(claim.get("format")=="poseidon-live-supervisor-claim-v2" and claim.get("binding")==plan["binding"] and
  type(claim.get("pid")) is int and claim["pid"] in (os.getpid(),os.getppid()),"owned_case_supervisor")
 info=process_info(claim["pid"])
 require(info is not None and str(info["start"])==claim.get("process_start") and info["state"] not in ("Z","X"),
  "owned_case_supervisor_identity")
 return claim
def inside(output):
 sys.path.insert(0,str(BASE))
 from hecate_python_env import VENV
 require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"pinned_worker")
 output=Path(output);plan=read(output/"plan.json");proposal=boundary(output,plan)
 from application_backend import PoseidonBackend
 require(PoseidonBackend().identity()["tools"]==proposal["source_stage_identity"]["backend"] and PoseidonBackend().identity_manifest()==proposal["application_stage_identity"],"backend_identity_changed")
 reserve_execution(output,plan)
 rows=[];save(output,plan,rows,"running")
 try:
  for case in proposal["cases"]:
   name=case["id"];boundary(output,plan);require(totals(states(output))["calls_resolved"],"unresolved_previous_call")
   folder=output/name;folder.mkdir(mode=0o700)
   started=time.time()
   limit=dict(format="poseidon-live-case-boundary-v2",case=name,plan_binding=plan["binding"],started_wall=started,
    deadline_wall=min(plan["deadline_wall"],started+proposal["budget"]["case_wall_seconds"]),
    initial_result_directories=sorted(p.name for p in RESULTS.iterdir()))
   limit["binding"]=digest(limit);write(folder/"case-boundary.json",limit)
   allow=("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG")
   env={k:os.environ[k] for k in allow if k in os.environ}
   env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(BASE),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HOME="/nonexistent")
   with (folder/"worker.log").open("xb") as log:
    tree=OwnedTree().start();child=None
    try:
     child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--case",str(output),name],
     env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     while child.poll() is None:
      boundary(output,plan,name);time.sleep(3)
    finally:
     background=tree.finish(exclude_reap=() if child is None else (child.pid,))
     if child is not None:child.wait(timeout=5)
   if background:
    rows.append(dict(case=name,status="failed",failure_layer="environment",code="case_background_exit"))
    raise ValueError("case_background_exit")
   try:row=checked_receipt(folder,name,plan,case)
   except (ValueError,KeyError,FileNotFoundError) as error:
    rows.append(dict(case=name,status="failed",failure_layer="environment",code="missing_case_receipt"
     if not (folder/"case-report.json").exists() else "case_receipt_identity"))
    raise
   row["exit_code"]=child.returncode
   if child.returncode:row["status"]="failed"
   rows.append(row);save(output,plan,rows,"running");boundary(output,plan,name)
   print(json.dumps(dict(case=name,status=row["status"],**totals(states(output)))),flush=True)
   require(row.get("failure_layer") not in ("integrity","environment","resources") and
    totals(states(output))["calls_resolved"],"campaign_boundary_or_unresolved_call")
  data=save(output,plan,rows)
  return 0 if data["passed"]==len(CASES) else 1
 except BaseException:
  save(output,plan,rows,"stopped");raise
def reserve_case(folder,name,plan):
 require(name in CASES,"reviewed_cases")
 marker=Path(folder)/"case-started.json"
 with marker.open("xb") as stream:
  os.chmod(marker,0o600)
  value=dict(case=name,plan_binding=plan["binding"],case_boundary_binding=read(Path(folder)/"case-boundary.json")["binding"],started_wall=time.time())
  value["binding"]=digest(value);stream.write(canonical(value))

def one(output,name):
 require(name in CASES,"reviewed_cases")
 output=Path(output);plan=read(output/"plan.json");proposal=boundary(output,plan,name)
 case=next(c for c in proposal["cases"] if c["id"]==name)
 supervisor_current(output,plan)
 require(totals(states(output))["calls_resolved"],"unresolved_previous_call")
 reserve_case(output/name,name,plan)
 sys.path.insert(0,str(BASE))
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from application_backend import PoseidonBackend
 require(PoseidonBackend().identity()["tools"]==proposal["source_stage_identity"]["backend"] and PoseidonBackend().identity_manifest()==proposal["application_stage_identity"],"backend_identity_changed")
 app=application(WorkerContext.capture(output/name/"tasks"))
 row=dict(format="poseidon-source-live-case-report-v2",case=name,status="failed",new_agent_generation=True,
  request_id=case["request_id"],plan_binding=plan["binding"],
  case_started_binding=read(output/name/"case-started.json")["binding"])
 try:
  limits=dict(generations=4,http_attempts=16,wall_seconds=max(1,int(read(output/name/"case-boundary.json")["deadline_wall"]-time.time())),
   api_timeout_seconds=1200,max_tokens=384000)
  options=dict(OPTIONS)
  from unified_capacity_constants import request_version
  options["capacity_packing_version"]=request_version(read(case["full_request_file"]))
  source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(ROOT/case["source_manifest"]),trust_local_code=True)
  task=app.submit_source(source,resource_profile=PROFILE,options=options,limits=limits)
  write(output/name/"task-id.json",dict(task_id=task));row["task_id"]=task
  state=app.prepare_source(task)
  if state["status"]!="queued":
   row.update(result=state,failure_layer=(state.get("failure") or {}).get("layer","input"))
   write(output/name/"case-report.json",row);return 1
  meta=app._checked_task_metadata(task)
  from candidate_contract import request_file_bytes
  from public_weight_context import project
  require(read(case["full_request_file"])==meta["request"],"frozen_mathematical_request")
  require(digest(meta["request"]["model"])==case["model_sha256"],"frozen_model_and_reference_provenance")
  require(meta["identity"]==proposal["application_stage_identity"] and meta["identity"]["validation"]["sources"]==case["current_validation_sources"],
   "current_reference_sources")
  boundary(output,plan,name)
  from agent_credentials import load_api_key
  from deepseek_provider import Config,DeepSeekProvider,HTTPSTransport
  key=load_api_key(ROOT,provider="deepseek")
  config=Config(model="deepseek-flash",reasoning_effort="high",max_tokens=384000,max_calls=4,
   timeout_seconds=1200,service_provider="deepseek",stream=True,provider_retries=3,loopback_proxy_port=0)
  class BoundTransport(HTTPSTransport):
   def post(self,body,timeout):
    boundary(output,plan,name)
    return super().post(body,timeout)
  transport=BoundTransport(api_key=key,approved_request_id=case["request_id"],approved_config=config,enabled=True)
  key=""
  result=app.run(task,DeepSeekProvider(config,transport=transport));row["result"]=result
  receipt=app._dir(task)/"provider-transport.json"
  if receipt.exists():
   from application_transport_receipt import snapshot
   metrics=read(receipt)  # Billing metrics omit detailed transport observations.
   require(read(receipt)==snapshot(metrics),"transport_receipt_binding")
   row["transport_receipt"]=str(receipt);row["transport_receipt_sha256"]=sha(receipt)
  if result["status"]!="succeeded":
   row["failure_layer"]=(result.get("failure") or {}).get("layer","generation")
   write(output/name/"case-report.json",row);return 1
  from candidate_bundle import check_bundle
  package=app._dir(task)/"program";manifest=check_bundle(package)
  require(manifest["agent_generated"] and result["generation"]["agent_generated"] and result["http_attempts"]>0
   and not result["package"]["reused"] and result["acceptance"]["numerically_validated"],"current_real_agent_numerical")
  require(result["acceptance"]["compiled"] is True and result["acceptance"]["encrypted_execution"] is True,
   "current_real_agent_numerical")
  row.update(compiled=result["acceptance"]["compiled"],encrypted_execution=result["acceptance"]["encrypted_execution"],
   source_capture_supervisor_executed=True,package=str(package),package_binding=manifest["binding"])
  # Reuse the qualified developer checks, not the rule-generation branch.
  spec=importlib.util.spec_from_file_location("reviewed_package_acceptance",HELPER)
  helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
  for phase in ("new_inputs","repeat","cache"):
   boundary(output,plan,name)
   require(helper.one(output,name,phase)==0,"package_"+phase)
   row[phase]=read(output/name/(phase+".json"))
  row["status"]="passed";write(output/name/"case-report.json",row);return 0
 except Exception as error:
  row["error_type"]=type(error).__name__;row["diagnostic"]=str(error)[:200]
  row.setdefault("failure_layer","environment")
  write(output/name/"case-report.json",row);return 1
if __name__=="__main__":
 for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
 parser=argparse.ArgumentParser(description=__doc__)
 mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument("--plan",action="store_true");mode.add_argument("--execute",type=Path)
 mode.add_argument("--inside",type=Path,help=argparse.SUPPRESS);mode.add_argument("--case",nargs=2,help=argparse.SUPPRESS)
 parser.add_argument("--approval-record",type=Path)
 args=parser.parse_args()
 if args.plan:print(json.dumps(preview(),indent=2,sort_keys=True));raise SystemExit(0)
 if args.execute:
  require(args.approval_record is not None,"approval_record_required")
  raise SystemExit(execute(args.execute,args.approval_record))
 if args.inside:raise SystemExit(inside(args.inside))
 raise SystemExit(one(*args.case))
