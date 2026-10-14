"""Three reviewed source requests through Application. Live opt-in; no auto-resume."""
import argparse,fcntl,importlib.util,json,os,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
from scope import ROOT,PROPOSAL,CASES,canonical,digest,sha,read,require,verify,authority,totals,terminal
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
HELPER=ROOT/"scripts/baseline/benchmarks/tools/staged_component_cutover_v1/post_cutover_acceptance.py"
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_bytes(canonical(value));os.chmod(tmp,0o600);tmp.replace(path)
def tool_hashes():
 return {str(p):sha(p) for p in (*sorted(Path(__file__).parent.glob("*.py")),HELPER)}
def preview():
 p=verify(read(PROPOSAL))
 return dict(format="poseidon-source-live-tool-plan-v1",proposal_binding=p["binding"],
  proposal_sha256=sha(PROPOSAL),cases=list(CASES),provider=p["provider"],budget=p["budget"],
  tool_hashes=tool_hashes(),approved=False,provider_calls=0,
  validation=["source_capture","numerical_qualification","new_input_execution","package_cache_reuse"])
def measure(path):
 return max(int(subprocess.check_output(["du",arg,"-s","-B1",str(path)],text=True,timeout=30).split()[0])
  for arg in ("--apparent-size","--count-links"))
def no_active():
 paths={str(Path(__file__).resolve()),str(BASE/"candidate_worker.py"),
  str(BASE/"run_candidate.py"),str(BASE/"deepseek_http_worker.py")}
 for p in Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:a=(p/"cmdline").read_bytes().split(b"\0")
  except OSError:continue
  require(not any(os.fsdecode(x) in paths for x in a if x),"active_project_execution")
def execute(output,approval):
 require(Path(output).is_absolute() and Path(output)==Path(output).resolve() and
  Path(output).parent==RESULTS and not Path(output).exists(),"fresh_campaign_root")
 proposal=verify(read(PROPOSAL));authority(read(approval),proposal,sha(PROPOSAL));no_active()
 lock=WORK/"deps/nix-portable-v012/launcher.lock"
 with lock.open("r+") as f:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 public=preview();initial=sorted(p.name for p in RESULTS.iterdir())
 require(measure(RESULTS)<=proposal["budget"]["global_result_bytes"] and
  shutil.disk_usage(WORK).free>=proposal["budget"]["reserve_bytes"],"artifact_or_disk_boundary")
 Path(output).mkdir(mode=0o700)
 plan=dict(public,approved=True,approval_record=str(Path(approval).resolve()),
  approval_sha256=sha(approval),created=time.time(),initial_result_directories=initial,
  deadline_wall=time.time()+proposal["budget"]["batch_wall_seconds"])
 plan["binding"]=digest(plan);write(Path(output)/"plan.json",plan)
 sys.path.insert(0,str(BASE))
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",str(output)])
 return enter_nix(command,seconds=proposal["budget"]["batch_wall_seconds"],keep_env=())
def boundary(output,plan,case=None):
 require(plan["binding"]==digest({k:v for k,v in plan.items() if k!="binding"}),"execution_plan_binding")
 require(sha(PROPOSAL)==plan["proposal_sha256"],"proposal_changed")
 proposal=verify(read(PROPOSAL));authority(read(plan["approval_record"]),proposal,sha(PROPOSAL))
 require(sha(plan["approval_record"])==plan["approval_sha256"],"approval_record_changed")
 require(all(sha(p)==h for p,h in plan["tool_hashes"].items()),"execution_tool_changed")
 require(time.time()<plan["deadline_wall"],"batch_time_boundary")
 require(shutil.disk_usage(WORK).free>=proposal["budget"]["reserve_bytes"],"disk_reserve")
 require(measure(RESULTS)<=proposal["budget"]["global_result_bytes"],"global_artifact_boundary")
 if case is not None:
  limit=read(Path(output)/case/"case-boundary.json")
  added=[p for p in RESULTS.iterdir() if p.name not in limit["initial_result_directories"]]
  require(measure(Path(output)/case)+sum(measure(p) for p in added)<=
   proposal["budget"]["per_case_bytes"],"case_artifact_boundary")
  require(time.time()<limit["deadline_wall"],"case_time_boundary")
 return proposal
def states(output):
 result=[]
 for c in CASES:
  for p in (Path(output)/c/"tasks").glob("*/state.json"):result.append(read(p))
 return result
def save(output,plan,rows,status=None):
 data=terminal(rows,states(output))
 if status is not None:data["status"]=status
 write(Path(output)/"report.json",dict(format="poseidon-source-live-report-v1",binding=plan["binding"],
  proposal_binding=plan["proposal_binding"],planned=3,rows=rows,cost_usd=None,
  new_agent_generation=True,automatic_restart=False,**data))
 return data
def stop(child):
 # Capture descendants before stopping the owned worker, including separate sessions.
 pairs={}
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:
   parts=(p/"stat").read_text().rsplit(")",1)[1].split()
   pairs[int(p.name)]=(int(parts[1]),parts[19])
  except (OSError,ValueError,IndexError):pass
 chosen={child.pid}
 while True:
  extra={pid for pid,(parent,_) in pairs.items() if parent in chosen}
  if extra<=chosen:break
  chosen|=extra
 for sig in (signal.SIGTERM,signal.SIGKILL):
  for pid in sorted(chosen,reverse=True):
   try:
    now=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
    if pid in pairs and now[19]==pairs[pid][1]:os.kill(pid,sig)
   except (OSError,IndexError):pass
  if sig==signal.SIGTERM:
   try:child.wait(timeout=2)
   except subprocess.TimeoutExpired:pass
 child.wait(timeout=5)
def inside(output):
 sys.path.insert(0,str(BASE))
 from hecate_python_env import VENV
 require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"pinned_worker")
 output=Path(output);plan=read(output/"plan.json");proposal=boundary(output,plan)
 from application_backend import PoseidonBackend
 require(PoseidonBackend().identity()==proposal["backend_identity"],"backend_identity_changed")
 write(output/"execution-claim.json",dict(pid=os.getpid(),started=time.time(),binding=plan["binding"]))
 rows=[];save(output,plan,rows,"running")
 try:
  for case in proposal["cases"]:
   name=case["id"];boundary(output,plan);require(totals(states(output))["calls_resolved"],"unresolved_previous_call")
   folder=output/name;folder.mkdir(mode=0o700)
   deadline=min(plan["deadline_wall"],time.time()+proposal["budget"]["case_wall_seconds"])
   write(folder/"case-boundary.json",dict(deadline_wall=deadline,
    initial_result_directories=sorted(p.name for p in RESULTS.iterdir())))
   allow=("PATH","IN_NIX_SHELL","POSEIDON_PLATFORM","POSEIDON_WORK_ROOT","HECATE_PYTHON_LIBRARY_PATH","LD_LIBRARY_PATH","LANG")
   env={k:os.environ[k] for k in allow if k in os.environ}
   env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",PYTHONPATH=str(BASE),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",HOME="/nonexistent")
   with (folder/"worker.log").open("xb") as log:
    child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--case",str(output),name],
     env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
     while child.poll() is None:
      boundary(output,plan,name);time.sleep(3)
    except BaseException:
     stop(child);raise
   row=read(folder/"case-report.json") if (folder/"case-report.json").is_file() else dict(
    case=name,status="failed",failure_layer="worker",code="missing_case_receipt")
   row["exit_code"]=child.returncode
   if child.returncode:row["status"]="failed"
   rows.append(row);save(output,plan,rows,"running");boundary(output,plan,name)
   print(json.dumps(dict(case=name,status=row["status"],**totals(states(output)))),flush=True)
   require(row.get("failure_layer") not in ("integrity","environment","resources") and
    totals(states(output))["calls_resolved"],"campaign_boundary_or_unresolved_call")
  data=save(output,plan,rows)
  return 0 if data["passed"]==3 else 1
 except BaseException:
  save(output,plan,rows,"stopped");raise
def one(output,name):
 output=Path(output);plan=read(output/"plan.json");proposal=boundary(output,plan,name)
 case=next(c for c in proposal["cases"] if c["id"]==name)
 sys.path.insert(0,str(BASE))
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from application_backend import PoseidonBackend
 require(PoseidonBackend().identity()==proposal["backend_identity"],"backend_identity_changed")
 app=application(WorkerContext.capture(output/name/"tasks"))
 row=dict(case=name,status="failed",new_agent_generation=True,request_id=case["request_id"])
 try:
  limits=dict(generations=4,http_attempts=16,wall_seconds=max(1,int(read(output/name/"case-boundary.json")["deadline_wall"]-time.time())),
   api_timeout_seconds=1200,max_tokens=384000)
  options=dict(chunk_period=4,generation_guidance="explicit-v11",capacity_packing_version="capacity-public-tensors-v9",public_weight_policy="descriptors-v1")
  source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(ROOT/case["source_manifest"]),trust_local_code=True)
  task=app.submit_source(source,resource_profile="general-tensors-v1",options=options,limits=limits)
  write(output/name/"task-id.json",dict(task_id=task));row["task_id"]=task
  state=app.prepare_source(task)
  if state["status"]!="queued":
   row.update(result=state,failure_layer=(state.get("failure") or {}).get("layer","input"))
   write(output/name/"case-report.json",row);return 1
  meta=app._checked_task_metadata(task)
  require(meta["request"]==read(ROOT/case["request_file"]),"frozen_mathematical_request")
  require(meta["identity"]["validation"]["sources"]==case["reference_binding"]["validation_sources"],
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
  if result["status"]!="succeeded":
   row["failure_layer"]=(result.get("failure") or {}).get("layer","generation")
   write(output/name/"case-report.json",row);return 1
  from candidate_bundle import check_bundle
  package=app._dir(task)/"program";manifest=check_bundle(package)
  require(manifest["agent_generated"] and result["generation"]["agent_generated"] and result["http_attempts"]>0
   and not result["package"]["reused"] and result["acceptance"]["numerically_validated"],"current_real_agent_numerical")
  row.update(compiled=result["acceptance"]["compiled"],encrypted_execution=result["acceptance"]["encrypted_execution"],
   source_capture_supervisor_executed=True,package=str(package),package_binding=manifest["binding"])
  # Reuse the qualified developer checks, not the rule-generation branch.
  spec=importlib.util.spec_from_file_location("reviewed_package_acceptance",HELPER)
  helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
  for phase in ("new_inputs","cache"):
   boundary(output,plan,name)
   require(helper.one(output,name,phase)==0,"package_"+phase)
   row[phase]=read(output/name/(phase+".json"))
  row["status"]="passed";write(output/name/"case-report.json",row);return 0
 except Exception as error:
  row["error_type"]=type(error).__name__
  row.setdefault("failure_layer","worker")
  write(output/name/"case-report.json",row);return 1
if __name__=="__main__":
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
