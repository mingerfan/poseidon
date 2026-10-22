"""JSON command interface over the existing ApplicationAssessment controller."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,sys,time,uuid
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[5] if Path(__file__).resolve().parents[3].name == "DSL_agent" and Path(__file__).resolve().parents[4].name == "DSL" else Path(__file__).resolve().parents[3])
FORMAT="poseidon-application-assessment-command-v1"
MAX_RESPONSE=16*1024**2+4096
MAX_LOG=20*1024**2
POLICY="registered-mathematical-float64-v1"
NATIVE_POLICY="bound-source-native-dtype-finite-v1"
CODES={
 "assessment_entry_input_rejected":("input","Command arguments or declared JSON inputs are invalid."),
 "assessment_entry_platform_unsupported":("capability","The selected execution platform does not match a configured Linux platform."),
 "assessment_entry_environment_unavailable":("environment","The pinned Poseidon worker environment is unavailable."),
 "assessment_entry_worker_identity_changed":("identity","The task store and current worker identities do not match."),
 "assessment_entry_response_rejected":("integrity","The worker response or invocation record is invalid."),
 "assessment_entry_timed_out":("execution","This command reached its declared deadline."),
 "assessment_entry_output_limit":("resource","The command reached its bounded output limit."),
 "assessment_entry_cancelled":("execution","The caller interrupted this command."),
 "assessment_entry_background_worker":("environment","The worker exited with unfinished child processes."),
 "assessment_entry_cleanup_unconfirmed":("environment","Worker cleanup could not be confirmed."),
 "assessment_entry_total_timed_out":("execution","The declared total dataset assessment deadline was reached."),
 "assessment_entry_assessment_cancelled":("execution","The dataset assessment was cancelled during this command."),
 "assessment_entry_clock_unavailable":("environment","The recorded assessment clock cannot be verified."),
 "assessment_entry_control_rejected":("integrity","The bound assessment execution-control metadata is invalid."),
}
class EntryError(ValueError):
 def __init__(self,code):
  self.code=code;super().__init__(CODES[code][1])
 def feedback(self):
  layer,message=CODES[self.code]
  return dict(code=self.code,layer=layer,message=message,retry_policy="never_automatic")
def require(ok,code="assessment_entry_input_rejected"):
 if not ok:raise EntryError(code)
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def bound(value):
 value={k:v for k,v in value.items()if k!="binding"};value["binding"]=digest(value);return value
def entry_identity():
 names=("scripts/agent_assessment.py","scripts/baseline/application_assessment_entry/__init__.py",
        "scripts/baseline/application_assessment_entry/cli.py",
        "scripts/baseline/application_assessment_entry/supervision.py",
        "scripts/baseline/assessment_time_budget.py")
 return digest({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
def absolute(value):
 p=Path(value).expanduser()
 require(p.is_absolute() and ".."not in p.parts and p==p.resolve() and
         p!=Path(p.anchor) and p!=Path.home() and not p.is_relative_to(ROOT))
 return p
def integer(low,high):
 def parse(value):
  require(re.fullmatch("[0-9]+",value)is not None)
  n=int(value);require(low<=n<=high);return n
 return parse
def hashed(value):
 require(re.fullmatch("[0-9a-f]{64}",value)is not None);return value
def task_id(value):
 require(re.fullmatch("[0-9a-f]{32}",value)is not None);return value
class Parser(argparse.ArgumentParser):
 def error(self,message):raise EntryError("assessment_entry_input_rejected")
def parse(argv):
 p=Parser(description="Evaluate an existing Poseidon program task on a bound local dataset.")
 p.add_argument("--tasks",required=True,type=absolute,help="Existing Application task store.")
 p.add_argument("--store",required=True,type=absolute,help="Persistent assessment store.")
 p.add_argument("--platform",choices=("x86_64-linux","aarch64-linux"))
 p.add_argument("--work-root",type=absolute)
 p.add_argument("--timeout",type=integer(1,14400),default=600,help="Whole command deadline, in seconds.")
 sub=p.add_subparsers(dest="action",required=True)
 submit=sub.add_parser("submit",help="Pin a dataset and an explicit execution quota.")
 submit.add_argument("--task",required=True,type=task_id)
 submit.add_argument("--dataset",required=True,type=Path)
 submit.add_argument("--dataset-binding",required=True,type=hashed)
 submit.add_argument("--max-new-assessments",type=integer(0,4096),default=0)
 submit.add_argument("--total-wall-seconds",type=integer(1,14400),
  help="Optional total assessment deadline shared by all advance commands.")
 submit.add_argument("--part-samples",type=integer(1,16),default=16)
 submit.add_argument("--max-receipt-bytes",type=integer(1,256*1024**2),default=64*1024**2)
 submit.add_argument("--reference-policy",choices=(POLICY,NATIVE_POLICY),default=POLICY)
 submit.add_argument("--criterion-file",type=Path)
 submit.add_argument("--input-dtypes-file",type=Path)
 submit.add_argument("--cache-receipts-file",type=Path)
 submit.add_argument("--trust-local-code",action="store_true")
 for action in ("status","result","advance","cancel"):
  item=sub.add_parser(action)
  item.add_argument("--evaluation-id",required=True,type=task_id)
  item.add_argument("--submission-binding",required=True,type=hashed)
  if action=="advance":item.add_argument("--max-new-parts",type=integer(0,4096),default=1)
 return p.parse_args(argv)
def json_input(path,limit):
 if path is None:return None
 require(path.is_absolute() and path==path.resolve() and path.suffix==".json" and not path.is_symlink())
 from model_weight_package import regular_bytes
 def pairs(items):
  result={}
  for k,v in items:require(k not in result);result[k]=v
  return result
 try:
  value=json.loads(regular_bytes(path,limit),object_pairs_hook=pairs,
   parse_constant=lambda _:(_ for _ in ()).throw(EntryError("assessment_entry_input_rejected")))
 except EntryError:raise
 except (ValueError,TypeError,OSError):raise EntryError("assessment_entry_input_rejected")from None
 require(type(value)is dict);return value
def submission(args):
 require(args.dataset.is_absolute() and args.dataset==args.dataset.resolve() and
         args.dataset.suffix==".json" and not args.dataset.is_symlink())
 require(args.reference_policy==NATIVE_POLICY or
         (not args.trust_local_code and args.input_dtypes_file is None))
 require(args.reference_policy!=NATIVE_POLICY or
         (args.trust_local_code and args.input_dtypes_file is not None))
 result=dict(expected_dataset_binding=args.dataset_binding,reference_policy=args.reference_policy,
  application_criterion=json_input(args.criterion_file,4*1024**2),
  input_dtypes=json_input(args.input_dtypes_file,131072),trust_local_code=args.trust_local_code,
  max_new_assessments=args.max_new_assessments,part_samples=args.part_samples,
  max_receipt_bytes=args.max_receipt_bytes,
  cached_receipts=json_input(args.cache_receipts_file,args.max_receipt_bytes))
 if args.total_wall_seconds is not None:result["total_wall_seconds"]=args.total_wall_seconds
 return result
def envelope(action,status,**values):
 return bound(dict(format=FORMAT,action=action,status=status,entry_identity=entry_identity(),**values))
def dispatch(args,factory):
 options=submission(args)if args.action=="submit"else None
 control=factory(args)
 if args.action=="submit":
  handle=control.submit(args.task,args.dataset,**options)
  return envelope(args.action,"submitted",handle=handle)
 handle=dict(format="poseidon-application-assessment-handle-v1",
             evaluation_id=args.evaluation_id,binding=args.submission_binding)
 kwargs=dict(max_new_parts=args.max_new_parts)if args.action=="advance"else{}
 state=getattr(control,args.action)(handle,**kwargs)
 require(state["status"]in ("queued","ready","running","succeeded","validation_failed",
  "unsupported","cancelled","timed_out","service_error","budget_exhausted"),"assessment_entry_response_rejected")
 return envelope(args.action,state["status"],handle=handle,result=state)
def feedback_response(action,error):
 feedback=error.feedback()
 status=("timed_out"if feedback["code"]in ("assessment_entry_timed_out","assessment_entry_total_timed_out")else
         "cancelled"if feedback["code"]in ("assessment_entry_cancelled","assessment_entry_assessment_cancelled")else
         "unsupported"if feedback["code"]=="assessment_entry_platform_unsupported"else
         "input_rejected"if feedback["layer"]=="input"or
          feedback["code"]=="assessment_controller_input_rejected"else"service_error")
 return envelope(action,status,failure=feedback)
def guarded(args,factory):
 try:return dispatch(args,factory)
 except EntryError as e:return feedback_response(args.action,e)
 except Exception as e:
  # All private details and traceback text remain outside the command result.
  from application_assessment import AssessmentControllerError
  from application_execution.context import ContextError
  if isinstance(e,AssessmentControllerError):return feedback_response(args.action,e)
  if isinstance(e,ContextError):
   code="assessment_entry_worker_identity_changed"if e.code in (
    "context_worker_changed","context_receipt_conflict")else"assessment_entry_environment_unavailable"
  else:code="assessment_entry_environment_unavailable"
  return feedback_response(args.action,EntryError(code))
def configured(args):
 if args.platform is not None:os.environ["POSEIDON_PLATFORM"]=args.platform
 if args.work_root is not None:os.environ["POSEIDON_WORK_ROOT"]=str(args.work_root)
 try:
  from workspace_paths import require_linux_backend,RESULTS
  require_linux_backend()
 except (ValueError,RuntimeError):raise EntryError("assessment_entry_platform_unsupported")from None
 return RESULTS
def factory(args):
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_assessment import ApplicationAssessment
 return ApplicationAssessment(application(WorkerContext.capture(args.tasks)),args.store)
def strict_response(raw):
 def pairs(items):
  value={}
  for k,v in items:require(k not in value,"assessment_entry_response_rejected");value[k]=v
  return value
 try:
  return json.loads(raw,object_pairs_hook=pairs,
   parse_constant=lambda _:(_ for _ in ()).throw(EntryError("assessment_entry_response_rejected")))
 except EntryError:raise
 except (ValueError,TypeError):raise EntryError("assessment_entry_response_rejected")from None
def safe_read(path,limit):
 from model_weight_package import regular_bytes
 return regular_bytes(path,limit)
def write_new(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,"wb")as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
def command_directory(results):
 require(results==results.resolve() and not results.is_symlink() and
         (not results.exists()or results.is_dir()),"assessment_entry_environment_unavailable")
 parent=results/"application-assessment-commands-v1"
 require(parent==parent.resolve() and not parent.is_symlink(),"assessment_entry_environment_unavailable")
 parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 st=parent.stat()
 require(parent==parent.resolve() and not parent.is_symlink() and
         st.st_uid==os.getuid() and st.st_mode&0o077==0,"assessment_entry_environment_unavailable")
 out=parent/uuid.uuid4().hex;out.mkdir(mode=0o700);return out
def _launch(argv,output,token):
 args=parse(argv);configured(args)
 from hecate_python_env import VENV,enter_nix
 if os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV:
  result=guarded(args,factory)
  write_new(Path(output)/"response.json",bound(dict(format=FORMAT,token=token,response=result)))
  return exit_status(result)
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([
  str(VENV/"bin/python"),"-B","-c",
  "import sys;sys.path.insert(0,sys.argv[1]);from application_assessment_entry.cli import _launch;raise SystemExit(_launch(sys.argv[4:],sys.argv[2],sys.argv[3]))",
  str(ROOT/"scripts/baseline"),output,token,*argv])
 return enter_nix(command,seconds=args.timeout,keep_env=(),environment="poseidon-cpu")
def stop(process):
 if process.poll()is None:
  try:os.killpg(process.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:process.wait(timeout=3)
  except subprocess.TimeoutExpired:
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   process.wait(timeout=5)
def advance_watch(args):
 if args.action!="advance":return None
 from assessment_time_budget import BudgetWatch,TimeBudgetError
 try:return BudgetWatch(args.store,args.evaluation_id,args.submission_binding)
 except (OSError,TimeBudgetError,ValueError,KeyError):
  raise EntryError("assessment_entry_control_rejected")from None
def check_advance(watch):
 if watch is None:return
 from assessment_time_budget import TimeBudgetError
 try:stop=watch.check()
 except (OSError,TimeBudgetError,ValueError,KeyError):
  raise EntryError("assessment_entry_control_rejected")from None
 if stop is not None:
  status,code=stop
  raise EntryError("assessment_entry_total_timed_out"if status=="timed_out"else
   "assessment_entry_assessment_cancelled"if status=="cancelled"else"assessment_entry_clock_unavailable")
def launch(args,argv,results):
 started=time.monotonic();out=command_directory(results);token=uuid.uuid4().hex
 record=bound(dict(format="poseidon-assessment-invocation-v1",token=token,action=args.action,
  arguments_sha256=digest(argv),entry_identity=entry_identity(),deadline_seconds=args.timeout))
 write_new(out/"invocation.json",record)
 command=[sys.executable,"-B","-c",
  "import sys;sys.path.insert(0,sys.argv[1]);from application_assessment_entry.cli import _launch;raise SystemExit(_launch(sys.argv[4:],sys.argv[2],sys.argv[3]))",
  str(ROOT/"scripts/baseline"),str(out),token,*argv]
 command=[sys.executable,"-B",str(Path(__file__).with_name("supervision.py")),str(out),token,*command]
 supervision=None;worker_started=False
 try:
  watch=advance_watch(args);check_advance(watch)
  fd=os.open(out/"worker.log",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,"wb")as log:
   process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   worker_started=True
   try:
    while process.poll()is None:
     check_advance(watch)
     if time.monotonic()-started>=args.timeout:raise EntryError("assessment_entry_timed_out")
     if (out/"worker.log").stat().st_size>MAX_LOG:raise EntryError("assessment_entry_output_limit")
     time.sleep(.1)
   finally:stop(process)
  require((out/"worker.log").stat().st_size<=MAX_LOG,"assessment_entry_output_limit")
  from application_assessment_entry.supervision import checked,SupervisionError
  try:
   supervision=checked(strict_response(safe_read(out/"process-supervision.json",65536)),
    token=token,invocation_binding=record["binding"])
  except (SupervisionError,OSError):
   raise EntryError("assessment_entry_cleanup_unconfirmed")from None
  require(supervision["cleanup_completed"],"assessment_entry_cleanup_unconfirmed")
  require(supervision["status"]=="completed","assessment_entry_background_worker")
  raw=safe_read(out/"response.json",MAX_RESPONSE)
  wire=strict_response(raw)
  require(type(wire)is dict and set(wire)=={"format","token","response","binding"} and
          wire["format"]==FORMAT and wire["token"]==token and
          wire["binding"]==digest({k:v for k,v in wire.items()if k!="binding"}),
          "assessment_entry_response_rejected")
  response=wire["response"]
  require(type(response)is dict and response["format"]==FORMAT and response["action"]==args.action and
   response["entry_identity"]==record["entry_identity"] and
   response["binding"]==digest({k:v for k,v in response.items()if k!="binding"})and
   process.returncode==exit_status(response),"assessment_entry_response_rejected")
  require(time.monotonic()-started<=args.timeout,"assessment_entry_timed_out")
  result=response
 except EntryError as e:result=feedback_response(args.action,e)
 except KeyboardInterrupt:result=feedback_response(args.action,EntryError("assessment_entry_cancelled"))
 except (OSError,ValueError,KeyError,subprocess.SubprocessError):
  result=feedback_response(args.action,EntryError("assessment_entry_environment_unavailable"))
 if supervision is None:
  try:
   from application_assessment_entry.supervision import checked
   supervision=checked(strict_response(safe_read(out/"process-supervision.json",65536)),
    token=token,invocation_binding=record["binding"])
  except (OSError,ValueError,RuntimeError,EntryError):
   supervision=None
 result=bound(dict(result,process_cleanup=dict(
  confirmed=bool(supervision and supervision["cleanup_completed"])or not worker_started,
  status=("not_started"if not worker_started else None)if supervision is None else supervision["status"],
  binding=None if supervision is None else supervision["binding"])))
 write_new(out/"outcome.json",bound(dict(format="poseidon-assessment-invocation-outcome-v1",
  invocation_binding=record["binding"],response_binding=result["binding"],
  elapsed_seconds=time.monotonic()-started,status=result["status"],
  process_supervision_binding=None if supervision is None else supervision["binding"],
  worker_tree_cleanup_confirmed=bool(supervision and supervision["cleanup_completed"])or not worker_started)))
 return result
def exit_status(result):
 status=result["status"]
 return (2 if status in ("input_rejected","unsupported")else
         3 if status in ("validation_failed","budget_exhausted")else
         4 if status in ("cancelled","timed_out")else
         5 if status=="service_error"else 0)
def main(argv=None,*,controller_factory=None,stdout=None):
 argv=list(sys.argv[1:]if argv is None else argv);out=sys.stdout if stdout is None else stdout
 action="unparsed"
 try:
  args=parse(argv);action=args.action
  # CLI options never expose test factories or alternative execution backends.
  if controller_factory is not None:result=guarded(args,controller_factory)
  else:
   results=configured(args)
   if args.action=="submit":submission(args)
   result=launch(args,argv,results)
 except EntryError as e:result=feedback_response(action,e)
 except (ValueError,TypeError,OSError,RuntimeError):
  result=feedback_response(action,EntryError("assessment_entry_environment_unavailable"))
 out.write(canonical(result).decode()+"\n");out.flush()
 return exit_status(result)
