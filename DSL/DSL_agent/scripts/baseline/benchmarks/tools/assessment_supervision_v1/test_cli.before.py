"""Entry controls use an injected backend; no API, source execution or real FHE."""
import copy,io,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from application_assessment import ApplicationAssessment,AssessmentControllerError
from application_execution.context import ContextError
from test_application_assessment import ApplicationAssessmentTests
from application_assessment_entry import cli
class CliTests(unittest.TestCase):
 setUp=ApplicationAssessmentTests.setUp
 ready=ApplicationAssessmentTests.ready
 prepared=ApplicationAssessmentTests.prepared
 def setup_entry(self,n=3):
  self.control,self.task,self.folder,self.manifest,self.provider,self.original=self.prepared(n)
  self.args=["--tasks",str(self.app.store),"--store",str(self.control.store)]
  self.factories=0
 def factory(self,args):
  self.factories+=1
  self.assertEqual(args.tasks,self.app.store)
  return ApplicationAssessment(self.app,args.store)
 def call(self,parts,*,factory=None):
  out=io.StringIO();code=cli.main(self.args+parts,controller_factory=self.factory if factory is None else factory,stdout=out)
  self.assertEqual(len(out.getvalue().splitlines()),1)
  result=json.loads(out.getvalue())
  self.assertEqual(result["format"],cli.FORMAT)
  self.assertEqual(result["binding"],cli.digest({k:v for k,v in result.items()if k!="binding"}))
  self.assertNotIn("PRIVATE",out.getvalue())
  return code,result
 def submit(self,extra=()):
  return self.call(["submit","--task",self.task,"--dataset",str(self.folder/"dataset.json"),
                   "--dataset-binding",self.manifest["binding"],*extra])
 def action(self,name,handle,extra=()):
  return self.call([name,"--evaluation-id",handle["evaluation_id"],"--submission-binding",handle["binding"],*extra])
 def test_submit_only_pins_inputs_and_returns_handle(self):
  self.setup_entry();code,r=self.submit()
  self.assertEqual(code,0);self.assertEqual(r["status"],"submitted")
  self.assertEqual(self.backend.execute_calls,0);self.assertEqual(self.provider.calls,1)
 def test_status_and_result_are_machine_readable_without_execution(self):
  self.setup_entry();_,r=self.submit();h=r["handle"]
  for action in ("status","result"):
   code,s=self.action(action,h);self.assertEqual(code,0);self.assertEqual(s["result"]["budget"]["used"],0)
  self.assertEqual(self.backend.execute_calls,0)
 def test_default_zero_quota_returns_budget_terminal(self):
  self.setup_entry();_,r=self.submit();code,s=self.action("advance",r["handle"])
  self.assertEqual(code,3);self.assertEqual(s["status"],"budget_exhausted")
  self.assertEqual(s["result"]["budget"]["used"],0);self.assertEqual(self.backend.execute_calls,0)
 def test_explicit_quota_executes_only_once_and_preserves_program(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"))
  code,s=self.action("advance",r["handle"]);self.assertEqual((code,s["status"]),(0,"succeeded"))
  self.assertEqual(self.backend.execute_calls,1);self.assertEqual(s["result"]["budget"]["used"],1)
  self.action("advance",r["handle"]);self.action("result",r["handle"])
  self.assertEqual(self.backend.execute_calls,1);self.assertEqual(self.app.result(self.task),self.original)
 def test_separate_command_objects_resume_without_repeat(self):
  self.setup_entry(17);_,r=self.submit(("--max-new-assessments","2"))
  self.action("advance",r["handle"]);self.assertEqual(self.backend.execute_calls,1)
  code,s=self.action("advance",r["handle"]);self.assertEqual(s["result"]["completed_parts"],2)
  self.assertEqual(self.backend.execute_calls,2);self.assertEqual(code,0)
 def test_zero_advance_limit_has_no_execution(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"))
  _,s=self.action("advance",r["handle"],("--max-new-parts","0"))
  self.assertEqual(s["status"],"ready");self.assertEqual(self.backend.execute_calls,0)
 def test_cancel_is_scoped_to_assessment(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"))
  code,s=self.action("cancel",r["handle"]);self.assertEqual((code,s["status"]),(4,"cancelled"))
  self.action("advance",r["handle"]);self.assertEqual(self.backend.execute_calls,0)
  self.assertFalse((self.app.store/self.task/"cancel.json").exists())
 def test_cancel_completed_task_preserves_result(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"))
  _,s=self.action("advance",r["handle"]);_,c=self.action("cancel",r["handle"])
  self.assertEqual(s["result"]["binding"],c["result"]["binding"]);self.assertEqual(c["status"],"succeeded")
 def test_numerical_failure_has_failure_exit_and_layer(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"));self.backend.delta=1
  code,s=self.action("advance",r["handle"])
  self.assertEqual(code,3);self.assertEqual(s["status"],"validation_failed")
  self.assertEqual(s["result"]["failure"]["code"],"assessment_numerical_mismatch")
 def test_invalid_cli_is_input_error_without_factory(self):
  self.setup_entry()
  for parts in (["unknown"],["submit","--task","PRIVATE"],["status","--evaluation-id","0"*32],
                ["submit","--provider","deepseek"],["--timeout","0","status"],["--timeout","14401","status"]):
   code,r=self.call(parts);self.assertEqual((code,r["status"]),(2,"input_rejected"))
  self.assertEqual(self.factories,0)
 def test_bad_handle_identifier_has_no_factory_or_execution(self):
  self.setup_entry()
  code,r=self.call(["status","--evaluation-id","../PRIVATE","--submission-binding","0"*64])
  self.assertEqual(code,2);self.assertEqual(self.factories,0)
 def test_controller_rejects_wrong_binding_without_execution(self):
  self.setup_entry();_,r=self.submit();h=dict(r["handle"],binding="0"*64)
  code,s=self.action("status",h);self.assertEqual(code,5)
  self.assertEqual(s["failure"]["code"],"assessment_controller_binding_changed")
  self.assertEqual(self.backend.execute_calls,0)
 def test_native_policy_needs_trust_and_dtypes(self):
  self.setup_entry();code,r=self.submit(("--reference-policy",cli.NATIVE_POLICY))
  self.assertEqual(code,2);self.assertEqual(self.factories,0)
 def test_mathematical_policy_rejects_source_trust(self):
  self.setup_entry();code,r=self.submit(("--trust-local-code",))
  self.assertEqual(code,2);self.assertEqual(self.factories,0)
 def test_criterion_duplicate_nonfinite_and_wrong_shape_are_input_errors(self):
  self.setup_entry();p=Path(self.tmp.name)/"criterion.json"
  for raw in ('{"kind":"max_absolute_error","limit":0.1,"limit":1}', '{"kind":"max_absolute_error","limit":NaN}',"[]","PRIVATE"):
   p.write_text(raw)
   code,r=self.submit(("--criterion-file",str(p)));self.assertEqual((code,r["status"]),(2,"input_rejected"))
  self.assertEqual(self.factories,0)
 def test_credential_named_input_is_rejected_before_read(self):
  self.setup_entry();p=Path(self.tmp.name)/".env";p.write_text("PRIVATE")
  code,r=self.submit(("--criterion-file",str(p)));self.assertEqual(code,2);self.assertEqual(self.factories,0)
 def test_store_under_source_rejected_before_factory(self):
  self.setup_entry();self.args=["--tasks",str(self.app.store),"--store",str(cli.ROOT/"PRIVATE")]
  code,r=self.submit();self.assertEqual(code,2);self.assertEqual(self.factories,0)
 def test_environment_exception_does_not_leak_details(self):
  self.setup_entry()
  def factory(_):raise RuntimeError("PRIVATE internal path")
  code,r=self.submit()
  code,r=self.call(["status","--evaluation-id","0"*32,"--submission-binding","0"*64],factory=factory)
  self.assertEqual(code,5);self.assertEqual(r["failure"]["code"],"assessment_entry_environment_unavailable")
 def test_worker_identity_conflict_is_distinct(self):
  self.setup_entry()
  def factory(_):raise ContextError("context_receipt_conflict")
  code,r=self.call(["status","--evaluation-id","0"*32,"--submission-binding","0"*64],factory=factory)
  self.assertEqual(code,5);self.assertEqual(r["failure"]["layer"],"identity")
 def test_controller_input_error_has_input_terminal(self):
  self.setup_entry()
  def factory(_):raise AssessmentControllerError("assessment_controller_input_rejected")
  code,r=self.call(["status","--evaluation-id","0"*32,"--submission-binding","0"*64],factory=factory)
  self.assertEqual((code,r["status"]),(2,"input_rejected"))
 def test_machine_result_keeps_numerical_and_accuracy_separate(self):
  self.setup_entry();criterion=Path(self.tmp.name)/"criterion.json"
  criterion.write_text(json.dumps({"kind":"max_absolute_error","limit":1.0}))
  _,r=self.submit(("--max-new-assessments","1","--criterion-file",str(criterion)))
  _,s=self.action("advance",r["handle"])
  self.assertTrue(s["result"]["acceptance"]["numerical_on_declared_dataset"])
  self.assertFalse(s["result"]["acceptance"]["original_source_accuracy_certified"])
 def test_interrupted_ledger_is_not_reexecuted(self):
  self.setup_entry();_,r=self.submit(("--max-new-assessments","1"));h=r["handle"]
  class Stopped(BaseException):pass
  with patch.object(self.control.numeric,"assess",side_effect=Stopped):
   with self.assertRaises(Stopped):self.control.advance(h)
  code,s=self.action("status",h);self.assertEqual(code,5)
  self.assertEqual(s["failure"]if"failure"in s else s["result"]["failure"],
    s["result"]["failure"])
  self.action("advance",h);self.assertEqual(self.backend.execute_calls,0)
 def launched(self,mutate=None):
  args=cli.parse(self.args+["status","--evaluation-id","0"*32,"--submission-binding","0"*64])
  roots=Path(self.tmp.name)/"command-results"
  def spawn(command,**kwargs):
   output=Path(command[5]);token=command[6]
   result=cli.envelope("status","queued",result={"status":"queued"})
   wire=cli.bound(dict(format=cli.FORMAT,token=token,response=result))
   if mutate:wire=mutate(wire)
   if isinstance(wire,dict):cli.write_new(output/"response.json",wire)
   elif isinstance(wire,bytes):(output/"response.json").write_bytes(wire)
   kwargs["stdout"].write(b"PRIVATE sdk startup log\n");kwargs["stdout"].flush()
   class Process:
    pid=os.getpid()
    returncode=cli.exit_status(result)
    def poll(self):return self.returncode
   return Process()
  with patch.object(cli.subprocess,"Popen",side_effect=spawn):
   return cli.launch(args,self.args+["status","--evaluation-id","0"*32,"--submission-binding","0"*64],roots)
 def test_launcher_filters_sdk_logs_from_machine_result(self):
  self.setup_entry();r=self.launched();self.assertEqual(r["status"],"queued")
  self.assertNotIn("PRIVATE",json.dumps(r))
  logs=list((Path(self.tmp.name)/"command-results").rglob("worker.log"))
  self.assertEqual(len(logs),1);self.assertIn("PRIVATE",logs[0].read_text())
 def test_launcher_checks_response_token(self):
  self.setup_entry()
  def changed(w):w["token"]="0"*32;return cli.bound(w)
  r=self.launched(changed);self.assertEqual(r["failure"]["code"],"assessment_entry_response_rejected")
 def test_launcher_checks_response_binding(self):
  self.setup_entry();r=self.launched(lambda w:dict(w,binding="0"*64))
  self.assertEqual(r["failure"]["code"],"assessment_entry_response_rejected")
 def test_launcher_rejects_duplicate_response_keys(self):
  self.setup_entry()
  r=self.launched(lambda w:cli.canonical(w).replace(b'"token":',b'"token":"PRIVATE","token":',1))
  self.assertEqual(r["failure"]["code"],"assessment_entry_response_rejected")
 def test_launcher_preserves_missing_response_failure(self):
  self.setup_entry();r=self.launched(lambda w:None)
  self.assertEqual(r["failure"]["code"],"assessment_entry_environment_unavailable")
 def test_deadline_stops_one_actual_local_child_without_fhe(self):
  self.setup_entry();args=cli.parse(self.args+["--timeout","1","status","--evaluation-id","0"*32,"--submission-binding","0"*64])
  original=cli.subprocess.Popen;children=[]
  def slow(command,**kwargs):
   p=original([sys.executable,"-B","-c","import time;time.sleep(30)"],**kwargs);children.append(p);return p
  started=time.monotonic()
  with patch.object(cli.subprocess,"Popen",side_effect=slow):
   r=cli.launch(args,[],Path(self.tmp.name)/"deadline-results")
  self.assertEqual(r["status"],"timed_out");self.assertEqual(len(children),1)
  self.assertIsNotNone(children[0].poll());self.assertLess(time.monotonic()-started,6)
 def test_existing_private_worker_receipt_is_not_rebound(self):
  self.setup_entry();p=self.app.store/"worker-context.json";p.write_text("PRIVATE preserved receipt")
  before=p.read_bytes()
  def factory(_):raise ContextError("context_receipt_conflict")
  code,r=self.call(["status","--evaluation-id","0"*32,"--submission-binding","0"*64],factory=factory)
  self.assertEqual(code,5);self.assertEqual(p.read_bytes(),before)
 def test_symlink_command_results_reject_before_creating_target(self):
  self.setup_entry();outside=Path(self.tmp.name)/"outside";outside.mkdir()
  alias=Path(self.tmp.name)/"alias";alias.symlink_to(outside,target_is_directory=True)
  with self.assertRaises(cli.EntryError):cli.command_directory(alias)
  self.assertFalse((outside/"application-assessment-commands-v1").exists())
 def test_world_readable_command_store_is_rejected(self):
  self.setup_entry();results=Path(self.tmp.name)/"public-results"
  target=results/"application-assessment-commands-v1";target.mkdir(parents=True,mode=0o755)
  target.chmod(0o755)
  with self.assertRaises(cli.EntryError):cli.command_directory(results)
  self.assertFalse(list(target.iterdir()))
 def test_all_terminal_exit_codes_are_explicit(self):
  self.assertEqual({s:cli.exit_status({"status":s}) for s in (
    "submitted","queued","ready","running","succeeded","input_rejected","unsupported",
    "validation_failed","budget_exhausted","cancelled","timed_out","service_error")},
    dict(submitted=0,queued=0,ready=0,running=0,succeeded=0,input_rejected=2,unsupported=2,
         validation_failed=3,budget_exhausted=3,cancelled=4,timed_out=4,service_error=5))
if __name__=="__main__":unittest.main()
