"""Stdlib-only controls and synthetic Linux processes; never Nix/SDK/Provider."""
import copy,json,os,signal,subprocess,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest import mock
import run
def seal(v):
 v=copy.deepcopy(v);v.pop("binding",None);v["binding"]=run.digest(v);return v
class Controls(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix="poseidon-live-control-",dir="/tmp");self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);self.results=self.root/"results";self.results.mkdir()
  self.output=self.results/"live";self.output.mkdir()
  self.proposal=self.root/"proposal.json";self.proposal.write_text("{}")
  self.approval=self.root/"approval.json";self.approval.write_text("{}")
  for key,value in dict(ROOT=self.root,RESULTS=self.results,RUN=self.output,LOCK=self.results/"live.lock",PROPOSAL=self.proposal).items():
   patch=mock.patch.object(run,key,value);patch.start();self.addCleanup(patch.stop)
  self.proposal_data=dict(binding="p",provider={},budget=dict(batch_wall_seconds=43200,case_wall_seconds=14400),
   application_backend_identity={},cases=[dict(id=n,request_id="r") for n in run.CASES])
  patch=mock.patch.object(run,"tool_hashes",return_value={"synthetic-controller":"a"*64});patch.start();self.addCleanup(patch.stop)
  self.lock=run.owned_lock(run.LOCK);self.stream=self.lock.__enter__();self.addCleanup(self.lock.__exit__,None,None,None)
  now=time.time()
  self.plan=seal(dict(run.public_plan(self.proposal_data),approved=True,approval_record=str(self.approval),
   approval_sha256=run.sha(self.approval),created=now,deadline_wall=now+43200,
   initial_result_directories=[],owner=run.process_identity(os.getpid())))
  run.write(self.output/"plan.json",self.plan);run.bind_lock(self.stream,self.plan)
 def folder(self,name=None):
  name=name or run.CASES[0];p=self.output/name;p.mkdir(exist_ok=True)
  started=time.time();value=seal(dict(format="poseidon-live-case-boundary-v2",case=name,plan_binding=self.plan["binding"],
   started_wall=started,deadline_wall=min(self.plan["deadline_wall"],started+14400),initial_result_directories=[self.output.name]))
  run.write(p/"case-boundary.json",value);return p
 def test_fixed_plan_and_real_owned_lock(self):
  run.validate_plan(self.plan,self.proposal_data)
 def test_plan_fields_cannot_be_removed_or_extended(self):
  for mutate in (lambda p:p.pop("owner"),lambda p:p.update(unreviewed=True)):
   value=copy.deepcopy(self.plan);mutate(value)
   with self.assertRaisesRegex(ValueError,"execution_plan_fields"):run.validate_plan(seal(value),self.proposal_data)
 def test_approved_public_fields_cannot_be_rebound(self):
  for key,value in dict(cases=["other"],budget={},provider={"model":"other"},tool_hashes={},proposal_binding="other",approved=False).items():
   plan=copy.deepcopy(self.plan);plan[key]=value
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"execution_preserves_approved_plan"):
    run.validate_plan(seal(plan),self.proposal_data)
 def test_batch_deadline_extension_refused(self):
  value=copy.deepcopy(self.plan);value["deadline_wall"]+=1
  with self.assertRaisesRegex(ValueError,"original_batch_deadline"):run.validate_plan(seal(value),self.proposal_data)
 def test_recreated_plan_not_accepted_by_live_lock_claim(self):
  value=copy.deepcopy(self.plan);value["created"]-=1;value["deadline_wall"]-=1
  run.write(self.output/"plan.json",seal(value))
  with self.assertRaisesRegex(ValueError,"execution_owner_plan_binding"):run.validate_plan(seal(value),self.proposal_data)
 def test_changed_owner_starttime_refused(self):
  value=copy.deepcopy(self.plan);value["owner"]["start"]+=1
  with self.assertRaisesRegex(ValueError,"execution_owner_identity"):run.validate_plan(seal(value),self.proposal_data)
 def test_owner_lock_must_still_be_held(self):
  import fcntl
  fcntl.flock(self.stream,fcntl.LOCK_UN)
  with self.assertRaisesRegex(ValueError,"execution_owner_lock"):run.validate_plan(self.plan,self.proposal_data)
 def test_case_writer_reader_and_marker(self):
  p=self.folder();name=p.name
  value=run.case_boundary(self.output,name,self.plan,self.proposal_data)
  run.reserve_case(p,name,self.plan)
  self.assertEqual(run.case_boundary(self.output,name,self.plan,self.proposal_data),value)
  with self.assertRaises(FileExistsError):run.reserve_case(p,name,self.plan)
 def test_case_deadline_cannot_be_extended(self):
  p=self.folder();v=run.read(p/"case-boundary.json");v["deadline_wall"]+=1;run.write(p/"case-boundary.json",seal(v))
  with self.assertRaisesRegex(ValueError,"original_case_deadline"):run.case_boundary(self.output,p.name,self.plan,self.proposal_data)
 def test_case_inventory_and_identity_refused(self):
  p=self.folder()
  for key,value in dict(case="unknown",plan_binding="wrong",initial_result_directories=[]).items():
   v=run.read(p/"case-boundary.json");v[key]=value;run.write(p/"case-boundary.json",seal(v))
   with self.assertRaises(ValueError):run.case_boundary(self.output,p.name,self.plan,self.proposal_data)
 def test_marker_prevents_case_limit_rebind(self):
  p=self.folder();run.reserve_case(p,p.name,self.plan)
  v=run.read(p/"case-boundary.json");v["started_wall"]=time.time();v["deadline_wall"]=v["started_wall"]+14400
  run.write(p/"case-boundary.json",seal(v))
  with self.assertRaisesRegex(ValueError,"case_started_identity"):run.case_boundary(self.output,p.name,self.plan,self.proposal_data)
 def test_supervisor_claim_single_use_and_starttime(self):
  run.reserve_execution(self.output,self.plan);run.supervisor_current(self.output,self.plan)
  with self.assertRaises(FileExistsError):run.reserve_execution(self.output,self.plan)
  v=run.read(self.output/"execution-claim.json");v["process_start"]="0";run.write(self.output/"execution-claim.json",v)
  with self.assertRaisesRegex(ValueError,"owned_case_supervisor_identity"):run.supervisor_current(self.output,self.plan)
 def test_receipt_missing_wrong_case_request_or_plan_refused(self):
  p=self.folder();run.reserve_case(p,p.name,self.plan);case={"request_id":"r"}
  with self.assertRaisesRegex(ValueError,"missing_case_receipt"):run.checked_receipt(p,p.name,self.plan,case)
  good=dict(format="poseidon-source-live-case-report-v2",case=p.name,request_id="r",plan_binding=self.plan["binding"],
   case_started_binding=run.read(p/"case-started.json")["binding"],status="failed",new_agent_generation=True)
  for key,value in dict(case="other",request_id="other",plan_binding="other",case_started_binding="other").items():
   row=dict(good);row[key]=value;run.write(p/"case-report.json",row)
   with self.assertRaisesRegex(ValueError,"case_receipt_identity"):run.checked_receipt(p,p.name,self.plan,case)
 def test_missing_registered_state_is_not_zero_calls(self):
  p=self.folder();run.write(p/"task-id.json",dict(task_id="a"*32))
  with self.assertRaisesRegex(ValueError,"registered_call_state_inventory"):run.states(self.output)
  data=run.save(self.output,self.plan,[],"running")
  self.assertEqual(data["status"],"stopped");self.assertIsNone(data["http_attempts"]);self.assertFalse(data["calls_resolved"])
 def test_unknown_state_refused(self):
  p=self.folder();t=p/"tasks"/("a"*32);t.mkdir(parents=True);run.write(t/"state.json",{})
  with self.assertRaisesRegex(ValueError,"registered_call_state_inventory"):run.states(self.output)
 def test_inside_writer_reaches_actual_case_reader_without_backend(self):
  # Only the backend and process execution are stubs. All boundary files,
  # exclusive markers, receipt identity, ownership and state readers are real.
  envmod=types.SimpleNamespace(VENV=Path(sys.prefix))
  backend=types.SimpleNamespace(PoseidonBackend=lambda:types.SimpleNamespace(identity=lambda:{}))
  def boundary(output,plan,case=None):
   run.validate_plan(plan,self.proposal_data)
   if case is not None:
    run.supervisor_current(output,plan);run.case_boundary(output,case,plan,self.proposal_data)
   return self.proposal_data
  class Child:
   returncode=0;pid=999999999
   def __init__(child,argv,**kwargs):
    name=argv[-1];folder=self.output/name
    run.reserve_case(folder,name,self.plan)
    row=dict(format="poseidon-source-live-case-report-v2",case=name,request_id="r",plan_binding=self.plan["binding"],
     case_started_binding=run.read(folder/"case-started.json")["binding"],status="failed",new_agent_generation=True,failure_layer="input")
    run.write(folder/"case-report.json",row)
   def poll(child):return 0
   def wait(child,**kwargs):return 0
  with mock.patch.dict(sys.modules,{"hecate_python_env":envmod,"application_backend":backend}),mock.patch.dict(os.environ,{"IN_NIX_SHELL":"pure"}),mock.patch.object(run,"boundary",side_effect=boundary),mock.patch.object(run.subprocess,"Popen",Child),mock.patch("builtins.print"):
   self.assertEqual(run.inside(self.output),1)
  for name in run.CASES:
   run.case_boundary(self.output,name,self.plan,self.proposal_data)

class Lifecycle(unittest.TestCase):
 def child(self,source,root):
  return subprocess.Popen([sys.executable,"-B","-c",source,str(root)],cwd=root,start_new_session=True,
   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 def wait_file(self,path):
  end=time.monotonic()+3
  while not path.exists() and time.monotonic()<end:time.sleep(.02)
  self.assertTrue(path.exists())
 def test_normal_exit_double_fork_setsid_background_killed(self):
  with tempfile.TemporaryDirectory(dir="/tmp") as d:
   root=Path(d);tree=run.OwnedTree().start()
   source="import os,sys,time;from pathlib import Path;pid=os.fork();\\nif pid==0:\\n os.setsid();pid2=os.fork();\\n if pid2==0:\\n  Path(sys.argv[1]+'/background').write_text(str(os.getpid()));time.sleep(60)\\n os._exit(0)\\nos._exit(0)"
   proc=self.child(source.replace("\\n","\n"),root)
   try:
    self.wait_file(root/"background");pid=int((root/"background").read_text());identity=run.process_identity(pid);proc.wait(timeout=3)
    self.assertTrue(tree.finish(exclude_reap=(proc.pid,)));self.assertFalse(run.same_process(identity))
   finally:
    tree.finish(exclude_reap=(proc.pid,))
    if proc.poll() is None:proc.kill();proc.wait()
 def test_ignored_term_separate_session_and_worker_cleanup(self):
  with tempfile.TemporaryDirectory(dir="/tmp") as d:
   root=Path(d);tree=run.OwnedTree().start()
   source="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True);Path(sys.argv[1]+'/background').write_text(str(p.pid));time.sleep(60)"
   proc=self.child(source,root)
   try:
    self.wait_file(root/"background");identity=run.process_identity(int((root/"background").read_text()))
    self.assertTrue(tree.finish(exclude_reap=(proc.pid,)));proc.wait(timeout=3)
    self.assertFalse(run.same_process(identity));self.assertNotEqual(proc.returncode,0)
   finally:
    tree.finish(exclude_reap=(proc.pid,))
    if proc.poll() is None:proc.kill();proc.wait()
 def test_supervisor_sigkill_orphans_remain_owned_by_outer_guardian(self):
  with tempfile.TemporaryDirectory(dir="/tmp") as d:
   root=Path(d);tree=run.OwnedTree().start()
   source="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True);Path(sys.argv[1]+'/background').write_text(str(p.pid));time.sleep(60)"
   proc=self.child(source,root)
   try:
    self.wait_file(root/"background");identity=run.process_identity(int((root/"background").read_text()))
    proc.kill();proc.wait(timeout=3)
    self.assertTrue(tree.finish(exclude_reap=(proc.pid,)));self.assertFalse(run.same_process(identity))
   finally:
    tree.finish(exclude_reap=(proc.pid,))
    if proc.poll() is None:proc.kill();proc.wait()
 def test_interrupt_handler_raises_for_term_hup_int(self):
  for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
   with self.subTest(signal=sig),self.assertRaises(InterruptedError):run.interrupted(sig,None)
 def test_pid_starttime_mismatch_not_signalled(self):
  tree=run.OwnedTree()
  tree.owned={123:1}
  with mock.patch.object(tree,"living",return_value=[123]),mock.patch.object(run,"process_info",return_value={"start":2}),mock.patch.object(run.os,"kill") as kill:
   tree.signal(signal.SIGKILL);kill.assert_not_called()
 def test_independent_lock_is_held_for_entire_subprocess(self):
  with tempfile.TemporaryDirectory(dir="/tmp") as d:
   root=Path(d);lock=root/"execution.lock"
   with run.owned_lock(lock):
    cmd="import fcntl,sys;f=open(sys.argv[1],'a+');\\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\\nexcept BlockingIOError:sys.exit(7)"
    value=subprocess.run([sys.executable,"-B","-c",cmd.replace("\\n","\n"),str(lock)],cwd=root,timeout=3)
    self.assertEqual(value.returncode,7)
   value=subprocess.run([sys.executable,"-B","-c","import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",str(lock)],cwd=root,timeout=3)
   self.assertEqual(value.returncode,0)
if __name__=="__main__":unittest.main()
