"""Unpaid supervisor and argument controls. No Provider constructed."""
import unittest,sys,os,time,subprocess,tempfile
from unittest.mock import patch
from pathlib import Path
import run,scope
class RunTests(unittest.TestCase):
 def setUp(self):
  # A real file fixture supplies hashes before the paid proposal can be frozen.
  # These unit tests exercise controls, never source intake or backend execution.
  self.temp=tempfile.TemporaryDirectory(prefix="poseidon-normalized-control-")
  self.addCleanup(self.temp.cleanup)
  fixture=Path(self.temp.name)/"proposal.json"
  fixture.write_text(__import__("json").dumps(self.proposal()))
  self.proposal_patch=patch.object(run,"PROPOSAL",fixture)
  self.proposal_patch.start();self.addCleanup(self.proposal_patch.stop)
 def proposal(self):
  return dict(binding=scope.PROPOSAL_BINDING,provider=scope.PROVIDER,budget=scope.BUDGET)
 def plan(self):
  public=run.public_plan(self.proposal());now=time.time()
  p=dict(public,approved=True,approval_record=str(scope.AUTHORIZATION),approval_sha256="0"*64,
   created=now,deadline_wall=now+scope.BUDGET["batch_wall_seconds"],initial_result_directories=[],owner=dict(pid=os.getpid(),start=1))
  p["binding"]=scope.digest(p);return p
 def test_no_approval_no_execution(self):
  with patch.object(run,"verify",side_effect=AssertionError("must not inspect production")),patch.dict(sys.modules,{"agent_credentials":None,"deepseek_provider":None}):
   with self.assertRaisesRegex(ValueError,"approval_record_required"):run.execute(scope.RUN,None)
 def test_wrong_root_rejected(self):
  with self.assertRaisesRegex(ValueError,"fixed_campaign_root"):run.execute(scope.RUN.parent/"other",scope.AUTHORIZATION)
 def test_existing_root_rejected(self):
  with patch.object(Path,"exists",return_value=True):
   with self.assertRaisesRegex(ValueError,"fresh_campaign_root"):run.execute(scope.RUN,scope.AUTHORIZATION)
 def test_other_campaign_supervisor_blocks_execution(self):
  script=scope.BASE/"benchmarks/tools/spatial_source_live_v1/run.py"
  self.assertTrue(run.active_project_command((b"python3",str(script).encode(),b"--execute",b"/tmp/results")))
 def test_supervisor_inside_and_case_block_execution(self):
  script=scope.BASE/"benchmarks/tools/other_fixture/run.py"
  for flag in ("--inside","--case"):
   with self.subTest(flag=flag):self.assertTrue(run.active_project_command(("python3",str(script),flag)))
 def test_read_only_plan_is_not_execution(self):
  for script in (run.__file__,str(scope.BASE/"benchmarks/tools/spatial_source_live_v1/run.py")):
   self.assertFalse(run.active_project_command(("python3",script,"--plan")))
 def test_cpu_executable_spellings_block_execution(self):
  for name in ("poseidon_agent_cpu","poseidon-agent-cpu"):
   with self.subTest(name=name):self.assertTrue(run.active_project_command((str(scope.WORK/name),)))
 def test_source_and_transport_workers_block_execution(self):
  for name in ("candidate_worker.py","deepseek_http_worker.py","candidate_trace.py"):
   with self.subTest(name=name):self.assertTrue(run.active_project_command(("python3",str(scope.BASE/name))))
 def test_unrelated_execution_does_not_block(self):
  self.assertFalse(run.active_project_command(("python3","/tmp/another-project/run.py","--execute")))
 def test_null_arguments_do_not_block(self):
  self.assertFalse(run.active_project_command((b"",)))
 def test_public_cases(self):self.assertEqual(run.public_plan(self.proposal())["cases"],list(scope.CASES))
 def test_public_budget(self):self.assertEqual(run.public_plan(self.proposal())["budget"],scope.BUDGET)
 def test_hidden_approval_path(self):
  with self.assertRaisesRegex(ValueError,"approval_record_path"):run.approval_path(Path("/tmp/.hidden.json"))
 def test_plan_unknown_field(self):
  p=self.plan();p["extra"]=1
  with self.assertRaisesRegex(ValueError,"execution_plan_fields"):run.validate_plan(p,self.proposal())
 def test_plan_tamper(self):
  p=self.plan();p["budget"]=dict(p["budget"],max_generations=13)
  with self.assertRaisesRegex(ValueError,"execution_plan_binding"):run.validate_plan(p,self.proposal())
 def test_plan_extended_deadline(self):
  p=self.plan();p["deadline_wall"]+=1;p["binding"]=scope.digest({k:v for k,v in p.items() if k!="binding"})
  with self.assertRaisesRegex(ValueError,"original_batch_deadline"):run.validate_plan(p,self.proposal())
 def test_pid_identity(self):
  info=run.process_identity(os.getpid());self.assertTrue(run.same_process(info))
  info["start"]+=1;self.assertFalse(run.same_process(info))
 def test_owned_tree_only_owned_child(self):
  tree=run.OwnedTree().start()
  child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"],start_new_session=True)
  try:
   time.sleep(.15);self.assertIn(child.pid,tree.living())
   self.assertTrue(tree.finish(exclude_reap=(child.pid,)))
   child.wait(timeout=3);self.assertIsNotNone(child.returncode)
  finally:
   if child.poll() is None:child.kill();child.wait()

 def test_approval_file_absent_does_not_construct_provider(self):
  with patch.dict(sys.modules,{"agent_credentials":None,"deepseek_provider":None}):
   with self.assertRaisesRegex(ValueError,"approval_record_path"):
    run.execute(scope.RUN,scope.AUTHORIZATION)

 def test_transport_receipt_normalized_roundtrip_preserves_diagnostics(self):
  # Validate the sidecar; the older billing format intentionally omits stream counters.
  sys.path.insert(0,str(scope.BASE))
  from application_transport_receipt import snapshot
  raw=dict(provider="deepseek_api",calls=[dict(index=0,generation_index=0,retry_index=0,status="failed",
   error="transport_invalid_stream",transport_diagnostics=dict(received_bytes=123,sse_events=4))],
   request_attempts=1,generation_attempts=1,transport_retries=0)
  receipt=snapshot(raw)
  self.assertEqual(receipt["calls"][0]["transport_diagnostics"],dict(received_bytes=123,sse_events=4))
  self.assertEqual(snapshot(receipt),receipt)


if __name__=="__main__":unittest.main()

