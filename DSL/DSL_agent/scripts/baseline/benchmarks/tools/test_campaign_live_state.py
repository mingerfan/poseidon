"""Real filesystem duplicate/crash/tamper gates; no network/provider execution."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from campaign_live_state import claim,resume_rows,sealed,check
from stage2_agent_campaign import identity
from run_stage2_agent_campaign import validate_plan
from stage2_agent_pilot_plan import PAID,LIMITS
class LiveStateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.output=self.root/"batch";self.output.mkdir();self.sources={"x.py":"a"}
  self.spec=dict(id="task",model_sha256="m",request_id="r")
  self.spec["evaluation_identity"]=identity(self.spec,self.sources)
  self.plan=sealed(dict(cases=[self.spec]))
 def report(self,**kw):return sealed(dict(plan_binding=self.plan["binding"],rows=[],seconds=2.5,**kw))
 def terminal(self):
  evidence=self.root/"candidate";evidence.mkdir();p=evidence/"report.json";p.write_text(json.dumps(dict(status="passed")))
  row=dict(id="task",status="runner_reported_pass_pending_audit",evaluation_identity=self.spec["evaluation_identity"],
           evidence=str(evidence),report_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  return row,p
 def test_same_task_claim_only_once_across_output_directories(self):
  first,a=claim(self.root/"claims",self.spec,self.sources,self.output,self.plan["binding"])
  second,b=claim(self.root/"claims",self.spec,self.sources,self.root/"another",self.plan["binding"])
  self.assertTrue(first);self.assertFalse(second);self.assertEqual(a,b)
  self.assertFalse(a["automatic_retry_allowed"])
 def test_changed_source_needs_different_identity(self):
  with self.assertRaisesRegex(ValueError,"identity changed"):
   claim(self.root/"claims",self.spec,{"x.py":"new"},self.output,"p")
 def test_symlink_registry_rejected(self):
  d=self.root/"real";d.mkdir();(self.root/"claims").symlink_to(d)
  with self.assertRaisesRegex(ValueError,"Unsafe"):claim(self.root/"claims",self.spec,self.sources,self.output,"p")
 def test_corrupt_existing_claim_not_overwritten(self):
  claim(self.root/"claims",self.spec,self.sources,self.output,"p")
  p=next((self.root/"claims").glob("*.json"));p.write_text('{"binding":"bad"}')
  with self.assertRaises(ValueError):claim(self.root/"claims",self.spec,self.sources,self.output,"p")
  self.assertEqual(p.read_text(),'{"binding":"bad"}')
 def test_launch_without_terminal_record_cannot_resume(self):
  f=self.output/"task";f.mkdir();(f/"launch.json").write_text("{}")
  with self.assertRaisesRegex(ValueError,"Uncertain launched"):
   resume_rows(self.plan,self.report(),self.output)
 def test_dangling_launch_symlink_is_not_unlaunched(self):
  f=self.output/"task";f.mkdir();(f/"launch.json").symlink_to(self.root/"missing")
  with self.assertRaisesRegex(ValueError,"Uncertain launched"):resume_rows(self.plan,self.report(),self.output)
 def test_unlaunched_rows_preserve_cumulative_wall_usage(self):
  rows,seconds=resume_rows(self.plan,self.report(),self.output)
  self.assertEqual(rows,[]);self.assertEqual(seconds,2.5)
 def test_terminal_hash_verified_before_resume(self):
  row,p=self.terminal();r=sealed(dict(plan_binding=self.plan["binding"],rows=[row],seconds=10.))
  self.assertEqual(resume_rows(self.plan,r,self.output),([row],10.))
  p.write_text('{"status":"changed"}')
  with self.assertRaisesRegex(ValueError,"evidence changed"):resume_rows(self.plan,r,self.output)
 def test_rehashed_still_running_report_rejected(self):
  row,p=self.terminal();p.write_text('{"status":"running"}');row["report_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
  r=sealed(dict(plan_binding=self.plan["binding"],rows=[row],seconds=10.))
  with self.assertRaisesRegex(ValueError,"still running"):resume_rows(self.plan,r,self.output)
 def test_duplicate_terminal_rows_rejected(self):
  row,p=self.terminal();r=sealed(dict(plan_binding=self.plan["binding"],rows=[row,row],seconds=10.))
  with self.assertRaisesRegex(ValueError,"task identity"):resume_rows(self.plan,r,self.output)
 def test_wrong_plan_rejected(self):
  r=sealed(dict(plan_binding="other",rows=[],seconds=1.))
  with self.assertRaisesRegex(ValueError,"plan mismatch"):resume_rows(self.plan,r,self.output)
 def test_invalid_elapsed_usage_rejected(self):
  for seconds in (-1,True,"0"):
   r=sealed(dict(plan_binding=self.plan["binding"],rows=[],seconds=seconds))
   with self.subTest(seconds=seconds),self.assertRaisesRegex(ValueError,"wall usage"):resume_rows(self.plan,r,self.output)
 def test_plan_missing_approval_and_expanded_budget_rejected(self):
  for approval in ("","wrong"):
   with self.assertRaisesRegex(ValueError,"approved"):validate_plan(self.plan,approval)
  p=dict(format="poseidon-stage2-agent-campaign-shard-v1",cases=[self.spec],paid_configuration=PAID,
         limits=dict(LIMITS,maximum_generations=5,maximum_http_attempts=16))
  p=sealed(p)
  with self.assertRaisesRegex(ValueError,"Budget changed"):validate_plan(p,p["binding"])
 def test_importer_current_binding_refuses_historical_rebinding(self):
  # The resume identity itself must change when a request/configuration changes.
  changed=dict(self.spec,request_id="w40-request")
  self.assertNotEqual(identity(changed,self.sources),self.spec["evaluation_identity"])
if __name__=="__main__":unittest.main()
