"""Offline tests of approval, reviewed scope and interrupted-call accounting."""
import copy,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from scope import *
def rebound(value):
 value=copy.deepcopy(value);value.pop("binding",None);value["binding"]=digest(value);return value
def state(g=0,h=0,statuses=None):
 ledger=[dict(event="http_reserved",index=i) for i in range(h)]
 if statuses is not None:
  ledger.append(dict(event="provider_checkpoint",outcomes=[dict(index=i,status=x)
   for i,x in enumerate(statuses)]))
 return dict(generations=g,http_attempts=h,ledger=ledger)
class ScopeTest(unittest.TestCase):
 def setUp(self):self.proposal=read(PROPOSAL)
 def receipt(self):
  return dict(format="poseidon-user-campaign-approval-v1",approved=True,
   proposal_binding=self.proposal["binding"],proposal_sha256=sha(PROPOSAL),
   authorization="human_explicit_case_budget_approval")
 def test_current_public_plan(self):
  verify(self.proposal)
  self.assertEqual(len(self.proposal["source_hashes"]),704)
 def test_changed_token_budget_is_rejected(self):
  p=copy.deepcopy(self.proposal);p["provider"]["max_tokens"]+=1
  with self.assertRaisesRegex(ValueError,"reviewed_provider_or_budget"):validate(rebound(p))
 def test_changed_wall_budget_is_rejected(self):
  p=copy.deepcopy(self.proposal);p["budget"]["batch_wall_seconds"]+=1
  with self.assertRaisesRegex(ValueError,"reviewed_provider_or_budget"):validate(rebound(p))
 def test_extra_model_is_rejected(self):
  p=copy.deepcopy(self.proposal);p["cases"].append(p["cases"][0])
  with self.assertRaisesRegex(ValueError,"reviewed_cases"):validate(rebound(p))
 def test_changed_request_is_rejected(self):
  p=copy.deepcopy(self.proposal);p["cases"][0]["request_id"]="0"*64
  with self.assertRaisesRegex(ValueError,"case_request_identity"):validate(rebound(p))
 def test_reviewed_unapproved_proposal_cannot_authorize(self):
  with self.assertRaises(ValueError):authority(self.proposal,self.proposal,sha(PROPOSAL))
 def test_unapproved_receipt_is_rejected(self):
  receipt=self.receipt();receipt["approved"]=False
  with self.assertRaisesRegex(ValueError,"approval_required"):authority(receipt,self.proposal,sha(PROPOSAL))
 def test_changed_binding_is_rejected(self):
  receipt=self.receipt();receipt["proposal_binding"]="0"*64
  with self.assertRaisesRegex(ValueError,"approval_required"):authority(receipt,self.proposal,sha(PROPOSAL))
 def test_changed_proposal_file_is_rejected(self):
  receipt=self.receipt();receipt["proposal_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"approval_required"):authority(receipt,self.proposal,sha(PROPOSAL))
 def test_complete_call_ledger(self):
  result=totals([state(4,16,["failed"]*15+["response_received"]) for _ in range(3)])
  self.assertEqual(result,dict(generations=12,http_attempts=48,calls_resolved=True))
 def test_cannot_roll_forward_unused_old_scope(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_ceiling"):
   totals([state(4,16,["response_received"]*16) for _ in range(4)])
 def test_per_case_ceiling(self):
  with self.assertRaisesRegex(ValueError,"case_call_ceiling"):totals([state(5,0)])
 def test_missing_reservation_is_rejected(self):
  s=state(1,2,["failed","response_received"]);s["ledger"].pop(0)
  with self.assertRaisesRegex(ValueError,"reservation_ledger"):totals([s])
 def test_unknown_call_stops_campaign(self):
  result=terminal([dict(status="passed"),dict(status="failed"),dict(status="failed")],
   [state(1,1,["response_received"]),state(1,1,["in_flight"])])
  self.assertEqual(result["status"],"stopped");self.assertFalse(result["calls_resolved"])
 def test_resolved_failure_is_not_counted_as_pass(self):
  result=terminal([dict(status="passed"),dict(status="failed"),dict(status="failed")],
   [state(1,1,["response_received"]) for _ in range(3)])
  self.assertEqual((result["status"],result["passed"],result["failed"],result["not_run"]),("complete",1,2,0))
 def test_unstarted_items_are_not_passes(self):
  result=terminal([dict(status="passed")],[state(1,1,["response_received"])])
  self.assertEqual((result["status"],result["passed"],result["not_run"]),("stopped",1,2))
 def test_duplicate_json_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"bad.json";p.write_text('{"approved":false,"approved":true}')
   with self.assertRaisesRegex(ValueError,"duplicate_json_field"):read(p)
 def test_nonfinite_json_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"bad.json";p.write_text('{"budget":NaN}')
   with self.assertRaisesRegex(ValueError,"nonfinite_json"):read(p)
 def test_preview_does_not_import_provider_or_credentials(self):
  code="""import json,sys;import run;value=run.preview();assert value["provider_calls"]==0 and value["approved"] is False;assert "agent_credentials" not in sys.modules and "deepseek_provider" not in sys.modules;print("offline_preview_only")"""
  p=subprocess.run([sys.executable,"-I","-B","-c",
   "import sys;sys.path.insert(0,"+repr(str(Path(__file__).parent))+");"+code],
   capture_output=True,text=True,timeout=30)
  self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),"offline_preview_only")
 def test_execute_rejects_before_creating_result_or_loading_credentials(self):
  import run
  target=run.RESULTS/"agent-v12-unauthorized-test-r1234"
  self.assertFalse(target.exists())
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/"approval.json";value=self.receipt();value["approved"]=False;p.write_text(json.dumps(value))
   with self.assertRaisesRegex(ValueError,"approval_required"):run.execute(target,p)
  self.assertFalse(target.exists());self.assertNotIn("agent_credentials",sys.modules)
 def test_stopping_worker_cleans_separate_session_descendants(self):
  import run,os,signal
  childcode="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
  parentcode="import subprocess,sys,time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);p=subprocess.Popen([sys.executable,'-c',"+repr(childcode)+"],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid,flush=True);time.sleep(60)"
  parent=subprocess.Popen([sys.executable,"-I","-B","-c",parentcode],stdout=subprocess.PIPE,text=True)
  pid=int(parent.stdout.readline().strip())
  try:
   run.stop(parent)
   self.assertIsNotNone(parent.poll())
   path=Path("/proc")/str(pid)/"stat"
   self.assertTrue(not path.exists() or path.read_text().rsplit(")",1)[1].split()[0]=="Z")
  finally:
   if parent.poll() is None:run.stop(parent)
   parent.stdout.close()
if __name__=="__main__":unittest.main(verbosity=2)
