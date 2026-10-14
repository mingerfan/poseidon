import copy,unittest
from accounting import select_unstarted,closed_calls,deadline,merge_closed
class Controls(unittest.TestCase):
 def setUp(self):
  self.cases=[dict(case=dict(id="a",request_id="1")),dict(case=dict(id="b",request_id="2")),dict(case=dict(id="c",request_id="3"))]
  self.row=dict(id="b",request_id="2",status="failed",result=dict(phase="complete",status="service_error",generations=3,http_attempts=3))
  self.prior=dict(id="a",request_id="1",status="passed",generations=1,http_attempts=1,calls_resolved=True,not_retried=True)
  self.add=dict(id="b",request_id="2",status="failed",generations=3,http_attempts=3,calls_resolved=True,not_retried=True)
 def test_failed_never_replayed(self):self.assertEqual(select_unstarted(self.cases[1:],[self.row]),[self.cases[2]])
 def test_prior_and_current_closure(self):self.assertEqual(len(merge_closed([self.prior],[self.add],self.cases,[self.cases[2]])),2)
 def test_fresh_inputs_unchanged(self):
  before=copy.deepcopy(self.cases);merge_closed([self.prior],[self.add],self.cases,[self.cases[2]]);self.assertEqual(before,self.cases)
 def test_no_closed_quota_transfer(self):
  self.add["generations"]=5
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[self.cases[2]])
 def test_no_replay_of_prior(self):
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[self.cases[0],self.cases[2]])
 def test_missing_context(self):
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[])
 def test_prior_binding_change(self):
  self.prior["request_id"]="changed"
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[self.cases[2]])
 def test_pending_binding_change(self):
  other=copy.deepcopy(self.cases[2]);other["case"]["request_id"]="changed"
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[other])
 def test_duplicate_closed(self):
  with self.assertRaises(ValueError):merge_closed([self.prior,self.prior],[self.add],self.cases,[self.cases[2]])
 def test_unresolved_closed(self):
  self.add["calls_resolved"]=False
  with self.assertRaises(ValueError):merge_closed([self.prior],[self.add],self.cases,[self.cases[2]])
 def test_open_task(self):
  self.row["result"]["phase"]="generating"
  with self.assertRaises(ValueError):select_unstarted(self.cases[1:],[self.row])
 def test_unknown_task(self):
  self.row["id"]="extra"
  with self.assertRaises(ValueError):select_unstarted(self.cases[1:],[self.row])
 def test_success_not_replayed(self):
  self.row["status"]="passed";self.row["result"]["status"]="succeeded"
  self.assertEqual(select_unstarted(self.cases[1:],[self.row]),[self.cases[2]])
 def test_confirmed_failed_provider(self):
  s=dict(phase="complete",status="service_error",http_attempts=1,ledger=[dict(event="provider_checkpoint",attempts=1,outcomes=[dict(index=0,status="failed")])]);self.assertTrue(closed_calls(s))
 def test_inflight_provider(self):
  s=dict(phase="complete",status="service_error",http_attempts=1,ledger=[dict(event="provider_checkpoint",attempts=1,outcomes=[dict(index=0,status="in_flight")])]);self.assertFalse(closed_calls(s))
 def test_original_deadline(self):self.assertEqual(deadline(100,90),100)
 def test_no_deadline_extension(self):
  with self.assertRaises(ValueError):deadline(100,101)
 def test_unstarted_attention(self):self.assertIsNone(deadline(None,100))
if __name__=="__main__":unittest.main(verbosity=2)
