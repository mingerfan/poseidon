"""Selection/closure checks only; no network or provider."""
import copy,unittest
from accounting import select_unstarted,closed_calls,deadline
class Controls(unittest.TestCase):
 def setUp(self):
  self.cases=[dict(case=dict(id="a",request_id="1")),dict(case=dict(id="b",request_id="2"))]
  self.row=dict(id="a",request_id="1",status="failed",
      result=dict(phase="complete",status="service_error",generations=3,http_attempts=3))
 def test_never_retry_failed(self):self.assertEqual(select_unstarted(self.cases,[self.row]),[self.cases[1]])
 def test_success_also_not_replayed(self):
  self.row.update(status="passed");self.row["result"]["status"]="succeeded"
  self.assertEqual(select_unstarted(self.cases,[self.row]),[self.cases[1]])
 def test_original_binding_not_mutated(self):
  before=copy.deepcopy(self.cases);select_unstarted(self.cases,[self.row]);self.assertEqual(before,self.cases)
 def test_running_rejected(self):
  self.row["status"]="running"
  with self.assertRaises(ValueError):select_unstarted(self.cases,[self.row])
 def test_extra_case_rejected(self):
  self.row["id"]="unknown"
  with self.assertRaises(ValueError):select_unstarted(self.cases,[self.row])
 def test_changed_request_rejected(self):
  self.row["request_id"]="changed"
  with self.assertRaises(ValueError):select_unstarted(self.cases,[self.row])
 def test_spare_quota_not_transferred(self):
  self.row["result"]["generations"]=5
  with self.assertRaises(ValueError):select_unstarted(self.cases,[self.row])
 def test_duplicate_row_rejected(self):
  with self.assertRaises(ValueError):select_unstarted(self.cases,[self.row,self.row])
 def test_closed_provider_error(self):
  s=dict(phase="complete",status="service_error",http_attempts=1,
     ledger=[dict(event="provider_checkpoint",attempts=1,outcomes=[dict(index=0,status="failed",error="transport_invalid_stream")])])
  self.assertTrue(closed_calls(s))
 def test_uncertain_provider_rejected(self):
  s=dict(phase="complete",status="service_error",http_attempts=1,
     ledger=[dict(event="provider_checkpoint",attempts=1,outcomes=[dict(index=0,status="in_flight")])])
  self.assertFalse(closed_calls(s))
 def test_unchanged_deadline(self):self.assertEqual(deadline(100,90),100)
 def test_expired_deadline_never_reset(self):
  with self.assertRaises(ValueError):deadline(100,101)
 def test_unstarted_group_has_no_previous_deadline(self):self.assertIsNone(deadline(None,100))
if __name__=="__main__":unittest.main(verbosity=2)
