import unittest,copy
from qualification_scope import calls,bound,digest,read,require
class FinancialAndIntegrityControls(unittest.TestCase):
 def state(self,g=1,h=0,outcomes=None):
  s=dict(generations=g,http_attempts=h,ledger=[dict(event="http_reserved",index=i)for i in range(h)])
  if outcomes is not None:s["ledger"].append(dict(event="provider_checkpoint",outcomes=outcomes))
  return s
 def test_no_http_rule(self):self.assertEqual(calls([self.state()],3,0),dict(generations=1,http_attempts=0,calls_resolved=True))
 def test_resolved_response(self):
  s=self.state(h=1,outcomes=[dict(index=0,status="response_received")])
  self.assertTrue(calls([s],12,48)["calls_resolved"])
 def test_resolved_failure(self):
  self.assertTrue(calls([self.state(h=1,outcomes=[dict(index=0,status="failed")])],12,48)["calls_resolved"])
 def test_unknown_http_stops_next_case(self):self.assertFalse(calls([self.state(h=1)],12,48)["calls_resolved"])
 def test_outcome_sequence(self):
  self.assertFalse(calls([self.state(h=1,outcomes=[dict(index=2,status="response_received")])],12,48)["calls_resolved"])
 def test_duplicate_reservation(self):
  s=self.state(h=1);s["ledger"].append(s["ledger"][0])
  with self.assertRaisesRegex(ValueError,"reservation_ledger"):calls([s],12,48)
 def test_case_generation_ceiling(self):
  with self.assertRaisesRegex(ValueError,"case_calls"):calls([self.state(g=5)],12,48)
 def test_campaign_generation_ceiling(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_budget"):calls([self.state(g=4)]*4,12,48)
 def test_campaign_http_ceiling(self):
  with self.assertRaisesRegex(ValueError,"campaign_call_budget"):calls([self.state(h=16)]*4,16,48)
 def test_boolean_count_rejected(self):
  with self.assertRaisesRegex(ValueError,"case_calls"):calls([self.state(g=True)],12,48)
 def test_model_binding_changes(self):
  v=dict(model_sha256="old",maximum=4);v["binding"]=digest(v);bound(v)
  v["model_sha256"]="changed"
  with self.assertRaisesRegex(ValueError,"binding"):bound(v)
 def test_closed_positive_boolean(self):
  with self.assertRaisesRegex(ValueError,"request"):require(False,"request")
