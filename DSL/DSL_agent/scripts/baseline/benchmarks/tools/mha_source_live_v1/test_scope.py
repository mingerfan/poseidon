"""Unpaid exact authority, request scope and call-accounting controls."""
import copy,unittest
from unittest.mock import patch
from pathlib import Path
import scope
class ScopeTests(unittest.TestCase):
 def setUp(self):self.p=scope.read(scope.PROPOSAL)
 def changed(self,fn):
  p=copy.deepcopy(self.p);fn(p);p["binding"]=scope.digest({k:v for k,v in p.items() if k!="binding"})
  with self.assertRaises(ValueError):scope.validate(p)
 def test_original_proposal(self):self.assertEqual(scope.validate(self.p),self.p)
 def test_source_case_order(self):self.changed(lambda p:p["cases"].reverse())
 def test_add_case(self):self.changed(lambda p:p["cases"].append(p["cases"][0]))
 def test_remove_case(self):self.changed(lambda p:p["cases"].pop())
 def test_model_name(self):self.changed(lambda p:p["provider"].update(model="other"))
 def test_budget(self):self.changed(lambda p:p["bounds"].update(max_generations=13))
 def test_budget_bool(self):self.changed(lambda p:p["bounds"].update(compiler_jobs=True))
 def test_threshold(self):self.changed(lambda p:p["numeric_tolerance"].update(atol=1e-4))
 def test_validation_grade(self):self.changed(lambda p:p["cases"][0].update(validation_level="compiled"))
 def test_missing_self_binding(self):
  p=copy.deepcopy(self.p);p.pop("binding")
  with self.assertRaises(ValueError):scope.validate(p)
 def test_duplicate_json(self):
  with self.assertRaises(ValueError):scope.decode(b'{"a":1,"a":2}')
 def test_nonfinite_json(self):
  with self.assertRaises(ValueError):scope.decode(b'{"a":NaN}')
 def test_missing_approval(self):
  with self.assertRaises(ValueError):scope.authority({},self.p,scope.PROPOSAL_SHA)
 def test_pending_calls(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0)])
  self.assertFalse(scope.totals([s])["calls_resolved"])
 def test_resolved_calls(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0),dict(event="provider_checkpoint",outcomes=[dict(index=0,status="response_received")])])
  self.assertEqual(scope.totals([s]),dict(generations=1,http_attempts=1,calls_resolved=True))
 def test_unknown_outcome(self):
  s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0),dict(event="provider_checkpoint",outcomes=[dict(index=0,status="unknown")])])
  self.assertFalse(scope.totals([s])["calls_resolved"])
 def test_http_bound(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=1,http_attempts=17,ledger=[])])
 def test_generation_bound(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=5,http_attempts=0,ledger=[])])
 def test_ledger_gap(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=1)])])
 def test_boolean_accounting(self):
  with self.assertRaises(ValueError):scope.totals([dict(generations=True,http_attempts=0,ledger=[])])
 def test_no_calls(self):self.assertTrue(scope.totals([])["calls_resolved"])
 def test_failure_not_pass(self):
  r=scope.terminal([dict(case=scope.CASES[0],status="failed")],[])
  self.assertEqual((r["passed"],r["failed"],r["not_run"],r["status"]),(0,1,2,"stopped"))
 def test_duplicate_terminal_case(self):
  with self.assertRaises(ValueError):scope.terminal([dict(case=scope.CASES[0],status="passed")]*2,[])
 def test_illegal_terminal_case(self):
  with self.assertRaises(ValueError):scope.terminal([dict(case="linear",status="passed")],[])
 def test_regular_symlink_scope(self):
  with self.assertRaises(ValueError):scope.regular(Path("/etc/passwd"))
 def test_tool_manifest_cannot_add_file(self):
  m=dict(format="poseidon-mha-source-live-runner-v1",proposal_binding=scope.PROPOSAL_BINDING,proposal_sha256=scope.PROPOSAL_SHA,tool_hashes={"x":"y"})
  m["binding"]=scope.digest(m)
  with patch.object(scope,"tools",return_value={}):
   with self.assertRaises(ValueError):scope.check_manifest(m)
if __name__=="__main__":unittest.main()
