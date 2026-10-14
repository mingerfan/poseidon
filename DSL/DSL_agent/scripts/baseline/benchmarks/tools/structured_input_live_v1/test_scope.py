"""Unpaid authority and accounting tests; no Provider import or construction."""
import copy,unittest
from unittest.mock import patch
import scope
class ScopeTests(unittest.TestCase):
 def setUp(self):self.p=scope.read(scope.PROPOSAL)
 def reject(self,fn):
  p=copy.deepcopy(self.p);fn(p);p["binding"]=scope.digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):scope.validate(p)
 def test_original(self):self.assertEqual(scope.validate(self.p),self.p)
 def test_reorder(self):self.reject(lambda p:p["cases"].reverse())
 def test_extra_case(self):self.reject(lambda p:p["cases"].append(p["cases"][0]))
 def test_missing_case(self):self.reject(lambda p:p["cases"].pop())
 def test_wrong_model(self):self.reject(lambda p:p["provider"].update(model="other"))
 def test_more_budget(self):self.reject(lambda p:p["bounds"].update(max_generations=49))
 def test_boolean_budget(self):self.reject(lambda p:p["bounds"].update(api_concurrency=True))
 def test_threshold(self):self.reject(lambda p:p["numeric_tolerance"].update(atol=1e-4))
 def test_compiled_only(self):self.reject(lambda p:p["cases"][0].update(validation_level="compiled"))
 def test_duplicate_json(self):
  with self.assertRaises(ValueError):scope.decode('{"x":1,"x":2}')
 def test_nonfinite_json(self):
  with self.assertRaises(ValueError):scope.decode('{"x":NaN}')
 def test_no_approval(self):
  with self.assertRaises(ValueError):scope.authority({},self.p,scope.PROPOSAL_SHA)
 def state(self,g=4,h=16):
  return dict(generations=g,http_attempts=h,ledger=[dict(event="http_reserved",index=i)for i in range(h)]+[
    dict(event="provider_checkpoint",outcomes=[dict(index=i,status="response_received")for i in range(h)])])
 def test_exact_ceiling(self):self.assertEqual(scope.totals([self.state()for _ in scope.CASES]),dict(generations=48,http_attempts=192,calls_resolved=True))
 def test_extra_calls(self):
  with self.assertRaises(ValueError):scope.totals([self.state()for _ in range(len(scope.CASES)+1)])
 def test_http_limit(self):
  with self.assertRaises(ValueError):scope.totals([self.state(h=17)])
 def test_generation_limit(self):
  with self.assertRaises(ValueError):scope.totals([self.state(g=5)])
 def test_unknown_call(self):
  s=self.state();s["ledger"][-1]["outcomes"].pop();self.assertFalse(scope.totals([s])["calls_resolved"])
 def test_reordered_ledger(self):
  s=self.state();s["ledger"][0]["index"]=2
  with self.assertRaises(ValueError):scope.totals([s])
 def test_failed_not_passed(self):
  r=scope.terminal([dict(case=scope.CASES[0],status="failed")],[self.state()])
  self.assertEqual((r["passed"],r["failed"],r["not_run"]),(0,1,len(scope.CASES)-1))
 def test_duplicate_case(self):
  with self.assertRaises(ValueError):scope.terminal([dict(case=scope.CASES[0],status="passed")]*2,[])
 def test_approval_cannot_use_old_format(self):
  with self.assertRaises(ValueError):scope.checked_authorization({},self.p,scope.PROPOSAL_SHA)
if __name__=="__main__":unittest.main()
