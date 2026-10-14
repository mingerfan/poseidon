import copy,json,unittest
from pathlib import Path
import replay_scope as scope
class ReplayScope(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.origin=scope.read(scope.origin_scope.PROPOSAL)
  cls.cases=scope.aliases(cls.origin)
 def proposal(self):
  p=dict(format="poseidon-normalized-source-replay-proposal-v1",status="frozen_plan",
   origin_proposal_binding=self.origin["binding"],stage_identity=self.origin["source_stage_identity"],
   bounds=copy.deepcopy(scope.BOUNDS),paid_generations=0,http_attempts=0,new_agent_generation=False,
   cache_cross_source_reuse=False,cases=copy.deepcopy(self.cases),numeric_tolerance=dict(atol=1e-5,rtol=1e-4))
  for c in p["cases"]:c.update(status="ready",origin=dict(agent_generated=True,manual_substitution=False))
  p["binding"]=scope.digest(p);return p
 def reject(self,change):
  p=self.proposal();change(p);p["binding"]=scope.digest({k:v for k,v in p.items()if k!="binding"})
  with self.assertRaises(ValueError):scope.validate(p,self.origin)
 def test_original_scope(self):self.assertEqual(len(scope.validate(self.proposal(),self.origin)["cases"]),25)
 def test_verified_original_remains_frozen(self):
  original=scope.checked_origin()
  self.assertEqual(original,self.origin)
  self.assertEqual(len(scope.aliases(original)),25)
 def test_execution_view_is_not_serialized_origin(self):
  view=scope.origin_scope.execution_view(self.origin,dict(tool_hashes={}))
  with self.assertRaisesRegex(ValueError,"replay_original_proposal_binding"):scope.aliases(view)
 def test_no_new_generation(self):self.reject(lambda p:p.update(paid_generations=1))
 def test_no_http(self):self.reject(lambda p:p.update(http_attempts=1))
 def test_no_human_substitution(self):self.reject(lambda p:p["cases"][0]["origin"].update(manual_substitution=True))
 def test_no_cross_source_cache(self):self.reject(lambda p:p.update(cache_cross_source_reuse=True))
 def test_no_new_agent_attribution(self):self.reject(lambda p:p.update(new_agent_generation=True))
 def test_no_tolerance_change(self):self.reject(lambda p:p["numeric_tolerance"].update(atol=1e-3))
 def test_no_budget_extension(self):self.reject(lambda p:p["bounds"].update(new_result_bytes=scope.BOUNDS["new_result_bytes"]+1))
 def test_no_missing_source(self):self.reject(lambda p:p["cases"].pop())
 def test_no_changed_source(self):self.reject(lambda p:p["cases"][0]["source"].update(request_id="0"*64))
 def test_blocked_is_not_passed(self):
  rows=[dict(id=c["id"],status="blocked_origin_failure")for c in self.cases]
  v=scope.terminal(rows,self.cases);self.assertEqual((v["passed"],v["failed"],v["blocked_origin_failure"]),(0,0,25))
 def test_not_run_is_not_passed(self):
  v=scope.terminal([],self.cases);self.assertEqual((v["passed"],v["not_run"]),(0,25))
 def test_duplicate_receipt(self):
  with self.assertRaises(ValueError):scope.terminal([dict(id=self.cases[0]["id"],status="passed")]*2,self.cases)
if __name__=="__main__":unittest.main()
