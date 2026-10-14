"""Cumulative quota and immutable public-request boundaries for SiLU continuation."""
import sys,unittest,copy,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
from stage2_agent_repair_plan_r196 import build,verify,NAME,REVIEW,AUTH_NAME
from workspace_paths import RESULTS
from campaign_live_state import sealed
from campaign_budget_claim import claim
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.p=build(REVIEW,RESULTS/AUTH_NAME,RESULTS/NAME)
 def test_exact_scope_and_cumulative_cap(self):
  p=self.p;self.assertEqual(p["planned"],1);self.assertEqual(p["case_generations_used"]+p["maximum_generations"],4)
  self.assertEqual(p["prior_generations"]+p["maximum_generations"],57)
  self.assertEqual(p["prior_http_attempts"]+p["maximum_http_attempts"],63)
  self.assertEqual(p["shards"][0]["plan"]["cases"][0]["id"],"free_bench_helper_0113")
 def test_only_diagnostic_code_changed(self):
  self.assertEqual(self.p["source_change"],["scripts/baseline/decorated_functions.py","scripts/baseline/test_native_diagnostics.py"])
  self.assertEqual(len(self.p["compiler_guard"]),93)
 def test_claim_and_args_are_remaining_quota(self):
  shard=self.p["shards"][0]["plan"];spec=shard["cases"][0];args=spec["candidate_arguments"]
  self.assertEqual(args[args.index("--max-repairs")+1],"1");self.assertIn("--live",args);self.assertNotIn("--replay",args)
  with tempfile.TemporaryDirectory() as d:
   self.assertTrue(claim(Path(d)/"claims",spec,self.p["source_hashes"],Path(d),shard["binding"],shard["paid_configuration"])[0])
 def test_rehash_cannot_expand_quota(self):
  p={k:v for k,v in self.p.items() if k!="binding"};p["maximum_generations"]=3
  with self.assertRaises(ValueError):verify(sealed(p),RESULTS/NAME)
if __name__=="__main__":unittest.main(verbosity=2)
