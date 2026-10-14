"""Offline execution authorization and public-data boundary regression tests."""
import copy,json,sys,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
from stage2_agent_repair_plan_r187 import build,authorization,verify,REVIEW,NAME,AUTH_NAME
from workspace_paths import RESULTS
from stage2_agent_repair_r187 import candidate_command
class RetryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.plan=build(REVIEW,RESULTS/AUTH_NAME,RESULTS/NAME)
 def test_exact_denominator_and_budget(self):
  p=self.plan;cases=[s for r in p["shards"] for s in r["plan"]["cases"]]
  self.assertEqual(len(cases),28);self.assertEqual(len({s["id"] for s in cases}),28)
  self.assertEqual(sum(s["max_repairs"]+1 for s in cases),60)
  self.assertEqual(sum(r["plan"]["limits"]["maximum_http_attempts"] for r in p["shards"]),240)
  self.assertEqual([len(r["plan"]["cases"]) for r in p["shards"]],[2,9,9,8])
 def test_no_source_answer_in_dispatch(self):
  for ref in self.plan["shards"]:
   for s in ref["plan"]["cases"]:
    def walk(x):
     if isinstance(x,dict):
      self.assertNotIn("candidate",x)
      for v in x.values():walk(v)
     elif isinstance(x,list):
      for v in x:walk(v)
     elif isinstance(x,str):self.assertFalse(x.startswith("@hc.func") and "\n    " in x)
    walk(s["request"])
    self.assertFalse(s["manual_repair_source_sent"])
    args=candidate_command(ref["plan"],s,RESULTS/"dummy")
    self.assertIn("--live",args);self.assertNotIn("--replay",args)
    self.assertEqual(args[args.index("--max-repairs")+1],str(ref["plan"]["paid_configuration"]["max_repairs"]))
 def test_rehashing_budget_change_does_not_grant_authority(self):
  from benchmark_graph import digest
  p=copy.deepcopy(self.plan);p["maximum_generations"]=112
  p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
  with self.assertRaises(ValueError):verify(p,RESULTS/NAME)
 def test_new_claims_and_current_authorization(self):
  a=authorization()
  self.assertIn("剩余的所有失败项",a["user_instruction"])
  self.assertTrue(a["old_authorizations_not_reused"])
  self.assertIn("r187",self.plan["claim_registry"])
  self.assertEqual(len(self.plan["compiler_guard"]),93)
  self.assertEqual(len(self.plan["checker_guard"]),1)
if __name__=="__main__":unittest.main(verbosity=2)
