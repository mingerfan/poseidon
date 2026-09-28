"""Offline execution authorization and public-data boundary regression tests."""
import copy,json,sys,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
from stage2_agent_repair_plan_r189 import build,authorization,verify,REVIEW,NAME,AUTH_NAME
from workspace_paths import RESULTS
from stage2_agent_repair_r189 import candidate_command
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
  self.assertIn("r189",self.plan["claim_registry"])
  self.assertEqual(len(self.plan["compiler_guard"]),93)
  self.assertEqual(len(self.plan["checker_guard"]),1)
 def test_all_real_claim_paths_and_duplicate_guard(self):
  import tempfile
  from campaign_budget_claim import claim
  from campaign_live_state import claim as legacy_claim
  for ref in self.plan["shards"]:
   shard=ref["plan"];s=shard["cases"][0]
   with tempfile.TemporaryDirectory() as d:
    registry=Path(d)/"claims"
    acquired,value=claim(registry,s,self.plan["source_hashes"],Path(d),shard["binding"],shard["paid_configuration"])
    self.assertTrue(acquired)
    self.assertFalse(claim(registry,s,self.plan["source_hashes"],Path(d),shard["binding"],shard["paid_configuration"])[0])
    altered=dict(shard["paid_configuration"],max_repairs=99)
    with self.assertRaises(ValueError):claim(registry,s,self.plan["source_hashes"],Path(d),shard["binding"],altered)
    if shard["paid_configuration"]["max_repairs"]==3:
     self.assertEqual(value["evaluation_identity"],s["evaluation_identity"])
     self.assertTrue(legacy_claim(Path(d)/"legacy",s,self.plan["source_hashes"],Path(d),shard["binding"])[0])
 def test_reporter_handles_unstarted_rows(self):
  import importlib.util
  p=BASE.parents[1]/"scripts/stage2_agent_repair_report_r190.py"
  spec=importlib.util.spec_from_file_location("report190",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertEqual(m.completion_layer({"status":"unaudited_or_not_run"}),"unknown")
if __name__=="__main__":unittest.main(verbosity=2)
