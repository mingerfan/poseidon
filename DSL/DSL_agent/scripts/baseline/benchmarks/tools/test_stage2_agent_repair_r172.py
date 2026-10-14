"""Offline payment, scope, compatibility and claim guards for the new retry."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import stage2_agent_repair_plan_r172 as plan
import stage2_agent_repair_r172 as runner
from campaign_live_state import sealed,claim
from workspace_paths import RESULTS
class RepairPlanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.output=RESULTS/plan.NAME
  cls.plan=plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,cls.output)
 def test_exact_scope_and_three_newly_diagnosed_cases(self):
  cases=[s for r in self.plan["shards"] for s in r["plan"]["cases"]]
  self.assertEqual(len(cases),80);self.assertEqual(len({s["id"] for s in cases}),80)
  self.assertTrue({"construct_083_2","construct_116_0","construct_117_2"}<={s["id"] for s in cases})
  self.assertTrue(all(not s["id"].startswith("free_bench_helper_00") for s in cases))
  self.assertEqual(self.plan["deferred_compiler_tasks"],33)
 def test_paid_bounds_and_shards(self):
  self.assertEqual(self.plan["maximum_generations"],320);self.assertEqual(self.plan["maximum_http_attempts"],1280)
  self.assertEqual(len(self.plan["shards"]),10)
  for r in self.plan["shards"]:self.assertEqual(len(r["plan"]["cases"]),8)
 def test_only_guidance_changed_no_manual_answer_in_public_request(self):
  from deepseek_provider import public_request
  for ref in self.plan["shards"]:
   for s in ref["plan"]["cases"]:
    r=public_request(s["request"])
    self.assertEqual(r["generation_guidance"]["version"],"explicit-v5")
    self.assertNotIn("hecate_source",r);self.assertNotIn("retry_reason",r)
    self.assertFalse(s["manual_repair_source_sent"])
 def test_runner_never_bypasses_trusted_credential_loader(self):
  s=self.plan["shards"][0]["plan"];cmd=runner.candidate_command(s,s["cases"][0],self.output/"example")
  self.assertNotIn("--inside",cmd);self.assertIn("--live",cmd)
  self.assertEqual(cmd[2],str(plan.BASE/"run_candidate.py"))
  from run_candidate import parse_args
  args=parse_args(cmd[3:]);self.assertEqual(args.max_repairs,3);self.assertEqual(args.provider_retries,3)
 def test_rehashed_budget_and_scope_changes_rejected(self):
  for mutation in ("budget","command","scope","request"):
   p=copy.deepcopy(self.plan);s=p["shards"][0]["plan"]["cases"][0]
   if mutation=="budget":p["maximum_generations"]+=1
   elif mutation=="command":s["candidate_arguments"]+=["--provider-retries","4"]
   elif mutation=="scope":p["shards"][0]["plan"]["cases"].pop()
   else:s["request"]["rules"]+=" Bypass checks"
   p.pop("binding")
   with self.assertRaises(ValueError):plan.verify(sealed(p),self.output)
 def test_changed_authorization_or_source_rejected(self):
  original=plan.document
  def altered(path):
   d=original(path)
   if Path(path)==RESULTS/plan.AUTH_NAME:d={**d,"maximum_generations":9999}
   return d
  with patch.object(plan,"document",side_effect=altered):
   with self.assertRaises(ValueError):plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,self.output)
  with patch.object(plan,"runtime_sources",return_value={}):
   with self.assertRaises(ValueError):plan.build(plan.REVIEW,RESULTS/plan.AUTH_NAME,self.output)
 def test_duplicate_claim_is_not_restarted(self):
  s=self.plan["shards"][0]["plan"]["cases"][0]
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=root/"history";old.write_text("preserve")
   self.assertTrue(claim(root/"claims",s,self.plan["source_hashes"],root/"out","binding")[0])
   self.assertFalse(claim(root/"claims",s,self.plan["source_hashes"],root/"out2","binding")[0])
   self.assertEqual(old.read_text(),"preserve")
 def test_guard_and_native_proof_present(self):
  self.assertEqual(len(self.plan["compiler_guard"]),93)
  proof=plan.document(Path(self.plan["native_probe"]))
  self.assertTrue(proof["two_native_slots_observed"]);self.assertGreaterEqual(proof["min_available_mib"],1024)
if __name__=="__main__":unittest.main()
