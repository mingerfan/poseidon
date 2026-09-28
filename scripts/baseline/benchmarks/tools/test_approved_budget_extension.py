"""Reject unauthorized retries, clock resets and changes to the approved 47 cases."""
import copy,json,unittest
from unittest.mock import patch
from pathlib import Path
from stage2_agent_campaign import ROOT
from workspace_paths import RESULTS
from campaign_live_state import sealed
from approved_budget_extension import verify_extension,check_selection
from budget_supplement_queue import verify,previous_terminal,shard_stop_kind

class BudgetTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.q=json.loads((RESULTS/"stage2-budget-supplement-queue-r127/plan.json").read_text())
  cls.plans=[json.loads(Path(r["plan"]).read_text()) for r in cls.q["shards"]]
 def test_all_47_cases_with_exact_original_clocks(self):
  verify(self.q);total=0
  for p in self.plans:
   v=verify_extension(p);total+=len(p["cases"])
   self.assertGreaterEqual(p["budget_extension"]["seconds_already_used"],3600)
   self.assertLess(v["limits"]["remaining_seconds"],3600)
  self.assertEqual(total,47)
 def test_request_or_model_change_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["cases"][0]["request_id"]="changed"
  with self.assertRaisesRegex(ValueError,"outside explicit authorization"):verify_extension(p)
 def test_added_case_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["cases"].append(p["cases"][0])
  with self.assertRaisesRegex(ValueError,"outside explicit authorization"):verify_extension(p)
 def test_original_clock_cannot_be_zeroed(self):
  p=copy.deepcopy(self.plans[0]);p["budget_extension"]["seconds_already_used"]=0
  with self.assertRaisesRegex(ValueError,"elapsed time reset"):verify_extension(p)
 def test_original_clock_cannot_be_reduced(self):
  p=copy.deepcopy(self.plans[0]);p["budget_extension"]["seconds_already_used"]-=1
  with self.assertRaisesRegex(ValueError,"elapsed time reset"):verify_extension(p)
 def test_extra_wall_budget_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["limits"]["max_wall_seconds"]=7201
  with self.assertRaisesRegex(ValueError,"cumulative limit"):verify_extension(p)
 def test_api_budget_inflation_rejected(self):
  for key in ("maximum_generations","maximum_http_attempts"):
   p=copy.deepcopy(self.plans[0]);p["limits"][key]+=1
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"API budget"):verify_extension(p)
 def test_authentication_proof_change_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["budget_extension"]["preparation_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"evidence changed"):verify_extension(p)
 def test_provider_change_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["paid_configuration"]["model"]="different"
  with self.assertRaisesRegex(ValueError,"configuration changed"):verify_extension(p)
 def test_unapproved_runtime_change_rejected(self):
  p=copy.deepcopy(self.plans[0]);p["source_hashes"]={}
  with self.assertRaisesRegex(ValueError,"Runtime drift"):verify_extension(p)
 def test_subsets_cannot_reset_remaining_time(self):
  p=self.plans[0];v=verify_extension(p);v["limits"]["remaining_seconds"]+=1
  with self.assertRaisesRegex(ValueError,"Remaining clock"):
   check_selection(p["cases"],v,p["budget_extension"]["seconds_already_used"],7200)
 def test_missing_predecessor_report_denies_live_dispatch(self):
  with patch("budget_supplement_queue.Path.exists",return_value=False):
   with self.assertRaisesRegex(ValueError,"still active"):previous_terminal(self.q)
 def test_predecessor_failure_denies_live_dispatch(self):
  previous=json.loads(Path(self.q["predecessor_queue_plan"]).read_text())
  terminal=sealed(dict(plan_binding=previous["binding"],failure="integrity",queue_finished=False,seconds=1))
  with patch("budget_supplement_queue.Path.exists",return_value=True),patch("budget_supplement_queue.strict_file",side_effect=[previous,terminal]):
   with self.assertRaisesRegex(ValueError,"needs reconciliation"):previous_terminal(self.q)
 def test_second_budget_stop_requires_7200(self):
  with self.assertRaisesRegex(ValueError,"Unproven"):
   shard_stop_kind(dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",seconds=3600),dict(binding="x"))
  self.assertEqual(shard_stop_kind(dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",seconds=7200),dict(binding="x")),"audited_budget_stopped")
 def test_environment_failure_is_not_local_timeout(self):
  with self.assertRaisesRegex(ValueError,"stopped or uncertain"):
   shard_stop_kind(dict(plan_binding="x",failure="ValueError: Runtime drift",seconds=7201),dict(binding="x"))
 def test_duplicate_shard_rejected(self):
  q=copy.deepcopy(self.q);q.pop("binding");q["shards"][1]=q["shards"][0]
  with self.assertRaisesRegex(ValueError,"Duplicate authorized"):verify(sealed(q))
 def test_parallelism_and_storage_not_expanded(self):
  for key,value in (("api_workers",11),("native_workers",3),("compile_jobs",3),("link_jobs",2),("max_retained_mib",9000)):
   q=copy.deepcopy(self.q);q.pop("binding");q[key]=value
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"Resource policy"):verify(sealed(q))
if __name__=="__main__":unittest.main()
