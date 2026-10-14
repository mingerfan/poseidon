"""Offline checks of stopped-queue continuation and isolated time budgets."""
import copy,json,unittest
from stage2_agent_campaign import ROOT
from workspace_paths import RESULTS
from campaign_live_state import sealed
from parallel_campaign_queue_v2 import verify,shard_stop_kind
class ContinuationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((RESULTS/"stage2-parallel-queue-r125/plan.json").read_text())
 def changed(self,key,value):
  p=copy.deepcopy(self.plan);p.pop("binding");p[key]=value;return sealed(p)
 def test_exact_unstarted_set(self):
  verify(self.plan)
  self.assertEqual([int(r["output"].rsplit("-",1)[-1]) for r in self.plan["shards"]],list(range(17,43)))
  self.assertEqual(self.plan["planned_tasks"],1214)
 def test_all_frozen_requests_and_budgets_unchanged(self):
  count=0
  for p in (RESULTS/"stage2-agent-campaign-v6-r125").glob("shard-*.json"):
   a=json.loads(p.read_text());b=json.loads((RESULTS/"stage2-agent-campaign-v5-r121"/p.name).read_text())
   for key in ("cases","limits","source_hashes","paid_configuration"):self.assertEqual(a[key],b[key])
   count+=len(a["cases"])
  self.assertEqual(count,2030)
 def test_prior_time_cannot_reset(self):
  c=copy.deepcopy(self.plan["continuation"]);c["seconds_used"]=0
  with self.assertRaisesRegex(ValueError,"budget reset"):verify(self.changed("continuation",c))
 def test_prior_remaining_cannot_inflate(self):
  c=copy.deepcopy(self.plan["continuation"]);c["seconds_remaining"]+=1
  with self.assertRaisesRegex(ValueError,"budget reset"):verify(self.changed("continuation",c))
 def test_previously_started_shard_rejected(self):
  old=json.loads((RESULTS/"stage2-parallel-queue-r121/plan.json").read_text())
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["previous_binding"]=old["shards"][0]["binding"]
  with self.assertRaisesRegex(ValueError,"Previously started"):verify(self.changed("shards",refs))
 def test_dispatched_without_batch_directory_still_rejected(self):
  old=json.loads((RESULTS/"stage2-parallel-queue-r121/plan.json").read_text())
  ref=next(x for x in old["shards"] if x["output"].endswith("-016"))
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["previous_binding"]=ref["binding"]
  with self.assertRaisesRegex(ValueError,"Previously started"):verify(self.changed("shards",refs))
 def test_duplicate_predecessor_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[1]["previous_binding"]=refs[0]["previous_binding"]
  with self.assertRaisesRegex(ValueError,"Repeated predecessor"):verify(self.changed("shards",refs))
 def test_resource_caps(self):
  for key,value in (("api_workers",11),("native_workers",3),("compile_jobs",4),("link_jobs",2),("max_retained_mib",16384),("min_free_mib",0)):
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,"resource policy"):verify(self.changed(key,value))
 def test_total_budget_tampering(self):
  for key in ("maximum_generations","maximum_http_attempts","max_wall_seconds"):
   with self.subTest(key=key),self.assertRaises(ValueError):verify(self.changed(key,self.plan[key]+1))
 def test_ordinary_terminal(self):
  self.assertEqual(shard_stop_kind(dict(plan_binding="x",failure=None,not_run_or_interrupted=[],seconds=30),dict(binding="x",max_wall_seconds=3600)),"audited_terminal")
 def test_shard_time_limit_is_local_and_auditable(self):
  d=dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",not_run_or_interrupted=["remaining"],seconds=3600.5)
  self.assertEqual(shard_stop_kind(d,dict(binding="x",max_wall_seconds=3600)),"audited_budget_stopped")
 def test_false_timeout_rejected(self):
  d=dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",not_run_or_interrupted=["remaining"],seconds=1)
  with self.assertRaisesRegex(ValueError,"Unproven"):shard_stop_kind(d,dict(binding="x",max_wall_seconds=3600))
 def test_integrity_environment_unknown_interruptions_still_stop(self):
  for failure in ("ValueError: Runtime drift","ValueError: Environment/integrity failure; stop related batch","KeyboardInterrupt: ","ValueError: Retained artifact budget exhausted"):
   with self.subTest(failure=failure),self.assertRaisesRegex(ValueError,"stopped or uncertain"):
    shard_stop_kind(dict(plan_binding="x",failure=failure,not_run_or_interrupted=["x"],seconds=3601),dict(binding="x",max_wall_seconds=3600))
 def test_unexplained_incomplete_terminal_rejected(self):
  with self.assertRaisesRegex(ValueError,"stopped or uncertain"):
   shard_stop_kind(dict(plan_binding="x",failure=None,not_run_or_interrupted=["x"],seconds=30),dict(binding="x",max_wall_seconds=3600))
 def test_wrong_binding_rejected(self):
  with self.assertRaisesRegex(ValueError,"identity"):
   shard_stop_kind(dict(plan_binding="wrong"),dict(binding="x"))
if __name__=="__main__":unittest.main()
