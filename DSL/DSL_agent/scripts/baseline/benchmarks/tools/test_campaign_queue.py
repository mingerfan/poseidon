"""Queue budget/tamper gates against real frozen plans. No provider or child launch."""
import copy,json,unittest
from campaign_queue import verify_queue
from campaign_live_state import sealed
from workspace_paths import RESULTS
PLAN=RESULTS/"stage2-paid-queue-r113-verified/plan.json"
class QueueGates(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.plan=json.loads(PLAN.read_text())
 def changed(self,key,value):
  p=copy.deepcopy(self.plan);p.pop("binding");p[key]=value;return sealed(p)
 def test_frozen_queue(self):
  verify_queue(self.plan);self.assertEqual(self.plan["planned_tasks"],1982)
 def test_expanded_generation_budget_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify_queue(self.changed("maximum_generations",7929))
 def test_expanded_time_budget_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify_queue(self.changed("max_wall_seconds",167581))
 def test_parallelism_rejected(self):
  with self.assertRaisesRegex(ValueError,"resource policy"):verify_queue(self.changed("concurrency",2))
 def test_disk_reserve_cannot_be_removed(self):
  with self.assertRaisesRegex(ValueError,"resource policy"):verify_queue(self.changed("min_free_mib",0))
 def test_duplicate_shard_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[1]=refs[0]
  with self.assertRaisesRegex(ValueError,"Duplicate"):verify_queue(self.changed("shards",refs))
 def test_output_escape_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["output"]="/tmp/foreign"
  with self.assertRaisesRegex(ValueError,"output boundary"):verify_queue(self.changed("shards",refs))
 def test_source_drift_rejected(self):
  with self.assertRaisesRegex(ValueError,"runtime drift"):verify_queue(self.changed("source_hashes",{}))
 def test_forged_shard_hash_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"plan drift"):verify_queue(self.changed("shards",refs))
if __name__=="__main__":unittest.main()
