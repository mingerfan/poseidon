"""Queue budget/tamper gates against real frozen plans. No provider or child launch."""
import copy,json,unittest
from campaign_queue_v2 import verify_queue
from campaign_live_state import sealed
from workspace_paths import RESULTS
PLAN=RESULTS/"stage2-paid-queue-r118/plan.json"
class QueueGates(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.plan=json.loads(PLAN.read_text())
 def changed(self,key,value):
  p=copy.deepcopy(self.plan);p.pop("binding");p[key]=value;return sealed(p)
 def test_frozen_queue(self):
  verify_queue(self.plan);self.assertEqual(self.plan["planned_tasks"],2030)
 def test_expanded_generation_budget_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify_queue(self.changed("maximum_generations",8121))
 def test_expanded_time_budget_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify_queue(self.changed("max_wall_seconds",171571))
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

 def test_pilot_requests_and_claims_are_preserved(self):
  from stage2_agent_pilot_plan import check_binding
  pilot=json.loads((RESULTS/'stage2-math-pilot-prepare-r117-ready/plan.json').read_text())
  specs={}
  for ref in self.plan['shards']:
   child=json.loads(__import__('pathlib').Path(ref['plan']).read_text())
   self.assertEqual(child['generation_guidance'],'explicit-v3')
   for spec in child['cases']:specs[spec['id']]=spec
  self.assertEqual(len(specs),2030)
  self.assertEqual(len(pilot['cases']),6)
  for case in pilot['cases']:
   spec=specs[case['id']]
   self.assertEqual(spec['request'],case['request'])
   self.assertEqual(spec['evaluation_identity'],case['evaluation_identity'])
   claim=json.loads((RESULTS/'stage2-agent-evaluation-claims'/(spec['evaluation_identity']+'.json')).read_text())
   check_binding(claim)
   self.assertEqual(claim['owner'],str(RESULTS/'stage2-math-pilot-r117-live'))
   self.assertEqual(claim['task_id'],spec['id'])

if __name__=="__main__":unittest.main()
