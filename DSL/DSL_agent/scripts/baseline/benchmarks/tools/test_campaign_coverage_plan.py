"""Reject misleading context counts and mappings without API or execution."""
import copy,json,unittest
from pathlib import Path
from audit_campaign_coverage_plan import contexts
from workspace_paths import RESULTS
class ContextPlanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  folder=RESULTS/"stage2-agent-campaign-v4-r118"
  cls.index=json.loads((folder/"index.json").read_text());cls.specs={}
  for ref in cls.index["shards"]:
   for s in json.loads((folder/ref["file"]).read_text())["cases"]:cls.specs[s["id"]]=s
 def test_real_frozen_contexts(self):
  rows,blocked=contexts(self.index,self.specs);self.assertEqual(len(rows),377);self.assertEqual(len(blocked),3)
 def changed(self):
  d=copy.deepcopy(self.index)
  p=next(p for p in d["semantic_partitions"] if p["layer"]=="model")
  return d,p
 def test_three_copies_do_not_make_three_contexts(self):
  d,p=self.changed();p["contexts"]=[p["contexts"][0]]*3
  with self.assertRaisesRegex(ValueError,"distinct topology"):contexts(d,self.specs)
 def test_unknown_task_rejected(self):
  d,p=self.changed();p["contexts"][0]["task_id"]="invented"
  with self.assertRaisesRegex(ValueError,"Unmapped"):contexts(d,self.specs)
 def test_forged_topology_rejected(self):
  d,p=self.changed();p["contexts"][0]["topology"]="0"*64
  with self.assertRaisesRegex(ValueError,"topology mismatch"):contexts(d,self.specs)
 def test_lower_context_requirement_rejected(self):
  d,p=self.changed();p["required_distinct_contexts"]=1
  with self.assertRaisesRegex(ValueError,"requirement"):contexts(d,self.specs)
 def test_missing_blocker_rejected(self):
  d=copy.deepcopy(self.index)
  next(p for p in d["semantic_partitions"] if p["current_status"]=="backend_blocked")["current_status"]="not_evaluated"
  with self.assertRaisesRegex(ValueError,"blocker"):contexts(d,self.specs)
if __name__=="__main__":unittest.main()
