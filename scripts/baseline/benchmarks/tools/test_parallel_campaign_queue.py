"""Offline concurrency/budget tampering and request-preservation checks."""
import copy,json,unittest
from pathlib import Path
from stage2_agent_campaign import ROOT
from workspace_paths import RESULTS
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
from parallel_campaign_queue import verify
class ParallelPlanTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.folder=RESULTS/"stage2-agent-campaign-v5-r121"
  idx=json.loads((cls.folder/"index.json").read_text());refs=[];sources=None
  for i,ref in enumerate(idx["shards"]):
   p=cls.folder/ref["file"];s=json.loads(p.read_text());sources=s["source_hashes"]
   if i<2:continue
   refs.append(dict(plan=str(p),sha256=sha(p),binding=s["binding"],tasks=len(s["cases"]),
    output=str(RESULTS/("parallel-test-unused-%03d"%i)),audit=str(RESULTS/("parallel-test-unused-%03d-audit"%i)),
    max_wall_seconds=s["limits"]["max_wall_seconds"]))
  cls.plan=sealed(dict(source_hashes=sources,runner_sha256=sha(Path(__file__).with_name("parallel_campaign_queue.py")),
   source_index=str(cls.folder/"index.json"),source_index_sha256=sha(cls.folder/"index.json"),
   shards=refs,api_workers=10,native_workers=2,compile_jobs=2,link_jobs=1,max_retained_mib=8192,min_free_mib=4096,
   admission_available_mib=3072,stop_available_mib=512,handoff_parents={},inherited_artifact_roots=[],
   planned_tasks=sum(r["tasks"] for r in refs),maximum_generations=4*sum(r["tasks"] for r in refs),
   maximum_http_attempts=16*sum(r["tasks"] for r in refs),max_wall_seconds=sum(r["max_wall_seconds"]+390 for r in refs)))
 def changed(self,key,value):
  p=copy.deepcopy(self.plan);p.pop("binding");p[key]=value;return sealed(p)
 def test_resource_only_fixture(self):
  verify(self.plan);self.assertEqual(len(self.plan["shards"]),41)
 def test_all_frozen_requests_unchanged(self):
  count=0
  for p in sorted(self.folder.glob("shard-*.json")):
   old=json.loads((RESULTS/"stage2-agent-campaign-v4-r118"/p.name).read_text())
   new=json.loads(p.read_text())
   self.assertEqual(old["cases"],new["cases"]);self.assertEqual(old["limits"],new["limits"])
   self.assertEqual(old["source_hashes"],new["source_hashes"]);count+=len(new["cases"])
  self.assertEqual(count,2030)
 def test_unbounded_worker_count_rejected(self):
  with self.assertRaisesRegex(ValueError,"resource policy"):verify(self.changed("api_workers",41))
 def test_excess_native_workers_rejected(self):
  with self.assertRaisesRegex(ValueError,"resource policy"):verify(self.changed("native_workers",3))
 def test_memory_reserve_removal_rejected(self):
  with self.assertRaisesRegex(ValueError,"resource policy"):verify(self.changed("stop_available_mib",0))
 def test_generation_budget_increase_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify(self.changed("maximum_generations",self.plan["maximum_generations"]+1))
 def test_http_budget_increase_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify(self.changed("maximum_http_attempts",self.plan["maximum_http_attempts"]+1))
 def test_wall_budget_reset_rejected(self):
  with self.assertRaisesRegex(ValueError,"budget drift"):verify(self.changed("max_wall_seconds",self.plan["max_wall_seconds"]+3600))
 def test_duplicate_shard_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[1]=refs[0]
  with self.assertRaisesRegex(ValueError,"Duplicate"):verify(self.changed("shards",refs))
 def test_foreign_output_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["output"]="/tmp/foreign"
  with self.assertRaisesRegex(ValueError,"Output boundary"):verify(self.changed("shards",refs))
 def test_per_shard_time_change_rejected(self):
  refs=copy.deepcopy(self.plan["shards"]);refs[0]["max_wall_seconds"]+=1
  with self.assertRaisesRegex(ValueError,"time drift"):verify(self.changed("shards",refs))
if __name__=="__main__":unittest.main()
