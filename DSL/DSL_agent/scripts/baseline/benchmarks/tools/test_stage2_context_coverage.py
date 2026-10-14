import copy,unittest
from audit_stage2_context_coverage import assess

class DistinctCoverageTests(unittest.TestCase):
 def fixture(self):
  p=dict(id="sem",layer="model",scope="Mathematical semantics",agent_status="incomplete_agent_contexts",
   required_distinct_contexts=3,contexts=[dict(task_id=t,topology=g) for t,g in zip("abcd","AABC")])
  states={t:dict(status="passed",audited_result=dict(status="passed")) for t in "abcd"}
  return p,states,dict(zip("abcd","AABC"))
 def test_duplicate_topology_does_not_supply_third_context(self):
  p,s,t=self.fixture();s["d"]["status"]="failed"
  r=assess(p,s,t);self.assertEqual(r["passed_distinct_topologies"],2);self.assertFalse(r["minimum_three_topologies_verified"])
 def test_minimum_and_all_are_independent(self):
  p,s,t=self.fixture();s["b"]["status"]="failed"
  r=assess(p,s,t);self.assertTrue(r["minimum_three_topologies_verified"]);self.assertFalse(r["all_designated_tasks_passed"])
  self.assertEqual(r["unmet_tasks"][0]["status"],"failed")
 def test_pending_and_skipped_are_not_passes(self):
  for status in ["terminal_pending_audit","running_verified","not_run","launched_unconfirmed","skipped"]:
   with self.subTest(status=status):
    p,s,t=self.fixture();s["d"]["status"]=status
    self.assertFalse(assess(p,s,t)["minimum_three_topologies_verified"])
 def test_unbacked_pass_rejected(self):
  p,s,t=self.fixture();s["d"].pop("audited_result")
  with self.assertRaises(ValueError):assess(p,s,t)
 def test_changed_topology_rejected(self):
  p,s,t=self.fixture();t["a"]="changed"
  with self.assertRaises(ValueError):assess(p,s,t)
 def test_missing_task_rejected(self):
  p,s,t=self.fixture();s.pop("b")
  with self.assertRaises(ValueError):assess(p,s,t)
 def test_insufficient_planned_topologies_rejected(self):
  p,s,t=self.fixture();p["contexts"][-1]["topology"]="B";t["d"]="B"
  with self.assertRaises(ValueError):assess(p,s,t)
 def test_backend_blocked_never_passes(self):
  p,s,t=self.fixture();p["agent_status"]="backend_blocked"
  r=assess(p,s,t);self.assertEqual(r["status"],"backend_blocked");self.assertFalse(r["executable_context_denominator"])
if __name__=="__main__":unittest.main(verbosity=2)
