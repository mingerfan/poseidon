"""Accounting and semantic-denominator tests; no API or FHE evidence fabricated."""
import copy,unittest
from summarize_agent_campaign import aggregate,merge_audited,partition_status
class SummaryTests(unittest.TestCase):
 def row(self):
  return dict(id="a",status="passed",report_sha256="h",evidence="/result",request_id="r",first_attempt_passed=True,
              generations=1,http_attempts=1,attempts=[dict(compiled=True,executed=True)],comparison={"max_absolute_error":0})
 def test_pending_failed_and_blocked_not_passes(self):
  r=aggregate([dict(status=s) for s in ("passed","failed","terminal_pending_audit","running_verified","not_run","request_preparation_blocked")])
  self.assertEqual(r["audited"],2);self.assertEqual(r["audited_task_success_rate"],.5)
  self.assertEqual(r["all_planned_task_success_rate"],1/6);self.assertFalse(r["all_tasks_audited"])
 def test_no_evaluation_has_no_audited_rate(self):
  self.assertIsNone(aggregate([dict(status="not_run")])["audited_task_success_rate"])
  self.assertFalse(aggregate([])["all_tasks_audited"])
 def test_same_result_multiple_audits_does_not_duplicate_case(self):
  states={"a":dict(id="a",status="not_run")};row=self.row()
  merge_audited(states,row,"batch","audit1");merge_audited(states,row,"batch","audit2")
  self.assertEqual(aggregate(list(states.values()))["audited"],1)
  self.assertEqual(states["a"]["supporting_audits"],["audit1","audit2"])
 def test_conflicting_retests_cannot_select_best_result(self):
  states={"a":dict(id="a",status="not_run")};row=self.row()
  merge_audited(states,row,"batch","audit1")
  with self.assertRaisesRegex(ValueError,"Conflicting"):merge_audited(states,row,"another-batch","audit2")
 def test_three_designated_contexts_require_all_audited_passes(self):
  p=dict(id="x",layer="construction",current_status="not_evaluated",contexts=[dict(task_id=x) for x in ("a","b","c")],required_distinct_contexts=3)
  states={x:dict(status="passed") for x in ("a","b")};states["c"]=dict(status="terminal_pending_audit")
  self.assertEqual(partition_status(p,states)["agent_status"],"incomplete_agent_contexts")
  states["c"]["status"]="passed";self.assertEqual(partition_status(p,states)["agent_status"],"designated_contexts_passed")
 def test_backend_static_and_compiler_are_separate(self):
  for layer,status,expected in (("helper","backend_blocked","backend_blocked"),("rejection","not_evaluated","static_controls_separate"),("compiler","not_evaluated","current_agent_artifact_audit_required")):
   p=dict(layer=layer,current_status=status)
   self.assertEqual(partition_status(p,{})["agent_status"],expected)
if __name__=="__main__":unittest.main()
