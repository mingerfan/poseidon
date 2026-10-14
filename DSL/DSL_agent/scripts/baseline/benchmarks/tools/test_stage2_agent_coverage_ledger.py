"""Accounting tests: keep manual/current/historical and blocked scopes distinct."""
import copy,json,unittest
from pathlib import Path
from stage2_agent_coverage_ledger import attach_history,task_counts,build,ROOT,digest
class LedgerTests(unittest.TestCase):
 def fixtures(self):
  return (dict(id="a",model_sha256="m",history=[],historical_agent_status="not_run",current_source_agent_status="not_run"),
          dict(id="a",model_sha256="m",request_id="r"),
          dict(id="a",model_sha256="m",request_id="r",status="passed",evidence="retained",
               first_attempt_passed=True,generations=1,http_attempts=1))
 def test_old_pass_is_not_current_pass(self):
  t,s,a=self.fixtures();attach_history(t,s,a,"old-source","old-plan")
  self.assertEqual(t["historical_agent_status"],"passed")
  self.assertEqual(t["current_source_agent_status"],"not_run")
  self.assertEqual(t["history"][0]["source_digest"],"old-source")
 def test_changed_model_rejected(self):
  t,s,a=self.fixtures();a["model_sha256"]="different"
  with self.assertRaisesRegex(ValueError,"Model identity"):attach_history(t,s,a,"old","plan")
 def test_changed_request_rejected(self):
  t,s,a=self.fixtures();a["request_id"]="different"
  with self.assertRaisesRegex(ValueError,"Audit identity"):attach_history(t,s,a,"old","plan")
 def test_duplicate_history_rejected(self):
  t,s,a=self.fixtures();attach_history(t,s,a,"old","plan")
  with self.assertRaisesRegex(ValueError,"Duplicate"):attach_history(t,s,a,"old","plan")
 def test_skip_or_prepared_is_not_pass(self):
  for state in ("skipped","prepared_not_generated","manual_pass","running"):
   t,s,a=self.fixtures();a["status"]=state
   with self.subTest(state=state),self.assertRaises(ValueError):attach_history(t,s,a,"old","plan")
 def test_task_id_mismatch_rejected(self):
  t,s,a=self.fixtures();a["id"]="b"
  with self.assertRaisesRegex(ValueError,"Task identity"):attach_history(t,s,a,"old","plan")
 def test_group_counts_preserve_failures_and_unrun(self):
  tasks=[dict(group="free",historical_agent_status=s,current_source_agent_status="not_run") for s in ("passed","failed","not_run")]
  self.assertEqual(task_counts(tasks)["free"],dict(total=3,historical=dict(passed=1,failed=1,not_run=1),current_source=dict(not_run=3)))
 def test_complete_join_and_reproducible_binding(self):
  report=build()
  disk=json.loads((ROOT/"docs/baseline/stage2-agent-coverage-ledger-r101.json").read_text())
  self.assertEqual(report,disk)
  self.assertEqual(report["task_inventory_count"],2030)
  self.assertEqual(len({t["id"] for t in report["tasks"]}),2030)
  self.assertEqual(len(report["partitions"]),401)
  self.assertEqual(sum(t["current_request_prepared"] for t in report["tasks"]),6)
  self.assertEqual(sum(t["current_source_agent_status"]!="not_run" for t in report["tasks"]),0)
  self.assertEqual(report["partition_states"]["backend_blocked"],3)
  self.assertEqual(report["primary_unique_models"],1200)
  self.assertFalse(report["stage2_complete"])
if __name__=="__main__":unittest.main()
