"""Budget/claim/admission regressions; no model API or confidential execution."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from remaining_campaign_policy import check_clock,unclaimed
from remaining_campaign_queue import guarded_submit,shard_stop_kind

class BudgetTests(unittest.TestCase):
 def test_original_clock_preserved(self):
  check_clock(dict(seconds_already_used=2900,cumulative_limit_seconds=3600,remaining_seconds=700),2900,3600)
 def test_clock_reset_rejected(self):
  with self.assertRaises(ValueError):check_clock(dict(seconds_already_used=0,cumulative_limit_seconds=3600,remaining_seconds=3600),2900,3600)
 def test_unapproved_time_expansion_rejected(self):
  with self.assertRaises(ValueError):check_clock(dict(seconds_already_used=2900,cumulative_limit_seconds=7200,remaining_seconds=4300),2900,3600)
 def test_exhausted_clock_rejected(self):
  with self.assertRaises(ValueError):check_clock(dict(seconds_already_used=3601,cumulative_limit_seconds=3600,remaining_seconds=-1),3601,3600)
 def test_exact_authorized_extension(self):
  check_clock(dict(seconds_already_used=3601,cumulative_limit_seconds=7200,remaining_seconds=3599),3601,7200)
 def test_space_failure_before_submission(self):
  pool=Mock();worker=Mock()
  def boundary():raise ValueError("Aggregate artifact boundary")
  with self.assertRaises(ValueError):guarded_submit(pool,worker,[1,2],boundary)
  pool.submit.assert_not_called();worker.assert_not_called()
 def test_admission_order(self):
  seen=[];pool=Mock()
  pool.submit.side_effect=lambda w,r:(seen.append(("submit",r)),r)[1]
  guarded_submit(pool,Mock(),[1,2],lambda:seen.append("boundary"))
  self.assertEqual(seen,["boundary",("submit",1),("submit",2)])
 def test_claim_and_broken_symlink_rejected(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
   r=Path(tmp);c=r/"stage2-agent-evaluation-claims";c.mkdir()
   with patch("remaining_campaign_policy.RESULTS",r):
    unclaimed([dict(evaluation_identity="id",id="task")])
    p=c/"id.json";p.write_text("{}")
    with self.assertRaises(ValueError):unclaimed([dict(evaluation_identity="id",id="task")])
    p.unlink();p.symlink_to(c/"missing")
    with self.assertRaises(ValueError):unclaimed([dict(evaluation_identity="id",id="task")])
 def test_normal_cumulative_stop_local(self):
  self.assertEqual(shard_stop_kind(dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",seconds=3600),dict(binding="x",max_wall_seconds=3600)),"audited_budget_stopped")
 def test_premature_timeout_rejected(self):
  with self.assertRaises(ValueError):shard_stop_kind(dict(plan_binding="x",failure="TimeoutError: Cumulative wall budget exhausted",seconds=3600),dict(binding="x",max_wall_seconds=7200))
 def test_environment_failure_still_stops(self):
  with self.assertRaises(ValueError):shard_stop_kind(dict(plan_binding="x",failure="ValueError: Runtime drift",seconds=20),dict(binding="x",max_wall_seconds=3600))
if __name__=="__main__":unittest.main(verbosity=2)
