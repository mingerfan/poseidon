import unittest
from pt_agent.experiments.r31_survivor_observer import GIB,guard_reason,identity_matches,terminal_status
class SurvivorObserverTests(unittest.TestCase):
 def test_pid_reuse(self):
  self.assertFalse(identity_matches(dict(pid=3,starttime_ticks=7),3,8));self.assertTrue(identity_matches(dict(pid=3,starttime_ticks=8),3,8))
 def test_partial_exit(self):self.assertEqual(terminal_status(dict(status='running'),False),'native_exited_without_terminal_client')
 def test_early_complete_snapshot(self):self.assertEqual(terminal_status(dict(status='ok',encrypted_plan_completed=True),True),'observing')
 def test_success_and_numeric_failure(self):
  self.assertEqual(terminal_status(dict(status='ok',encrypted_plan_completed=True),False),'native_completed_observed');self.assertEqual(terminal_status(dict(status='error'),False),'native_error_observed')
 def test_original_deadline(self):self.assertEqual(guard_reason(dict(vmrss_kib=1),3*GIB,20*GIB,100,10800),'original_native_wall_above10800seconds')
 def test_memory_and_space(self):
  self.assertEqual(guard_reason(dict(vmrss_kib=15*GIB//1024),3*GIB,20*GIB,100,20),'native_RSS_above14GiB');self.assertEqual(guard_reason(dict(vmrss_kib=1),GIB,20*GIB,100,20),'available_RAM_below1.25GiB');self.assertIsNone(guard_reason(dict(vmrss_kib=11*GIB//1024),3*GIB,20*GIB,100,20))
if __name__=='__main__':unittest.main()
