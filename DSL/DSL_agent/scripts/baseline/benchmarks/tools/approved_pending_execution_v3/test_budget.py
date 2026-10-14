"""Offline call and deadline controls. Never touches provider credentials."""
import unittest
from budget import remaining,limits,exact_certificate_stop
class Accounting(unittest.TestCase):
 def test_used_attempt_subtracted(self):self.assertEqual(remaining(1,1),{"generations":3,"http_attempts":15})
 def test_fresh_case(self):self.assertEqual(remaining(0,0),{"generations":4,"http_attempts":16})
 def test_no_extra_after_exhaustion(self):self.assertIsNone(limits(100,1,4,16))
 def test_expired_not_restarted(self):self.assertIsNone(limits(100,100))
 def test_original_deadline_caps_task(self):self.assertEqual(limits(1000,900,1,1)["wall_seconds"],100)
 def test_case_time_cap(self):self.assertEqual(limits(100000,1)["wall_seconds"],14400)
 def test_bad_accounting_rejected(self):
  for g,h in [(-1,0),(5,1),(0,17),(True,1),(1,False)]:
   with self.assertRaises(ValueError):remaining(g,h)
 def test_unknown_provider_outcome_not_retried(self):
  s=dict(status="service_error",generations=1,http_attempts=1,package=None,ledger=[dict(event="http_reserved",outcome="unknown_until_checkpoint")])
  self.assertFalse(exact_certificate_stop(s))
 def test_only_explicit_certificate_failure(self):
  s=dict(status="service_error",generations=1,http_attempts=1,package=None,ledger=[dict(event="provider_checkpoint",outcomes=[dict(index=0,status="failed",error="transport_certificate_failed")])])
  self.assertTrue(exact_certificate_stop(s));s["ledger"][-1]["outcomes"][0]["error"]="transport_timeout";self.assertFalse(exact_certificate_stop(s))
if __name__=="__main__":unittest.main(verbosity=2)
