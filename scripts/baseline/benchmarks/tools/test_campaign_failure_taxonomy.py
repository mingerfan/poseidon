"""Provider failure is separate from prior numerical attempts and billing."""
import copy,json,unittest
from campaign_failure_taxonomy import provider_failure
from stage2_agent_campaign import ROOT
from workspace_paths import RESULTS
class FailureClassification(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.real=json.loads((RESULTS/"agent-deepseek-e2tyuyra/report.json").read_text())
 def test_real_tls_has_no_dsl_attempt(self):
  r=provider_failure(self.real)
  self.assertEqual(r["category"],"tls_failure_unresolved")
  self.assertEqual(r["attempts_evaluated_before_provider_failure"],0)
  self.assertEqual(r["billing_status"],"not_determined")
  self.assertFalse(r["automatic_retry_authorized_by_classification"])
 def test_provider_failure_after_numerical_failure_is_not_lost(self):
  r=copy.deepcopy(self.real);r["attempts"]=[dict(compiled=True,executed=True,failure_layer="numerical_comparison")]
  result=provider_failure(r)
  self.assertEqual(result["layer"],"provider");self.assertTrue(result["encrypted_before_provider_failure"])
 def test_unfinished_call_is_not_terminal(self):
  r=copy.deepcopy(self.real);r["provider_metrics"]["calls"][-1]["status"]="in_flight"
  with self.assertRaises(ValueError):provider_failure(r)
 def test_success_is_not_provider_failure(self):
  r=copy.deepcopy(self.real);r["status"]="passed"
  with self.assertRaises(ValueError):provider_failure(r)
if __name__=="__main__":unittest.main()
