"""Terminal classification: previous candidate errors never mask service stops."""
import unittest
from audit import classify
class TerminalTests(unittest.TestCase):
    def state(self,layer="numerical_comparison"):
        return dict(phase="complete",status="service_error",http_attempts=2,
            failure=dict(layer=layer,code="numerical_mismatch"),
            ledger=[dict(event="provider_checkpoint",outcomes=[
                dict(status="response_received"),dict(status="failed")])])
    def test_provider_stop_supersedes_old_candidate_failure(self):
        r=classify(self.state(),dict(status="provider_failed"))
        self.assertEqual(r["terminal_layer"],"provider");self.assertTrue(r["historical_label_differs"])
        self.assertEqual(r["historical_failure"]["layer"],"numerical_comparison")
    def test_resolved_exhaustion_is_service_layer(self):
        self.assertEqual(classify(self.state(),dict(status="provider_exhausted"))["terminal_code"],"provider_failure")
    def test_unresolved_transport_is_not_a_closed_audit(self):
        state=self.state();state["ledger"][-1]["outcomes"][-1]["status"]="pending"
        with self.assertRaisesRegex(ValueError,"Resolved"):classify(state,dict(status="provider_failed"))
    def test_missing_transport_outcome_is_rejected(self):
        state=self.state();state["http_attempts"]=3
        with self.assertRaisesRegex(ValueError,"Resolved"):classify(state,dict(status="provider_failed"))
    def test_running_task_is_excluded(self):
        state=self.state();state["phase"]="generating"
        with self.assertRaisesRegex(ValueError,"Closed"):classify(state,dict(status="provider_failed"))
    def test_budget_exhaustion_preserves_candidate_layer(self):
        state=self.state("static_check");state["status"]="budget_exhausted"
        r=classify(state,dict(status="repair_budget_exhausted"))
        self.assertEqual(r["terminal_layer"],"static_check");self.assertFalse(r["historical_label_differs"])
    def test_integrity_and_environment_take_precedence(self):
        for terminal,layer in [("integrity_failed","integrity"),("infrastructure_failed","environment")]:
            self.assertEqual(classify(self.state(),dict(status=terminal))["terminal_layer"],layer)
    def test_never_classifies_compile_failure_as_compiler_bug(self):
        state=self.state("compiler");state["status"]="budget_exhausted"
        r=classify(state,dict(status="repair_budget_exhausted"))
        self.assertEqual(r["terminal_layer"],"compiler")
        self.assertNotIn("compiler_bug",str(r))
if __name__=="__main__":unittest.main(verbosity=2)
