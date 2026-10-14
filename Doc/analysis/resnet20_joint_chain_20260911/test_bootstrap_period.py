"""Regression counterexamples: a passing scale ledger is not a bootstrap proof."""
import json
import unittest
from pathlib import Path
from audit_bootstrap_period import audit
from verify_joint import verify


class BootstrapPeriodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((Path(__file__).parent/'q50.json').read_text())
        cls.result=audit(cls.report)

    def test_saved_scale_ledger_still_passes_but_period_does_not(self):
        self.assertEqual(verify(self.report)['status'],'PASS_SCALE_LEDGER')
        self.assertFalse(self.result['period_alignment_pass'])
        self.assertGreater(abs(self.result['period_error_fraction']),0.01)

    def test_zero_message_nonzero_wrap_is_counterexample(self):
        case=next(c for c in self.result['cases'] if c['input']==0 and c['integer_wrap']==1)
        self.assertGreater(case['absolute_error'],0.1)

    def test_integer_free_case_hides_period_error(self):
        case=next(c for c in self.result['cases'] if c['input']==0.05 and c['integer_wrap']==0)
        self.assertLess(case['absolute_error'],1e-4)

    def test_period_aligned_counterfactual_is_not_amplitude_normalized(self):
        c=self.result['diagnostic_counterfactual']
        self.assertAlmostEqual(c['periods_per_integer_wrap'],1,places=12)
        self.assertGreater(abs(c['remaining_small_signal_message_gain']-1),0.01)
        self.assertFalse(c['precision_validated'])


if __name__=='__main__':unittest.main()
