import unittest
import tempfile
from pathlib import Path

from pt_agent.experiments import s4_qwen_block as s4


class S4PolicyTests(unittest.TestCase):
    def test_global_deadline_bounds_every_case(self):
        self.assertEqual(s4.remaining_before_deadline(10800, now=s4.DEADLINE - 90), 90)
        with self.assertRaises(TimeoutError):
            s4.remaining_before_deadline(7200, now=s4.DEADLINE)

    def test_slow_mock_selects_smaller_fixed_split_without_repetition(self):
        value = s4.heldout_decision(8.0, 20)
        self.assertEqual(value['requested_heldout_samples'], 256)
        self.assertEqual(value['actual_heldout_samples'], 20)
        self.assertTrue(value['original_split_preserved'])
        self.assertTrue(value['decision_before_Q1'])

    def test_fast_mock_keeps_original_real_samples(self):
        value = s4.heldout_decision(1.0, 1000)
        self.assertEqual(value['actual_heldout_samples'], 1000)
        self.assertFalse(value['split_shortfall'])

    def test_cold_driver_keeps_original_wall_policy_and_block_identity(self):
        from pt_agent.experiments import s4_qwen_cold as cold
        namespace = cold.namespace()
        self.assertEqual(namespace['TOTAL_SECONDS'], 10800)
        self.assertEqual(namespace['SEARCH_CEILING'], 9000)
        self.assertEqual(namespace['TAIL_RESERVE'], 1800)
        self.assertEqual(namespace['serial'].STATE_POLICY['tuner_budget_s'], 2400)
        self.assertEqual(namespace['serial'].STATE_POLICY['split_version'], cold.SPLIT_VERSION)
        self.assertIs(namespace['execute'].__globals__, namespace)
        self.assertIs(namespace['state_summary'].__globals__, namespace)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.json'
            namespace['save'](target, dict(status='running'))
            self.assertEqual(s4.read(target), dict(status='running'))


if __name__ == '__main__':
    unittest.main()
