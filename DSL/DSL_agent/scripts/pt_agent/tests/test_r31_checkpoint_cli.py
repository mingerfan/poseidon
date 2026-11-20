"""Capability checks for explicit sparse native observations."""
import unittest
from pt_agent.experiments.r31_lattigo_runtime_plan import checkpoint_arguments


class CheckpointArgumentsTests(unittest.TestCase):
    def test_original_prepared_clients_keep_original_command(self):
        self.assertEqual(checkpoint_arguments({}), [])

    def test_old_protocol_number_alone_does_not_enable_new_argument(self):
        with self.assertRaises(ValueError):
            checkpoint_arguments({'client_cache_protocol': 9}, 1000)

    def test_actual_completed_selftest_enables_sparse_snapshot(self):
        prepared = {'cache_selftest': {'checkpoint_policy': {
            'interval': 1, 'unit': 'existing_save_request',
            'forced_requests': 1, 'writes_completed': 1, 'write_errors': 0}}}
        self.assertEqual(checkpoint_arguments(prepared, 1000),
                         ['-checkpoint-interval', '1000'])
        prepared['cache_selftest']['checkpoint_policy']['write_errors'] = 1
        with self.assertRaises(ValueError):
            checkpoint_arguments(prepared, 1000)

    def test_invalid_intervals_are_rejected_before_use(self):
        for interval in (True, 0, -1, 1000001, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                checkpoint_arguments({}, interval)


if __name__ == '__main__':
    unittest.main()
