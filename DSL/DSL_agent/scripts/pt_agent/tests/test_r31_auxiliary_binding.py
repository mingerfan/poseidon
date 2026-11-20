"""Explicit native auxiliary-prime grouping; no crypto or model fixtures."""
import unittest
from pt_agent.experiments.r31_lattigo_binding import NativeBinding, map_precision_plan
from pt_agent.experiments.r31_lattigo_runtime_plan import auxiliary_arguments, check_bound_target


class AuxiliaryBindingTests(unittest.TestCase):
    def test_default_preserves_recorded_original_native_group(self):
        value = NativeBinding(50, mod1_inv_degree=15, log_message_ratio=2,
                              bootstrap_normalization_log2=3)
        self.assertEqual(value.metadata()['log_p'], [61] * 3)
        self.assertEqual(value.metadata()['bootstrap_log_p'], [61] * 4)
        self.assertEqual(value.variant(32),
            'lattigo-n16-q55-50x8-p61x3-scale50-logicalslots14-bootcache32-residualcache1-plaintext1-mod1inv15-mr2-inputdiv8')

    def test_new_group_explicitly_changes_both_auxiliary_lists_only(self):
        original = NativeBinding(50, mod1_inv_degree=15, log_message_ratio=2,
                                 bootstrap_normalization_log2=3)
        new = NativeBinding(50, mod1_inv_degree=15, log_message_ratio=2,
                            bootstrap_normalization_log2=3, auxiliary_prime_bits=57)
        before, after = original.metadata(), new.metadata()
        self.assertEqual({k for k in before if before[k] != after[k]},
                         {'log_p', 'bootstrap_log_p'})
        self.assertEqual(after['log_p'], [57] * 3)
        self.assertEqual(after['bootstrap_log_p'], [57] * 4)
        self.assertIn('-p57x3-', new.variant(32))
        self.assertTrue(new.variant(32).endswith('-bootp57x4'))
        self.assertIsNone(after['security_acceptance'])

    def test_precision_mapping_preserves_arithmetic_and_original_plan(self):
        source = {'target': {'target_id': 'original'},
                  'values': [{'id': 'x', 'scale_log2': 43}, {'id': 'y', 'scale_log2': 86}],
                  'initialization': [], 'execution': [{'op': 'mul_cc', 'inputs': ['x', 'x'],
                                                       'output': 'y'}], 'finalization': [],
                  'external_inputs': ['x'], 'final_outputs': ['y']}
        mapped, report = map_precision_plan(source, NativeBinding(50, auxiliary_prime_bits=57))
        self.assertEqual(mapped['execution'], source['execution'])
        self.assertEqual(source['values'][0]['scale_log2'], 43)
        self.assertEqual(mapped['values'][1]['scale_log2'], 100)
        self.assertFalse(report['raw_data_modified'])
        self.assertEqual(report['native_binding']['bootstrap_log_p'], [57] * 4)

    def test_other_auxiliary_values_rejected(self):
        for value in (True, 56, 58, 61.0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NativeBinding(auxiliary_prime_bits=value)

    def test_old_binary_only_supports_legacy_auxiliary_command(self):
        self.assertEqual(auxiliary_arguments({}, 61), [])
        with self.assertRaises(ValueError):
            auxiliary_arguments({'client_cache_protocol': 9}, 57)
        self.assertEqual(auxiliary_arguments({'supported_auxiliary_prime_bits': [57,61]},57),
                         ['-auxiliary-prime-bits','57'])

    def test_mapped_plan_group_must_match_selected_client_binding(self):
        binding = NativeBinding(50, auxiliary_prime_bits=57)
        plan = {'target': {'target_id': 'pt-agent-lattigo-native-adapter',
                          'adapter_variant': binding.variant(32), 'precision_bits': 50}}
        check_bound_target(plan, binding, 32)
        for wrong, capacity in ((NativeBinding(50),32), (binding,1)):
            with self.subTest(wrong=wrong,capacity=capacity), self.assertRaises(ValueError):
                check_bound_target(plan,wrong,capacity)


if __name__ == '__main__':
    unittest.main()
