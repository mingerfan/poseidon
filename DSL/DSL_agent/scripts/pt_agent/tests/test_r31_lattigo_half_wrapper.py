"""New dyadic bootstrap wrapper semantics, without native key generation."""
import copy
from fractions import Fraction
import unittest

from pt_agent.experiments.r31_lattigo_binding import NativeBinding, map_precision_plan
from pt_agent.experiments.r31_lattigo_input_worker import make_bootstrap_plan


class HalfWrapperChecks(unittest.TestCase):
    def test_explicit_policy_retains_original_plan_and_reads_source_after_boot(self):
        base = NativeBinding(50, mod1_inv_degree=7, log_message_ratio=2)
        half = NativeBinding(50, mod1_inv_degree=7, log_message_ratio=2,
                             bootstrap_input_divisor=2)
        self.assertNotIn('inputdiv2', base.variant(32))
        self.assertEqual(half.variant(32), base.variant(32) + '-inputdiv2')
        self.assertNotIn('bootstrap_input_value_divisor', NativeBinding().metadata())
        for kwargs in ({'bootstrap_input_divisor': 2},
                       {'precision_bits': 50, 'mod1_inv_degree': 7,
                        'bootstrap_input_divisor': 2},
                       {'precision_bits': 50, 'mod1_inv_degree': 7,
                        'log_message_ratio': 2, 'bootstrap_input_divisor': True}):
            with self.assertRaises(ValueError):
                NativeBinding(**kwargs)
        source = make_bootstrap_plan({'target_id': 'original'}, branch=True)
        before = copy.deepcopy(source)
        mapped, _ = map_precision_plan(source, half, bootstrap_cache=32)
        unchanged, _ = map_precision_plan(source, base, bootstrap_cache=32)
        self.assertEqual(source, before)
        self.assertEqual({k: v for k, v in mapped.items() if k != 'target'},
                         {k: v for k, v in unchanged.items() if k != 'target'})
        boot_index = next(i for i, op in enumerate(mapped['execution']) if op['op'] == 'boot')
        reuse = [op for op in mapped['execution'][boot_index+1:] if 'x' in op['inputs']]
        self.assertEqual([op['op'] for op in reuse], ['add_cc'])

    def test_scale_divide_changes_value_and_integer_restore_is_exact_dyadic_math(self):
        # Coefficients stay in the copied ciphertext; doubling its denominator
        # explicitly halves the represented value before native bootstrap.
        for coefficient in (0, 1, -1, 123456789012345, -(1 << 62)):
            original = Fraction(coefficient, 1 << 50)
            copied = Fraction(coefficient, 1 << 51)
            self.assertEqual(copied, original / 2)
            self.assertEqual(2 * copied, original)

    def test_inverse15_with_log2_normalization_preserves_legacy_default_and_plan(self):
        policy=NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,bootstrap_normalization_log2=3)
        self.assertEqual(policy.value_divisor,8)
        self.assertEqual(policy.metadata()['bootstrap_normalization_log2'],3)
        self.assertIn('mod1inv15-mr2-inputdiv8',policy.variant(32))
        self.assertEqual(NativeBinding().value_divisor,0)
        self.assertEqual(NativeBinding(50,mod1_inv_degree=7,log_message_ratio=2,bootstrap_input_divisor=2).value_divisor,2)
        for n in (-1,4,True):
            with self.assertRaises(ValueError):
                NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,bootstrap_normalization_log2=n)
        with self.assertRaises(ValueError):
            NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,bootstrap_input_divisor=2,bootstrap_normalization_log2=3)
        plan=make_bootstrap_plan({'target_id':'original'},branch=True)
        before=copy.deepcopy(plan)
        old,_=map_precision_plan(plan,NativeBinding(50,mod1_inv_degree=7,log_message_ratio=2),bootstrap_cache=32)
        new,_=map_precision_plan(plan,policy,bootstrap_cache=32)
        self.assertEqual(plan,before)
        self.assertEqual({k:v for k,v in new.items() if k!='target'},
                         {k:v for k,v in old.items() if k!='target'})
        for coefficient in (0,1,-1,123456789012345):
            self.assertEqual(8*Fraction(coefficient,1<<53),Fraction(coefficient,1<<50))


if __name__ == '__main__':
    unittest.main()
