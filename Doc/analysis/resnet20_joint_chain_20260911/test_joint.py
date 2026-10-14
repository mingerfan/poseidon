"""Positive witness and corrupted-ledger rejection tests. No HE context needed."""
import copy
import json
from pathlib import Path
import unittest
from verify_joint import verify


class JointLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.witness=json.loads((Path(__file__).parent/'q50.json').read_text())

    def reject(self,mutator):
        bad=copy.deepcopy(self.witness)
        mutator(bad)
        with self.assertRaises(AssertionError):verify(bad)

    def test_same_chain_closes(self):
        out=verify(self.witness)
        self.assertEqual(out['status'],'PASS_SCALE_LEDGER')
        self.assertEqual((out['full_q'],out['bootstrap_out'],out['relu_out'],out['conv_out'],out['s2c_out']),
                         (50,31,9,6,2))
        self.assertEqual(out['relu_drop'],22)
        self.assertGreaterEqual(out['min_relu_plain_logscale'],40)
        self.assertGreaterEqual(out['min_effective_eval_coefficient_logscale'],40)
        self.assertFalse(self.witness.get('normalization'))

    def test_relabel_without_coefficient_folding_rejected(self):
        self.reject(lambda r:r['bootstrap']['fold'].__setitem__('enabled',False))

    def test_wrong_folded_polynomial_rejected(self):
        self.reject(lambda r:r['bootstrap']['fold']['folded_coefficients'][0].__setitem__(0,1.0))

    def test_unfolded_double_angle_constant_rejected(self):
        self.reject(lambda r:r['bootstrap']['fold']['folded_constants'].__setitem__(
            1,r['bootstrap']['fold']['original_constants'][1]))

    def test_wrong_relu_input_scale_rejected(self):
        self.reject(lambda r:r['relu']['stages'][0].__setitem__('input_scale',41))

    def test_missing_physical_prime_rejected(self):
        self.reject(lambda r:r['q_bottom_first'].pop())

    def test_duplicate_auxiliary_prime_rejected(self):
        self.reject(lambda r:r['p'].__setitem__(0,r['q_bottom_first'][0]))

    def test_branch_level_mismatch_rejected(self):
        self.reject(lambda r:r['relu']['stages'][0]['tree']['quotient'].__setitem__('output',0))

    def test_scale_overflow_rejected(self):
        self.reject(lambda r:r['relu']['stages'][0]['tree'].__setitem__('pre',10000))

    def test_omitted_convolution_rescale_rejected(self):
        self.reject(lambda r:r['conv'].pop(2))

    def test_wrong_preparation_integer_rejected(self):
        self.reject(lambda r:r['preparation'].__setitem__('multiplier',1))

    def test_wrong_evalmod_drop_rejected(self):
        self.reject(lambda r:r['bootstrap']['double_angle_drops'].__setitem__(0,2))


if __name__=='__main__':unittest.main()
