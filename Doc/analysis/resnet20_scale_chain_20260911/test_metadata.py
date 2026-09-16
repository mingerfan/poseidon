"""Lightweight public-metadata tests; never instantiate an HE context."""
import copy
import unittest
from search import TREES, actual_chain, ntt_primes
from verify import candidate, verify


class MetadataTest(unittest.TestCase):
    def test_tree_shape(self):
        self.assertEqual(TREES[15], ((8,(4,None,None),(4,None,(2,None,None))),4,2))
        self.assertEqual(TREES[27], ((16,(8,None,None),(8,None,None)),5,3))

    def test_small_prime_supply(self):
        self.assertEqual(ntt_primes(20),(786433,))
        with self.assertRaises(ValueError):
            actual_chain([20,20])

    def test_candidates(self):
        for mixed in (False,True):
            _,primes,plan = candidate(mixed)
            checks = verify(plan,primes)
            self.assertEqual(checks['total_q_dropped'],22)
            self.assertEqual(checks['output_q_count'],16)
            self.assertGreaterEqual(checks['minimum_plain_logscale'],40)
            self.assertEqual([s['end']-s['start'] for s in plan['stages']],[6,6,7])
            self.assertEqual(plan['tail_drop'],3)

    def test_corrupt_scale_rejected(self):
        _,primes,plan = candidate()
        bad = copy.deepcopy(plan)
        bad['stages'][1]['tree']['scale'] += .25
        with self.assertRaises(AssertionError):
            verify(bad,primes)

    def test_missing_modulus_rejected(self):
        _,primes,plan = candidate()
        bad = copy.deepcopy(plan)
        bad['total'] -= 1
        with self.assertRaises(AssertionError):
            verify(bad,primes)

    def test_misaligned_branch_rejected(self):
        _,primes,plan = candidate()
        bad = copy.deepcopy(plan)
        bad['stages'][0]['tree']['quotient']['output'] += 1
        with self.assertRaises(AssertionError):
            verify(bad,primes)


if __name__ == '__main__':
    unittest.main()
