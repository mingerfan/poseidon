"""CPU-only fixture regressions. These do not certify HE precision."""
import copy
import json
import unittest
from pathlib import Path

from export_relu_precision import HERE, DEFAULT_COEFFICIENTS, attach_coefficients, export, leaves


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((HERE/'q50.json').read_text())
        self.values = list(map(float, DEFAULT_COEFFICIENTS.read_text().split()))

    def test_coefficient_order_is_heap_not_dfs(self):
        stages = self.report['relu']['stages']
        sentinel = [0 if i%2 == 0 else float(i) for i in range(len(self.values))]
        attach_coefficients(stages, sentinel)
        offset = 0
        for stage, factor in zip(stages, (0.5, 1/1.7, 0.5)):
            for _, n in sorted(leaves(stage['tree'])):
                self.assertEqual(n['coefficients'], [sentinel[offset+i]*factor for i in range(1,n['degree']+1,2)])
                offset += n['degree']+1
        self.assertEqual(offset,len(sentinel))

    def test_original_coefficients_not_refitted(self):
        stages=self.report['relu']['stages']
        attach_coefficients(stages,self.values)
        offset=0
        for stage,factor in zip(stages,(0.5,1/1.7,0.5)):
            for _,n in sorted(leaves(stage['tree'])):
                for i,c in enumerate(n['coefficients']):
                    self.assertEqual(c,self.values[offset+2*i+1]*factor)
                offset+=n['degree']+1

    def test_reject_truncated_or_extra_file(self):
        for values in (self.values[:-1],self.values+[0]):
            with self.assertRaises(ValueError):
                attach_coefficients(copy.deepcopy(self.report['relu']['stages']),values)

    def test_unused_even_position_guard(self):
        self.values[0]=1
        with self.assertRaises(ValueError):
            attach_coefficients(self.report['relu']['stages'],self.values)

    def test_saved_fixture_deterministic(self):
        self.assertEqual(export(HERE/'q50.json',DEFAULT_COEFFICIENTS),(HERE/'relu_precision_fixture.txt').read_text())

    def test_actual_prime_witness(self):
        r=self.report['relu']
        self.assertEqual([(31-s['start'],31-s['end']) for s in r['stages']],[(31,25),(25,19),(19,12)])
        self.assertEqual((r['tail_work'],r['tail_drop'],r['total']),(19,3,22))
        self.assertEqual(len(self.report['q_bottom_first']),50)
        self.assertEqual(len(self.report['p']),25)

    def test_fused_leaf_inventory_without_term_pruning(self):
        stages = self.report['relu']['stages']
        attach_coefficients(stages, self.values)
        nodes = [node for stage in stages for _, node in leaves(stage['tree'])]
        terms = [len(node['coefficients']) for node in nodes]
        self.assertEqual(len(nodes), 14)
        self.assertEqual(sum(terms), 30)
        self.assertEqual(sum(count - 1 for count in terms), 16)
        self.assertEqual(max(terms), 4)


if __name__=='__main__':
    unittest.main()
