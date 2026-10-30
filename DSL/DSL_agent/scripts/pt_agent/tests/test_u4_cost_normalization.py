"""U4 §7.8 frontend subtraction, ordered CSE and cost-only output reachability."""

import unittest
import numpy as np

from pt_agent.backends.cost_dag import CostDag
from pt_agent.cost import estimate,estimate_normalized,load_latency
from pt_agent.dsl.parser import parse
from pt_agent.placement import depth_dp


class CostDagTests(unittest.TestCase):
    def test_public_and_scalar_subtraction_reuse_negative_constant(self):
        b = CostDag(8)
        x = b.input(0)
        k = b.const([2.])
        a,c,d = x-k,x-2.,x+b.const([-2.])
        self.assertEqual(a.nid,c.nid)
        self.assertEqual(a.nid,d.nid)
        self.assertEqual(b.operands[a.nid][1],('plain',b.const([-2.]).name))
        np.testing.assert_array_equal(k.data,np.full(8,2.))
        b.output([a]);b.finalize()
        self.assertEqual(b.stats()['counts']['sub'],0)
        self.assertEqual(b.stats()['counts']['add'],1)
        self.assertEqual(b.stats()['distinct_consts'],1)

    def test_cipher_subtraction_shares_negation_and_retains_first_owner(self):
        b = CostDag(8)
        x,y = b.input(0),b.input(1)
        b.op_context = 3
        a = x-y
        b.op_context = 8
        c = x-y
        self.assertEqual(a.nid,c.nid)
        b.output([a]);b.finalize()
        self.assertEqual(b.kind,['input','input','neg','add'])
        self.assertEqual(b.prog_op[-2:],[3,3])
        self.assertEqual(b.args[-1],(0,2))

    def test_ordered_numbering_preserves_attributes_and_operand_order(self):
        b = CostDag(8)
        x,y = b.input(0),b.input(1)
        self.assertEqual(x.rotate(1).nid,x.rotate(1).nid)
        self.assertNotEqual(x.rotate(1).nid,x.rotate(2).nid)
        self.assertNotEqual((x*y).nid,(y*x).nid)
        self.assertEqual((x*b.const([2.])).nid,(x*b.const([2.])).nid)
        self.assertNotEqual((x*b.const([2.])).nid,(x*b.const([3.])).nid)

    def test_dead_nodes_removed_ancestors_order_and_outputs_preserved(self):
        b = CostDag(8)
        x = b.input(0)
        dead = x.rotate(2)
        a = x*x
        dead = dead*b.const([7.])
        out = a.rotate(1)+x
        b.output([out]);b.finalize()
        self.assertEqual(b.kind,['input','mul_cc','rot','add'])
        self.assertEqual(b.args,[(),(0,0),(1,),(2,0)])
        self.assertEqual(b._outputs[0].nid,3)
        self.assertEqual(b.stats()['rot_steps'],[1])
        self.assertEqual(b.stats()['distinct_consts'],0)
        self.assertEqual(b.normalization['unreachable_nodes'],2)
        self.assertEqual(sum(row['reachable'] for row in b.normalization['by_op']),4)
        self.assertEqual(sum(row['unreachable'] for row in b.normalization['by_op']),2)

    def test_hand_computed_chain_depth_and_bootstrap(self):
        b = CostDag(8)
        y = b.input(0)
        for _ in range(5):
            y = y*y
        b.output([y]);b.finalize()
        p = depth_dp(b,last_boot_level=4,boot_in_level=1,boot_out_level=5,
                     latency_table=load_latency('qwen48'))
        self.assertEqual(p.max_depth,5)
        self.assertEqual(p.cuts,[4])
        self.assertEqual(p.barriers,1)
        self.assertEqual(p.boots,1)

    def test_activation_scope_preserves_legacy_and_selects_token_graphs(self):
        text = ('@fhe.program(slots=1024)\ndef small(x: Cipher[1,4]):\n'
                ' a = pack(x, layout="flat")\n'
                ' z = act(a, fn="tanh", method="cheb", deg=3, domain=(-1.,1.))\n return z\n')
        p = parse(text)
        self.assertEqual(estimate(p,{}),estimate(p,{},frontend_normalize=False))
        self.assertEqual(estimate_normalized(p,{}),estimate(p,{},frontend_normalize=True))
        token = parse(text.replace('Cipher[1,4]','Cipher[1,2,4]').replace('layout="flat"','layout="token"'))
        self.assertEqual(estimate(token,{}),estimate(token,{},frontend_normalize=True))


if __name__=='__main__':
    unittest.main()
