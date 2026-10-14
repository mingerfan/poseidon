"""Exact public support, with independent finite-ring evaluation."""
import unittest,copy
import numpy as np
from capacity_algebraic_schedule import schedule,BalancedSchedule
from test_capacity_nested_masks import request,program
from test_capacity_neutral_factors import straight_line

class WeightedSupportTests(unittest.TestCase):
    def check(self,weights,mask,expression,removed):
        r=request(weights,mask);before=copy.deepcopy(r);original=program(expression)
        result,record=schedule(original,r)
        self.assertEqual(r,before)
        self.assertEqual(bool(record["removed_supported_bound_masks"]),removed)
        self.assertFalse(record["provider_candidate_rewritten"])
        self.assertFalse(record["depth_is_capacity_proof"])
        for x in (np.zeros(4),np.array([.2,-.4,.7,-.9]),np.arange(4.)+.1,
                  np.random.default_rng(1664).uniform(-1,1,4)):
            np.testing.assert_allclose(straight_line(result,r,x),straight_line(original,r,x),rtol=2e-15,atol=1e-300)
        self.assertEqual((result,record),schedule(original,r))
        return result
    def test_signed_weights_with_mask_superset(self):
        self.check([-.5,0.,.25,0.],[1.,0.,1.,0.],"(x0 * a) * b",True)
    def test_gain_is_retained(self):
        result=self.check([-.5,0.,.25,0.],[1.,0.,1.,0.],"(x0 * a) * b",True)
        self.assertIn("a",result)
    def test_weights_are_not_idempotent(self):
        self.check([.5,0.,.25,0.],[1.,0.,1.,0.],"(x0 * a) * a",False)
    def test_tiny_nonzero_outside_mask_retains_mask(self):
        self.check([.5,1e-300,0.,0.],[1.,0.,0.,0.],"(x0 * a) * b",False)
    def test_tiny_nonzero_inside_mask_not_truncated(self):
        self.check([1e-200,0.,0.,0.],[1.,0.,0.,0.],"(x0 * a) * b",True)
    def test_addition_may_expand_support(self):
        result=self.check([.5,0.,0.,0.],[1.,0.,0.,0.],"(x0 * a + x0) * b",True)
        self.assertIn("* b",result)  # The previously unmasked x0 still needs b.
    def test_rotated_destination_support(self):
        self.check([.5,0.,0.,0.],[0.,0.,0.,1.],"(x0 * a).rotate(1) * b",True)
    def test_wrong_rotated_destination_keeps_mask(self):
        self.check([.5,0.,0.,0.],[1.,0.,0.,0.],"(x0 * a).rotate(1) * b",False)
    def test_cipher_product_support(self):
        self.check([.5,0.,0.,0.],[1.,0.,0.,0.],"(x0 * a) * x0 * b",True)
    def test_union_of_weighted_support(self):
        self.check([.5,0.,0.,0.],[1.,1.,0.,0.],"(x0 * a + (x0 * a).rotate(3)) * b",True)
    def test_repeated_scalar_is_not_period_vector(self):
        for weights in(.5,[.5],True,[True,False,False,False],[0.,0.,0.,0.],
                       [.5,0.,0.],[.5,0.,0.,0.,0.],[.5,float("nan"),0.,0.]):
            r=request([.5,0.,0.,0.],[1.,0.,0.,0.]);r["public_constants"]["a"]=weights
            s=BalancedSchedule(r)
            self.assertIsNone(s.weighted_slots(s.leaf("a",plain=True)))
    def test_ciphertext_leaf_support_remains_unknown(self):
        s=BalancedSchedule(request([.5,0.,0.,0.],[1.,0.,0.,0.]))
        self.assertIsNone(s.weighted_slots(s.leaf("x0")))
        self.assertEqual(s.slot_support(s.leaf("x0")),15)
if __name__=="__main__":unittest.main()
