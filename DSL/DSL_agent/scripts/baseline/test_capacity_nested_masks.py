"""Bound-mask containment/disjointness; independent slot-ring differences."""
import copy,unittest
import numpy as np
from capacity_algebraic_schedule import schedule,BalancedSchedule
from test_capacity_neutral_factors import straight_line
def request(a,b,*,trusted=True):
 p=len(a)
 return dict(layout=dict(input_slot_period=p,inputs=[dict(dsl_name="x0")],
  auxiliary_ciphertexts=[dict(dsl_name="zero_ct",kind="encrypted_zero",source="trusted_client",slot_period=p)]if trusted else[]),
  public_constants=dict(a=a,b=b,gain=.375))
def program(expression):return '@hc.func("c,c")\ndef golden(x0, zero_ct):\n    v0 = '+expression+'\n    return [v0]\n'
class MaskTests(unittest.TestCase):
 def check(self,a,b,expr="(x0 * a) * b",*,trusted=True,expected=True):
  r=request(a,b,trusted=trusted);saved=copy.deepcopy(r);before=program(expr)
  after,record=schedule(before,r);self.assertEqual(saved,r)
  self.assertEqual(record["applied"],expected)
  self.assertFalse(record["provider_candidate_rewritten"]);self.assertFalse(record["depth_is_capacity_proof"])
  for x in [np.zeros(4),np.array([.2,-.4,.7,-.9]),np.array([1.,2.,3.,4.]),
            np.random.default_rng(191).uniform(-1,1,4)]:
   np.testing.assert_allclose(straight_line(after,r,x),straight_line(before,r,x),atol=0,rtol=0)
  self.assertEqual((after,record),schedule(before,r))
  return after,record
 def test_previous_mask_subset(self):
  _,r=self.check([1,0,0,0],[1,1,0,0]);self.assertEqual(r["removed_nested_bound_masks"]+r["removed_supported_bound_masks"],1)
 def test_target_mask_subset(self):
  _,r=self.check([1,1,0,0],[1,0,0,0]);self.assertEqual(r["removed_nested_bound_masks"]+r["removed_supported_bound_masks"],1)
 def test_equal_content_different_bound_names(self):
  _,r=self.check([1,0,1,0],[1,0,1,0]);self.assertEqual(r["removed_nested_bound_masks"]+r["removed_supported_bound_masks"],1)
 def test_disjoint_masks_retained_with_declared_cipher_zero(self):
  _,r=self.check([1,0,0,0],[0,1,0,0],expected=False);self.assertEqual(r["removed_disjoint_bound_masks"],0)
 def test_disjoint_without_trusted_zero_retained(self):
  self.check([1,0,0,0],[0,1,0,0],trusted=False,expected=False)
 def test_partial_overlap_retained(self):self.check([1,1,0,0],[0,1,1,0],expected=False)
 def test_numeric_weights_support_binary_mask(self):self.check([.5,0,0,0],[1,1,0,0])
 def test_boolean_masks_retained(self):self.check([True,False,False,False],[1,1,0,0],expected=False)
 def test_rotation_changes_mask_coordinates(self):
  self.check([1,0,0,0],[1,1,0,0],"(x0 * a).rotate(1) * b",expected=False)
 def test_public_gain_retained(self):
  s,r=self.check([1,0,0,0],[1,1,0,0],"(x0 * a) * b + x0 * gain");self.assertIn("gain",s)
 def test_nested_chain(self):
  self.check([1,1,0,0],[1,0,0,0],"((x0 * a) * b) * a")
 def test_allzero_mask_not_an_identity(self):self.check([0,0,0,0],[1,1,0,0],expected=False)
 def test_rotated_support_matches_destination_mask(self):
  _,r=self.check([1,0,0,0],[0,0,0,1],"(x0 * a).rotate(1) * b")
  self.assertGreater(r["removed_supported_bound_masks"],0)
 def test_cipher_product_support(self):
  self.check([1,0,0,0],[1,1,0,0],"(x0 * a) * x0 * b")
 def test_squared_support(self):
  self.check([1,0,0,0],[1,1,0,0],"(x0 * a) * (x0 * a) * b")
 def test_union_of_two_masked_terms(self):
  self.check([1,0,0,0],[1,1,0,0],"(x0 * a + (x0 * a).rotate(3)) * b")
 def test_unknown_addition_keeps_required_mask(self):
  self.check([1,0,0,0],[1,1,0,0],"(x0 * a + x0) * b",expected=True)
 def test_slot_support_recursion_budget(self):
  r=request([1,0,0,0],[1,1,0,0]);s=BalancedSchedule(r)
  with self.assertRaisesRegex(ValueError,"slot-support recursion budget"):s.slot_support(s.leaf("x0"),256)
 def test_recursion_budget_retained(self):
  r=request([1,0,0,0],[1,1,0,0]);s=BalancedSchedule(r)
  with self.assertRaisesRegex(ValueError,"recursion budget"):s.factor(s.leaf("x0"),s.leaf("a",plain=True),256)
if __name__=="__main__":unittest.main()
