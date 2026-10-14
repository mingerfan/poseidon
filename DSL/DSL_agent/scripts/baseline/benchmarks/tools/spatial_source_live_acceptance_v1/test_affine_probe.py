"""Independent public-affine diagnostic tests; no provider/frontend/backend."""
import copy,unittest
from affine_probe import analyze
class AffineTests(unittest.TestCase):
 def request(self):
  return dict(model=dict(inputs=[dict(name="x",shape=[2,2])],constants=dict(w=[[1.,2.],[3.,4.]]),
   nodes=[dict(op="linear",inputs=["x","w"],attrs={},outputs=["y"])],outputs=[dict(name="out",value="y")]),
   public_constants=dict(pv0=[1.,4.,1.,4.],pv1=[2.,0.,2.,0.],pv2=[0.,3.,0.,3.]),
   layout=dict(input_slot_period=4,inputs=[dict(name="x",dsl_name="x0",offset=0,elements=4)],
    output_ciphertexts=1,output_selectors=[[0,i]for i in range(4)]))
 def source(self,expression):
  return '@hc.func("c,c")\ndef golden(x0, zero_ct):\n    return ['+expression+']\n'
 def correct(self):
  return "x0 * pv0 + x0.rotate(1) * pv1 + x0.rotate(1).rotate(2) * pv2"
 def test_weighted_linear_reduction(self):
  self.assertTrue(analyze(self.request(),self.source(self.correct()))["exact_public_affine_match"])
 def test_leading_linear_dimensions(self):
  q=self.request();q["model"]["inputs"][0]["shape"]=[1,2,2]
  self.assertTrue(analyze(q,self.source(self.correct()))["exact_public_affine_match"])
 def test_elementwise_is_not_linear(self):
  result=analyze(self.request(),self.source("x0 * pv0"))
  self.assertFalse(result["exact_public_affine_match"]);self.assertEqual(result["maximum_absolute_coefficient_difference"],3.)
 def test_reshape_preserves_order(self):
  q=self.request();q["model"]["nodes"]=[dict(op="reshape",inputs=["x"],attrs=dict(shape=[4]),outputs=["y"])]
  self.assertTrue(analyze(q,self.source("x0"))["exact_public_affine_match"])
 def test_permute_changes_coordinates(self):
  q=self.request();q["model"]["nodes"]=[dict(op="permute",inputs=["x"],attrs=dict(dims=[1,0]),outputs=["y"])]
  q["public_constants"]=dict(a=[1.,0.,0.,1.],b=[0.,1.,0.,0.],c=[0.,0.,1.,0.])
  source=self.source("x0 * a + x0.rotate(1) * b + x0.rotate(1).rotate(2) * c")
  self.assertTrue(analyze(q,source)["exact_public_affine_match"])
  self.assertFalse(analyze(q,self.source("x0"))["exact_public_affine_match"])
 def test_bias_uses_constant_coordinate(self):
  q=self.request();q["model"]["constants"]["bias"]=[1.,-1.]
  q["model"]["nodes"][0]["inputs"].append("bias");q["public_constants"]["b"]=[1.,-1.,1.,-1.]
  self.assertTrue(analyze(q,self.source(self.correct()+" + b"))["exact_public_affine_match"])
 def test_alias_rebind_has_same_expression(self):
  source='@hc.func("c,c")\ndef golden(x0, zero_ct):\n    a = x0\n    b = a\n    return ['+self.correct().replace("x0","b")+']\n'
  self.assertTrue(analyze(self.request(),source)["exact_public_affine_match"])
 def test_encrypted_zero_has_no_input_coordinate(self):
  result=analyze(self.request(),self.source("zero_ct"))
  self.assertEqual(result["public_input_elements"],4);self.assertFalse(result["exact_public_affine_match"])
 def test_multiple_inputs_tail_and_output_binding(self):
  q=dict(model=dict(inputs=[dict(name="a",shape=[2]),dict(name="b",shape=[2])],constants={},nodes=[],
   outputs=[dict(name="out_a",value="a"),dict(name="out_b",value="b")]),public_constants={},
   layout=dict(input_slot_period=4,inputs=[dict(name="a",dsl_name="x0",offset=0,elements=2),
    dict(name="b",dsl_name="x1",offset=0,elements=2)],output_ciphertexts=2,output_selectors=[[0,0],[0,1],[1,0],[1,1]]))
  source='@hc.func("c,c,c")\ndef golden(x0,x1,zero_ct):\n    return [x0,x1]\n'
  self.assertTrue(analyze(q,source)["exact_public_affine_match"])
 def test_unregistered_call_rejected_without_execution(self):
  with self.assertRaisesRegex(ValueError,"unsupported_probe_source_syntax"):
   analyze(self.request(),self.source('__import__("os").system("false")'))
 def test_encrypted_multiply_rejected(self):
  with self.assertRaisesRegex(ValueError,"non_affine_encrypted_multiply"):analyze(self.request(),self.source("x0*x0"))
 def test_wrong_public_period_rejected(self):
  q=self.request();q["public_constants"]["pv0"]=[1.,4.]
  with self.assertRaisesRegex(ValueError,"public_constant_period"):analyze(q,self.source(self.correct()))
 def test_wrong_weight_shape_rejected(self):
  q=self.request();q["model"]["constants"]["w"]=[[1.],[2.]]
  with self.assertRaisesRegex(ValueError,"linear_public_weight_shape"):analyze(q,self.source(self.correct()))
 def test_unsupported_graph_operator_rejected(self):
  q=self.request();q["model"]["nodes"][0]["op"]="square"
  with self.assertRaisesRegex(ValueError,"unsupported_probe_graph_operator"):analyze(q,self.source(self.correct()))
 def test_probe_capacity_is_bounded(self):
  q=self.request();q["model"]["inputs"][0]["shape"]=[65]
  with self.assertRaisesRegex(ValueError,"probe_capacity"):analyze(q,self.source(self.correct()))
 def test_private_arrays_are_not_required(self):
  r=analyze(self.request(),self.source(self.correct()))
  self.assertFalse(r["private_test_inputs_used"]);self.assertFalse(r["private_reference_outputs_used"])
if __name__=="__main__":unittest.main()
