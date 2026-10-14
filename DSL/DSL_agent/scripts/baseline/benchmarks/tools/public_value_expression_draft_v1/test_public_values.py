"""Independent real Torch comparisons, exact public bindings and rejection boundaries."""
import unittest
import numpy as np
import torch
from public_values import evaluate,reciprocal,signature
COUNTERS=dict(contexts=0,reference_groups=0)
class PublicValuesTests(unittest.TestCase):
 def step(self,op,attrs,parent="std",name="result"):return dict(name=name,op=op,input=parent,attributes=attrs)
 def compare(self,source,step,torch_value):
  values,record=evaluate({"std":source},[step]);out=values["result"]
  np.testing.assert_array_equal(out,torch_value.numpy())
  self.assertFalse(record["production_integrated"]);self.assertFalse(out.flags.writeable)
  for seed in range(4):
   rng=np.random.default_rng(100+seed);x=np.asarray(rng.uniform(-.3,.3,out.shape),dtype=np.float64)
   if source.dtype==np.dtype("bool"):
    actual=x*out.astype(np.float64);expected=(torch.from_numpy(x)*torch_value).numpy()
   else:
    actual=x*reciprocal(out);expected=(torch.from_numpy(x)/torch_value).numpy()
   np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
   COUNTERS['reference_groups']+=1
  COUNTERS['contexts']+=1
  return record
 def test_reshape_contexts(self):
  for shape,target in [((6,),[2,3]),((2,3),[1,2,3]),((1,),[])]:
   with self.subTest(shape=shape,target=target):
    a=np.linspace(.5,3.,np.prod(shape)).reshape(shape).astype(np.float64)
    self.compare(a,self.step("reshape",dict(shape=target)),torch.from_numpy(a).reshape(target))
 def test_permute_contexts(self):
  for shape,dims in [((2,3),[1,0]),((1,2,3),[2,0,1]),((1,2,1,3),[-1,0,1,2])]:
   with self.subTest(shape=shape,dims=dims):
    a=np.linspace(.5,3.,np.prod(shape)).reshape(shape).astype(np.float64)
    self.compare(a,self.step("permute",dict(dims=dims)),torch.from_numpy(a).permute(dims))
 def test_slice_contexts(self):
  for shape,selectors,fn in [
   ((3,),[None,dict(start=None,stop=None,step=None)],lambda a:a[None,:]),
   ((2,3),[dict(start=None,stop=None,step=None),dict(start=1,stop=None,step=None)],lambda a:a[:,1:]),
   ((2,2,3),[dict(start=None,stop=None,step=None),-1,1],lambda a:a[:,-1,1])]:
   with self.subTest(shape=shape):
    a=np.linspace(.5,3.,np.prod(shape)).reshape(shape).astype(np.float64)
    self.compare(a,self.step("slice",dict(selectors=selectors)),fn(torch.from_numpy(a)))
 def test_bool_cast_contexts(self):
  for shape,dtype in [((3,),"float32"),((2,1),"float64"),((1,2,1,3),"float32")]:
   with self.subTest(shape=shape,dtype=dtype):
    a=(np.arange(np.prod(shape)).reshape(shape)%2==0)
    self.compare(a,self.step("bool_cast",dict(dtype=dtype)),torch.from_numpy(a).to(getattr(torch,dtype)))
 def test_chain_matches_torch(self):
  a=np.linspace(.5,3.,6).reshape(2,3)
  steps=[self.step("permute",dict(dims=[1,0]),name="transpose"),self.step("reshape",dict(shape=[1,6]),parent="transpose")]
  values,r=evaluate({"std":a},steps);np.testing.assert_array_equal(values["result"],torch.from_numpy(a).T.reshape(1,6).numpy())
  self.assertEqual(len(r["steps"]),2)
 def test_binding_changes_with_public_content(self):
  a=np.array([1.,2.]);p=[self.step("reshape",dict(shape=[1,2]))]
  self.assertNotEqual(evaluate({"std":a},p)[1]["binding"],evaluate({"std":a+1},p)[1]["binding"])
 def test_binding_changes_with_operation(self):
  a=np.ones((2,2));self.assertNotEqual(evaluate({"std":a},[self.step("reshape",dict(shape=[4]))])[1]["binding"],evaluate({"std":a},[self.step("permute",dict(dims=[1,0]))])[1]["binding"])
 def test_source_not_mutated(self):
  a=np.array([[1.,2.],[3.,4.]]);before=a.copy();evaluate({"std":a},[self.step("permute",dict(dims=[1,0]))]);np.testing.assert_array_equal(a,before)
 def test_dtype_is_bound(self):self.assertNotEqual(signature(np.ones(3))["sha256"],signature(np.ones(3,dtype=np.bool_))["sha256"])
 def reject(self,step):
  with self.assertRaises(ValueError):evaluate({"std":np.ones((2,3))},[step])
 def test_encrypted_dependency_rejected(self):self.reject(self.step("reshape",dict(shape=[6]),parent="x"))
 def test_cycle_rejected(self):self.reject(self.step("reshape",dict(shape=[6]),parent="result"))
 def test_name_overwrite_rejected(self):self.reject(self.step("reshape",dict(shape=[6]),name="std"))
 def test_extra_attributes(self):self.reject(self.step("reshape",dict(shape=[6],file="/tmp/x")))
 def test_unregistered_operation(self):self.reject(self.step("call",dict()))
 def test_boolean_dimension(self):self.reject(self.step("reshape",dict(shape=[True,6])))
 def test_multiple_inferred_dimensions(self):self.reject(self.step("reshape",dict(shape=[-1,-1])))
 def test_shape_changes_element_count(self):self.reject(self.step("reshape",dict(shape=[7])))
 def test_empty_shape_result(self):self.reject(self.step("reshape",dict(shape=[0,6])))
 def test_rank_capacity(self):self.reject(self.step("reshape",dict(shape=[1]*33)))
 def test_element_capacity(self):
  with self.assertRaises(ValueError):evaluate({"std":np.ones(4097)},[])
 def test_empty_state(self):
  with self.assertRaises(ValueError):evaluate({},[])
 def test_nonfinite(self):
  for n in (np.nan,np.inf):
   with self.subTest(n=n),self.assertRaises(ValueError):evaluate({"std":np.array([n])},[])
 def test_object_values(self):
  with self.assertRaises(ValueError):evaluate({"std":np.array([object()],dtype=object)},[])
 def test_repeated_axes(self):self.reject(self.step("permute",dict(dims=[0,0])))
 def test_invalid_axis(self):self.reject(self.step("permute",dict(dims=[0,2])))
 def test_boolean_axis(self):self.reject(self.step("permute",dict(dims=[True,0])))
 def test_negative_slice_step(self):self.reject(self.step("slice",dict(selectors=[dict(start=None,stop=None,step=-1)])))
 def test_out_of_bounds_index(self):self.reject(self.step("slice",dict(selectors=[2])))
 def test_empty_slice(self):self.reject(self.step("slice",dict(selectors=[dict(start=2,stop=2,step=None)])))
 def test_too_many_indices(self):self.reject(self.step("slice",dict(selectors=[0,0,0])))
 def test_boolean_index(self):self.reject(self.step("slice",dict(selectors=[True])))
 def test_integer_bool_cast_rejected(self):self.reject(self.step("bool_cast",dict(dtype="float64")))
 def test_unknown_cast_dtype(self):
  with self.assertRaises(ValueError):evaluate({"std":np.ones(3,dtype=np.bool_)},[self.step("bool_cast",dict(dtype="int8"))])
 def test_original_integer_requires_explicit_future_contract(self):
  with self.assertRaises(ValueError):evaluate({"std":np.array([1,2],dtype=np.int64)},[])
 def test_scalar_reciprocal(self):
  value=reciprocal(np.asarray(2.));self.assertEqual(value.shape,());self.assertEqual(value.item(),.5)
 def test_attributes_do_not_alias_input(self):
  step=self.step("reshape",dict(shape=[1,2]));_,record=evaluate({"std":np.ones(2)},[step])
  step["attributes"]["shape"][0]=2
  self.assertEqual(record["steps"][0]["attributes"]["shape"],[1,2])
 def test_operation_is_data(self):
  class Unexpected:
   def __eq__(self,other):raise AssertionError("arbitrary equality must not execute")
  self.reject(self.step(Unexpected(),{}))
 def test_cast_dtype_is_data(self):
  class Unexpected:
   def __eq__(self,other):raise AssertionError("arbitrary equality must not execute")
  with self.assertRaises(ValueError):evaluate({"std":np.ones(3,dtype=np.bool_)},[self.step("bool_cast",dict(dtype=Unexpected()))])
 def test_divisor_zero(self):
  with self.assertRaises(ValueError):reciprocal(np.array([0.]))
 def test_divisor_large(self):
  with self.assertRaises(ValueError):reciprocal(np.array([1025.]))
 def test_divisor_inverse_large(self):
  with self.assertRaises(ValueError):reciprocal(np.array([1e-10]))
 def test_divisor_float32_requires_normalization_contract(self):
  with self.assertRaises(ValueError):reciprocal(np.array([1.],dtype=np.float32))
if __name__=="__main__":unittest.main()
