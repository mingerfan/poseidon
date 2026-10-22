import importlib.util,unittest
from pathlib import Path
import numpy as np,torch
from benchmark_graph import canonical,samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from operator_reference import evaluate as high_reference
from pytorch_model_capture import capture,functional_input_rules
from pytorch_source_contract import output_tree
from pytorch_split_plan import arguments,plan,VERSION

path=Path(__file__).parent/"cases/trusted-pytorch-split-v1/models.py"
spec=importlib.util.spec_from_file_location("split_unit_sources",path)
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
def captured(module,shape,names):
 return capture(module,[dict(name="x",shape=shape)],names,resource_profile="general-tensors-v1",
  dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,
  capture_policy="signature-static-index-v1",approximation_policy="node-module-default-v1",
  matrix_policy="static-tensor-matmul-v1",
  approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))

class SplitCaptureTests(unittest.TestCase):
 def compare(self,family,context):
  model=fixtures.make_model(family,context);dims=fixtures.SHAPES[family][context];names=fixtures.source_outputs(family,context)
  original,low,binding=captured(model,dims,names)
  self.assertEqual((original,low,binding),captured(model,dims,names))
  for values in samples(low,16):
   with torch.no_grad():raw=model(torch.from_numpy(values["x"]))
   leaves,tree=output_tree(raw,lambda v:isinstance(v,torch.Tensor),names,scalar_tensor=lambda v:v.ndim==0)
   self.assertEqual(tree,binding["pytorch_capture"]["output_structure"])
   a=mathematical(low,values);b=tensor_reference(low,values);h=high_reference(original,values)
   for name,tensor in zip(names,leaves):
    for expected in (b[name],h[name],tensor.detach().numpy()):
     np.testing.assert_allclose(a[name],expected,atol=2e-12,rtol=2e-12)
  return original,low,binding
 def test_function_sections_three_contexts(self):
  for n in range(3):self.compare("function_sections",n)
 def test_method_size_and_remainder_three_contexts(self):
  for n in range(3):self.compare("method_size",n)
 def test_keyword_and_rank32_three_contexts(self):
  for n in range(3):self.compare("keyword_sections",n)
 def test_direct_tuple_outputs_three_contexts(self):
  for n in range(3):
   _,_,b=self.compare("tuple_output",n)
   self.assertEqual(b["pytorch_capture"]["output_structure"]["kind"],"tuple")
 def test_public_piece_loop_three_contexts(self):
  for n in range(3):
   a,_,_=self.compare("public_loop",n)
   self.assertEqual(sum(x["op"]=="slice" for x in a["nodes"]),5+n)
 def test_public_weight_split_three_contexts(self):
  for n in range(3):self.compare("public_weight",n)
 def test_single_piece_split_preserves_tensor_tuple(self):
  class Model(torch.nn.Module):
   def forward(self,x):return torch.split(x,8,dim=0)
  original,low,b=captured(Model(),[3],["out"])
  self.assertEqual(b["pytorch_capture"]["output_structure"]["kind"],"tuple")
  self.assertEqual(len(original["nodes"]),1)
  values=samples(low,4)[1];np.testing.assert_array_equal(mathematical(low,values)["out"],values["x"])
 def test_static_shape_queries_and_tuple_slices(self):
  class Model(torch.nn.Module):
   def forward(self,x):
    parts=x.split(2,-1);y=parts[-1]
    return torch.cat(parts[1:],-1).reshape(x.size(0),-1),y.reshape(y.shape[0],y.numel()//y.shape[0])
  a,low,b=captured(Model(),[2,6],["joined","tail"])
  for v in samples(low,16):
   result=Model()(torch.from_numpy(v["x"]))
   ref=mathematical(low,v)
   for name,t in zip(("joined","tail"),result):np.testing.assert_allclose(ref[name],t.numpy(),atol=2e-12,rtol=2e-12)
 def test_split_size_one_many_internal_ports(self):
  class Model(torch.nn.Module):
   def forward(self,x):
    parts=x.split(1,0);out=parts[0]
    for part in parts[1:]:out=out+part
    return out
  _,low,_=captured(Model(),[9],["out"])
  for v in samples(low,16):np.testing.assert_allclose(mathematical(low,v)["out"],np.sum(v["x"]).reshape(1),atol=2e-12,rtol=2e-12)
 def test_legacy_mode_is_unchanged_and_closed(self):
  class Model(torch.nn.Module):
   def forward(self,x):return torch.split(x,2)[0]
  with self.assertRaisesRegex(ValueError,"Unsupported PyTorch call"):
   capture(Model(),[dict(name="x",shape=[4])],["out"])
 def test_static_non_index_policy_keeps_split_closed(self):
  class Model(torch.nn.Module):
   def forward(self,x):return x.split(2)[0]
  with self.assertRaisesRegex(ValueError,"Unsupported PyTorch call"):
   capture(Model(),[dict(name="x",shape=[4])],["out"],preserve_output_structure=True,capture_policy="signature-static-v1")
 def test_empty_sections_are_rejected_before_real_shape_propagation(self):
  class Model(torch.nn.Module):
   def forward(self,x):return x.split([0,4])[1]
  from unittest.mock import patch
  with patch("torch.fx.passes.shape_prop.ShapeProp.propagate",side_effect=AssertionError("Real propagation forbidden")):
   with self.assertRaisesRegex(ValueError,"Static split sections must be positive"):captured(Model(),[4],["out"])
 def test_split_node_expansion_is_bounded_before_real_propagation(self):
  class Model(torch.nn.Module):
   def forward(self,x):return x.split(1)[0]
  from unittest.mock import patch
  with patch("torch.fx.passes.shape_prop.ShapeProp.propagate",side_effect=AssertionError("Real propagation forbidden")):
   with self.assertRaisesRegex(ValueError,"Static FX expansion budget"):
    capture(Model(),[dict(name="x",shape=[64])],["out"],preserve_output_structure=True,
     capture_policy="signature-static-index-v1")
 def test_encrypted_split_size_is_rejected(self):
  class Model(torch.nn.Module):
   def forward(self,x):return torch.split(x,x[0])[0]
  with self.assertRaises(ValueError):captured(Model(),[4],["out"])
 def test_encrypted_axis_is_rejected(self):
  class Model(torch.nn.Module):
   def forward(self,x):return torch.split(x,2,dim=x[0])[0]
  with self.assertRaises(ValueError):captured(Model(),[4],["out"])
 def test_unregistered_source_shape_is_not_executed_to_infer_split(self):
  class Model(torch.nn.Module):
   def forward(self,x):return torch.sin(x).split(2)[0]
  with self.assertRaisesRegex(ValueError,"Static shape unavailable"):captured(Model(),[4],["out"])
 def test_bool_size_and_axis_are_rejected(self):
  for sections,axis in ((True,0),(2,True)):
   with self.assertRaises(ValueError):plan([4],sections,axis)
 def test_invalid_size_sections_and_axis(self):
  for args in (([4],0,0),([4],-1,0),([4],[1,2],0),([4],[0,4],0),([4],[2,2],1),([4],[2,2],-2),([4],[2.,2.],0),([4],None,0)):
   with self.subTest(args=args),self.assertRaises(ValueError):plan(*args)
 def test_duplicate_unknown_and_method_input_keywords_rejected(self):
  for args,kw,method in (((object(),2),dict(split_size_or_sections=2),False),
       ((object(),2),dict(out=None),False),((object(),2),dict(split_size_or_sections=2),True),
       ((object(),2),dict(input=object()),True)):
   with self.assertRaises(ValueError):arguments(args,kw,method=method)
 def test_function_and_method_keyword_signatures_are_distinct(self):
  x=object()
  self.assertEqual(arguments((),dict(tensor=x,split_size_or_sections=[2,2],dim=-1)),dict(input=x,sections=[2,2],axis=-1))
  self.assertEqual(arguments((x,),dict(split_size=2),method=True),dict(input=x,sections=2,axis=0))
 def test_functional_capability_uses_explicit_split_version(self):
  row=functional_input_rules()["tensor_split"]
  self.assertEqual(row["version"],VERSION)
  self.assertEqual(row["calls"],["torch.split","Tensor.split"])
 def test_rank_over_profile_rejected(self):
  with self.assertRaises(ValueError):plan([1]*33,1,0)

if __name__=="__main__":unittest.main()
