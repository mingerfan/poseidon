"""Static transforms: independent math, copy effects, boundaries and compatibility."""
import copy, unittest
from unittest.mock import patch
import numpy as np
import torch
from benchmark_graph import samples, digest
from benchmark_math import evaluate as core
from operator_arithmetic_graph import decompose,evaluate_reference
from operator_tensor_transform_graph import FORMAT
from pytorch_static_tensor_transforms import POLICY,normalize,check
from pytorch_static_trace import trace,ALIAS_POLICY
from static_tensor_transform_plan import plan

CASES=[
 ("repeat",[2],dict(repeats=[2])),("repeat",[2,2],dict(repeats=[2,1])),
 ("repeat",[1,2],dict(repeats=[2,2,1])),("tile",[2,2],dict(repeats=[2])),
 ("tile",[2],dict(repeats=[2,1])),("tile",[1,1,2],dict(repeats=[1,2,1])),
 ("repeat_interleave",[2],dict(repeats=2,axis=None,output_size=4)),
 ("repeat_interleave",[2,2],dict(repeats=3,axis=-1,output_size=6)),
 ("repeat_interleave",[1,2],dict(repeats=2,axis=0,output_size=None)),
 ("flip",[4],dict(axes=[-1])),("flip",[2,2],dict(axes=[0,1])),
 ("flip",[1,2,2],dict(axes=[-2])),("roll",[4],dict(shifts=[1],axes=None)),
 ("roll",[2,2],dict(shifts=[1,-1],axes=[0,-1])),
 ("roll",[1,2,2],dict(shifts=[1,2],axes=[-1,-1])),
]
def model(op,shape,attrs):
 return dict(format=FORMAT,id="transformation",inputs=[dict(name="x",shape=shape)],
  constants={},nodes=[dict(id="transform",op=op,inputs=["x"],attrs=attrs,outputs=["y"])],
  outputs=[dict(name="output",value="y")])

def torch_original(op,x,a):
 if op=="repeat":return x.repeat(tuple(a["repeats"]))
 if op=="tile":return torch.tile(x,tuple(a["repeats"]))
 if op=="repeat_interleave":return torch.repeat_interleave(x,a["repeats"],dim=a["axis"],output_size=a["output_size"])
 if op=="flip":return torch.flip(x,tuple(a["axes"]))
 return torch.roll(x,tuple(a["shifts"]),None if a["axes"] is None else tuple(a["axes"]))

class TransformMathTests(unittest.TestCase):
 def test_fifteen_math_contexts_and_immutable_source(self):
  for op,shape,a in CASES:
   with self.subTest(op=op,shape=shape):
    m=model(op,shape,a);before=copy.deepcopy(m);low,bound=decompose(m,resource_profile="general-tensors-v1")
    self.assertEqual(m,before);self.assertEqual((low,bound),decompose(m,resource_profile="general-tensors-v1"))
    self.assertEqual(bound["original_sha256"],digest(m));self.assertTrue(bound["no_dsl_answer"])
    self.assertEqual(set(bound["source_map"]),{n["id"] for n in low["nodes"]})
    for data in samples(m,16):
     expected=torch_original(op,torch.from_numpy(data["x"].copy()),a).numpy()
     np.testing.assert_array_equal(expected,evaluate_reference(m,data)["output"])
     np.testing.assert_array_equal(expected,core(low,data)["output"])
 def test_identity_and_singleton_copy_values(self):
  for op,shape,a in [("repeat",[1,2],dict(repeats=[1,1])),("tile",[2,2],dict(repeats=[])),
   ("repeat_interleave",[2,2],dict(repeats=1,axis=None,output_size=4)),
   ("flip",[1,2],dict(axes=[])),("flip",[1,2],dict(axes=[0])),
   ("roll",[2,2],dict(shifts=[0],axes=None))]:
   m=model(op,shape,a);low,_=decompose(m)
   data={"x":np.arange(np.prod(shape),dtype=np.float64).reshape(shape)}
   np.testing.assert_array_equal(evaluate_reference(m,data)["output"],core(low,data)["output"])
 def test_binary_repetition_preserves_eight_operand_gate(self):
  p=plan("repeat",[1],dict(repeats=[257]),resource_profile="general-tensors-v1")
  self.assertLess(len(p["steps"]),20);self.assertTrue(all(len(n["inputs"])<=8 for n in p["steps"]))
 def test_negative_large_and_repeated_roll_axes(self):
  m=model("roll",[2,3],dict(shifts=[-(10**8),3,5],axes=[0,1,-1]));low,_=decompose(m)
  data={"x":np.arange(6,dtype=np.float64).reshape(2,3)}
  np.testing.assert_array_equal(core(low,data)["output"],evaluate_reference(m,data)["output"])
 def test_rank_32_and_rank_overflow(self):
  p=plan("tile",[1]*31+[2],dict(repeats=[1]),resource_profile="general-tensors-v1")
  self.assertEqual(len(p["output_shape"]),32)
  with self.assertRaises(ValueError):plan("repeat",[2],dict(repeats=[1]*33),resource_profile="general-tensors-v1")
 def test_prior_format_refuses_new_operator(self):
  for version in range(1,5):
   m=model("repeat",[2],dict(repeats=[2]));m["format"]="poseidon-operator-graph-v"+str(version)
   with self.assertRaises(ValueError):decompose(m)
 def test_no_reference_uses_decomposition_plan(self):
  m=model("flip",[2,2],dict(axes=[0,1]));data={"x":np.arange(4,dtype=np.float64).reshape(2,2)}
  with patch("static_tensor_transform_plan.plan",side_effect=AssertionError("reference called lowering")):
   np.testing.assert_array_equal(evaluate_reference(m,data)["output"],data["x"][::-1,::-1])
 def test_rejections(self):
  bad=[("repeat",[2],dict(repeats=[0])),("repeat",[2],dict(repeats=[True])),
   ("repeat",[2,2],dict(repeats=[2])),("tile",[2],dict(repeats=[-1])),
   ("repeat_interleave",[2],dict(repeats=[1,2],axis=0,output_size=None)),
   ("repeat_interleave",[2],dict(repeats=True,axis=0,output_size=None)),
   ("repeat_interleave",[2],dict(repeats=2,axis=0,output_size=5)),
   ("flip",[2],dict(axes=[0,-1])),("flip",[2],dict(axes=[1])),
   ("roll",[2],dict(shifts=[],axes=[])),("roll",[2],dict(shifts=[1,2],axes=None)),
   ("roll",[2],dict(shifts=[True],axes=[0])),("roll",[2],dict(shifts=[1],axes=[1])),
   ("repeat",[2],dict(repeats=[65536])),("flip",[5000],dict(axes=[0]))]
  for op,shape,a in bad:
   with self.subTest(op=op,a=a):
    with self.assertRaises(ValueError):plan(op,shape,a,resource_profile="general-tensors-v1")
 def test_plain_operand_rejected(self):
  m=model("repeat",[2],dict(repeats=[2]));m["constants"]["public"]=[1,2];m["nodes"][0]["inputs"]=["public"]
  with self.assertRaises(ValueError):decompose(m)
 def test_node_expansion_budget(self):
  with self.assertRaises(ValueError):plan("repeat_interleave",[256],dict(repeats=2,axis=0,output_size=None),resource_profile="legacy-v1")

class TransformSourceTests(unittest.TestCase):
 def staged(self,module,shape):
  spec=[dict(name="x",shape=shape)]
  fx=trace(module,spec,resource_profile="general-tensors-v1",policy=ALIAS_POLICY,tensor_transform_source_policy=POLICY)
  normalized,record=normalize(fx,spec,resource_profile="general-tensors-v1")
  check(record,spec,resource_profile="general-tensors-v1")
  return fx,normalized,record
 def test_all_fifteen_source_normalizations(self):
  for op,shape,a in CASES:
   with self.subTest(op=op,shape=shape):
    class M(torch.nn.Module):
     def forward(self,x):return torch_original(op,x,a)
    fx,normalized,record=self.staged(M(),shape)
    x=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)/7
    torch.testing.assert_close(fx(x),normalized(x),rtol=0,atol=0)
    self.assertEqual(record["entries"][0]["source_node"],record["normalized_inventory"][-2]["name"])
 def test_method_and_keyword_forms(self):
  class M(torch.nn.Module):
   def forward(self,x):
    a=x.tile((2,1)).repeat_interleave(repeats=2,dim=-1,output_size=4)
    b=a.flip(dims=(0,)).roll(shifts=1,dims=1)
    return b
  _,normalized,_=self.staged(M(),[1,2])
  x=torch.tensor([[.7,-.8]],dtype=torch.float64)
  torch.testing.assert_close(normalized(x),M()(x),rtol=0,atol=0)
 def test_copy_storage_and_alias_mutation(self):
  class M(torch.nn.Module):
   def forward(self,x):
    y=x.repeat(1,1);y.add_(.25)
    return x,y
  _,normalized,_=self.staged(M(),[2,2])
  x=torch.tensor([[.1,-.3],[.5,-.7]],dtype=torch.float64);a,c=normalized(x.clone())
  torch.testing.assert_close(a,x,rtol=0,atol=0);torch.testing.assert_close(c,x+.25,rtol=0,atol=0)
 def test_flip_empty_and_roll_zero_copy(self):
  for which in ("flip","roll"):
   class M(torch.nn.Module):
    def forward(self,x):return x.flip(()) if which=="flip" else x.roll(0)
   _,normalized,_=self.staged(M(),[2,2]);x=torch.ones((2,2),dtype=torch.float64);y=normalized(x)
   self.assertNotEqual(x.data_ptr(),y.data_ptr())
 def test_transport_tamper_even_if_rebound(self):
  class M(torch.nn.Module):
   def forward(self,x):return x.roll(1)
  _,_,record=self.staged(M(),[4]);record["entries"][0]["attributes"]["shifts"]=[2]
  record["binding"]=digest({k:v for k,v in record.items() if k!="binding"})
  with self.assertRaises(ValueError):check(record,[dict(name="x",shape=[4])],resource_profile="general-tensors-v1")
 def test_prior_trace_cannot_enable_calls(self):
  class M(torch.nn.Module):
   def forward(self,x):return x.repeat(2)
  fx=trace(M(),[dict(name="x",shape=[2])],resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
  with self.assertRaises(ValueError):normalize(fx,[dict(name="x",shape=[2])],resource_profile="general-tensors-v1")
 def test_dynamic_count_is_rejected(self):
  class M(torch.nn.Module):
   def forward(self,x,y):return x.repeat_interleave(y)
  spec=[dict(name="x",shape=[2]),dict(name="y",shape=[2])]
  fx=trace(M(),spec,resource_profile="general-tensors-v1",policy=ALIAS_POLICY,tensor_transform_source_policy=POLICY)
  with self.assertRaises(ValueError):normalize(fx,spec,resource_profile="general-tensors-v1")
 def test_capability_catalog_has_explicit_graph_and_source_partition(self):
  from application_capabilities import describe
  c=describe()
  self.assertEqual(c["tensor_transform_decomposition_registry"]["graph_format"],FORMAT)
  self.assertEqual(set(c["tensor_transform_decomposition_registry"]["operators"]),{"repeat","tile","repeat_interleave","flip","roll"})
  for item in c["tensor_transform_decomposition_registry"]["operators"].values():
   self.assertEqual(item["encrypted_status"],"unverified");self.assertEqual(item["agent_status"],"unverified")
 def test_source_v30_routing_and_old_defaults(self):
  from pytorch_ordered_contract import shim,package_version,reference_version,allows_none_output,SOURCE_FORMATS
  from pytorch_static_tensor_transforms import SOURCE_FORMAT,PACKAGE_FORMAT,REFERENCE_FORMAT
  from pytorch_model_intake import ordered
  from pytorch_source_package_policy import selected
  from pytorch_spatial_policy import supports_spatial,supports_input_call
  spec=dict(format=SOURCE_FORMAT)
  self.assertIn(SOURCE_FORMAT,SOURCE_FORMATS);self.assertTrue(ordered(spec));self.assertTrue(selected(spec))
  self.assertTrue(supports_spatial(spec));self.assertTrue(supports_input_call(spec));self.assertTrue(allows_none_output(spec))
  self.assertEqual(package_version(spec),PACKAGE_FORMAT);self.assertEqual(reference_version(spec),REFERENCE_FORMAT)

if __name__=="__main__":unittest.main()
