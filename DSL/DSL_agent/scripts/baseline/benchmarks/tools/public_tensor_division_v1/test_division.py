"""Numerical, provenance, broadcast, effect and rejection tests for an inactive prototype."""
import copy,hashlib,math,struct,unittest
from public_tensor_division import POLICY,PROFILE,plan,normalize,check,state_payload,metadata,source_identity
from benchmark_graph import digest
def raw(values):return b"".join(struct.pack("<d",v)for v in values)
def rebind(r):r["binding"]=digest({k:v for k,v in r.items()if k!="binding"});return r
class PlanTests(unittest.TestCase):
 def test_channel_broadcast(self):self.assertEqual(plan([1,3,2,2],[1,3,1,1],raw([.5,2.,-3.]))[0]["result_shape"],[1,3,2,2])
 def test_output_expansion(self):self.assertEqual(plan([2,1],[3],raw([1.,2.,4.]))[0]["result_shape"],[2,3])
 def test_scalar(self):self.assertEqual(plan([],[],raw([2.]))[0]["result_shape"],[])
 def test_rank32(self):self.assertEqual(len(plan([1]*31+[2],[2],raw([2.,3.]))[0]["result_shape"]),32)
 def test_ieee_reciprocal_bits(self):
  p,v=plan([4],[4],raw([3.,-7.,1/1024,1024]))
  self.assertEqual(v,raw([1/3.,-1/7.,1024,1/1024]))
  self.assertEqual(p["divisor_sha256"],hashlib.sha256(raw([3.,-7.,1/1024,1024])).hexdigest())
 def test_not_approximate(self):self.assertEqual(plan([2],[2],raw([2.,3.]))[0]["approximation"],"none")
 def test_invalid_shape(self):
  for s in [[0],[-1],[True],[1]*33,[65537]]:
   with self.subTest(shape=s),self.assertRaises(ValueError):plan(s,[1],raw([1.]))
 def test_invalid_broadcast(self):
  with self.assertRaisesRegex(ValueError,"broadcast"):plan([2,2],[3],raw([1.,2.,3.]))
 def test_inplace_expansion_rejected(self):
  with self.assertRaisesRegex(ValueError,"cannot expand"):plan([2,1],[3],raw([1.,2.,3.]),inplace=True)
 def test_boolean_inplace_rejected(self):
  with self.assertRaisesRegex(ValueError,"flag"):plan([2],[2],raw([1.,2.]),inplace=1)
 def test_wrong_payload_bytes(self):
  for data in [b"",raw([1.]),[1.,2.],bytearray(raw([1.,2.]))]:
   with self.subTest(kind=type(data).__name__),self.assertRaises(ValueError):plan([2],[2],data)
 def test_invalid_divisor(self):
  for value in [0.,-0.,float("nan"),float("inf"),-float("inf"),2048.,-2048.,.00001]:
   with self.subTest(value=value),self.assertRaises(ValueError):plan([2],[2],raw([1.,value]))
 def test_intermediate_elements_bounded(self):
  with self.assertRaisesRegex(ValueError,"tensor budget"):plan([512,1],[512],raw([1.]*512))
 def test_large_weight_plan_has_no_values(self):
  n=24000;p,v=plan([1,1],[n],raw([.5+i/n for i in range(n)]))
  self.assertEqual(p["elements"],n);self.assertEqual(len(v),8*n)
  self.assertLess(len(str(p)),1200);self.assertNotIn("values",p)
METRICS=[]
def module(mode,divisor):
 import torch
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.register_buffer("std",torch.tensor(divisor,dtype=torch.float64))
  def forward(self,x):
   if mode=="function":return torch.div(input=x,other=self.std,rounding_mode=None,out=None)
   if mode=="method":return x.true_divide(self.std)
   if mode=="square_residual":return (x/self.std).square()+x
   if mode=="snapshot":
    a=x.clone();x.div_(self.std);return {"snapshot":a,"updated":x}
   if mode=="transpose":
    a=x.transpose(0,1);a.divide_(self.std);return x,a
   if mode=="expand":return (x/self.std).sum(dim=-1)
   return x/self.std
 return Model().eval()
def capture(m,shapes,*,named=None):
 import torch
 from torch.fx.passes.shape_prop import ShapeProp
 from pytorch_fused_matrix_normalization import _state
 graph=torch.fx.symbolic_trace(m)
 ShapeProp(graph).propagate(*[torch.zeros(s,dtype=torch.float64)for s in shapes])
 for n in graph.graph.nodes:
  value=n.meta.get("tensor_meta")
  if hasattr(value,"shape"):
   n.meta["poseidon_static_tensor"]=dict(policy=metadata([])["policy"],shape=list(value.shape),dtype=str(value.dtype).removeprefix("torch."))
 graph.meta["public_tensor_division_policy"]=POLICY
 names=named or ["x","y"][:len(shapes)]
 signature=[dict(name=n,shape=s)for n,s in zip(names,shapes)]
 states=_state(graph);declared={k:v for k,v in states.items()if k=="std"}
 return graph,signature,declared
def flat(value):
 import numpy as np,torch
 if isinstance(value,torch.Tensor):return[value.detach().cpu().numpy()]
 if type(value)in(tuple,list):return[a for v in value for a in flat(v)]
 if type(value)is dict:return[a for v in value.values()for a in flat(v)]
 if isinstance(value,np.ndarray):return[value]
 raise ValueError("test output type")
def scalar_math_divide(x,d):
 """Reference uses scalar division, independently of the rewrite's reciprocal/multiply."""
 import numpy as np
 a,b=np.broadcast_arrays(x,d);out=np.empty(a.shape,dtype=np.float64)
 for idx in np.ndindex(a.shape):out[idx]=float(a[idx])/float(b[idx])
 return out
def reference(mode,x,d):
 import numpy as np
 if mode=="transpose":
  a=scalar_math_divide(x.T,d);return a.T,a
 divided=scalar_math_divide(x,d)
 if mode=="square_residual":return divided*divided+x
 if mode=="snapshot":return {"snapshot":x.copy(),"updated":divided}
 if mode=="expand":return divided.sum(axis=-1)
 return divided
class NumericalTests(unittest.TestCase):
 def exercise(self,mode,shape,divisor):
  import numpy as np,torch
  graph,inputs,declared=capture(module(mode,divisor),[shape])
  original=str(graph.graph);before={n:v.clone()for n,v in graph.state_dict().items()}
  revised,receipt=normalize(graph,inputs,declared)
  payload={n:state_payload(graph,n)for n in declared}
  from pytorch_fused_matrix_normalization import inventory,_state
  expected_identity=source_identity(inputs,inventory(graph),_state(graph),str(graph.graph))
  check(receipt,inputs,declared,payload,expected_source_identity=expected_identity)
  count=math.prod(shape);rng=np.random.default_rng(20261010)
  samples=[np.zeros(shape,dtype=np.float64),np.linspace(-.3,.5,count).reshape(shape),
   rng.uniform(-.25,.25,shape),np.where(np.arange(count)%2,.4,-.4).reshape(shape)]
  maximum=0.;values=0
  for x in samples:
   a=flat(graph(torch.from_numpy(x.copy())));b=flat(revised(torch.from_numpy(x.copy())))
   expected=flat(reference(mode,x,np.asarray(divisor,dtype=np.float64)))
   self.assertEqual(len(a),len(b));self.assertEqual(len(a),len(expected))
   for one,two,ref in zip(a,b,expected):
    np.testing.assert_allclose(one,ref,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(two,ref,rtol=2e-12,atol=2e-12)
    maximum=max(maximum,float(np.max(np.abs(two-ref))));values+=ref.size
  self.assertEqual(str(graph.graph),original)
  self.assertTrue(all(torch.equal(v,graph.state_dict()[n])and torch.equal(v,revised.state_dict()[n])for n,v in before.items()))
  METRICS.append(dict(mode=mode,input_shape=shape,divisor_shape=list(np.shape(divisor)),input_groups=4,
   output_values=values,max_abs_error=maximum,source_preserved=True,public_state_preserved=True))
 def test_vector(self):self.exercise("operator",[2,3],[.5,-2.,3.])
 def test_channels(self):self.exercise("operator",[1,3,2,2],[[[[.5]],[[-2.]],[[3.]]]])
 def test_dense(self):self.exercise("operator",[2,2],[[.5,-2.],[3.,.25]])
 def test_scalar_state(self):self.exercise("operator",[2,2],-3.)
 def test_function_keywords(self):self.exercise("function",[2,3],[.5,2.,3.])
 def test_method_alias(self):self.exercise("method",[2,3],[.5,2.,3.])
 def test_residual(self):self.exercise("square_residual",[2,3],[.5,2.,3.])
 def test_snapshot_and_mutation(self):self.exercise("snapshot",[2,3],[.5,-2.,3.])
 def test_transpose_and_shared_write(self):self.exercise("transpose",[2,3],[.5,-2.])
 def test_broadcast_expand_reduction(self):self.exercise("expand",[2,1],[.5,2.,3.])
 def test_large_nonrepeated_weights(self):self.exercise("expand",[1,1],[.5+i/24000 for i in range(24000)])
 def test_multi_input_cross_branches(self):
  import torch
  class Two(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer("std",torch.tensor([.5,2.,3.],dtype=torch.float64))
   def forward(self,x,y):return x/self.std+y.div(self.std),x/y.square().add(1.)
  graph,inputs,declared=capture(Two(),[[2,3],[2,3]])
  with self.assertRaisesRegex(ValueError,"Direct declared"):normalize(graph,inputs,declared)
 def test_two_inputs_public_division_fusion(self):
  import torch,numpy as np
  class Two(torch.nn.Module):
   def __init__(self):super().__init__();self.register_buffer("std",torch.tensor([.5,2.,3.],dtype=torch.float64))
   def forward(self,x,y):return x/self.std+y.div(self.std)
  graph,inputs,declared=capture(Two(),[[2,3],[2,3]])
  revised,r=normalize(graph,inputs,declared)
  for seed in range(4):
   rng=np.random.default_rng(seed);x=rng.uniform(-.3,.3,(2,3));y=rng.uniform(-.3,.3,(2,3))
   d=np.array([.5,2.,3.]);expected=scalar_math_divide(x,d)+scalar_math_divide(y,d)
   np.testing.assert_allclose(revised(torch.from_numpy(x),torch.from_numpy(y)).detach().numpy(),expected,rtol=2e-12,atol=2e-12)
  self.assertEqual(len(r["entries"]),2)
 def test_production_v21_still_rejects_tensor_divisor(self):
  from pytorch_static_trace import trace,ALIAS_POLICY
  from pytorch_exact_arithmetic import POLICY as OLD
  with self.assertRaisesRegex(ValueError,"scalar|shape"):
   trace(module("operator",[.5,2.,3.]),[dict(name="x",shape=[2,3])],
    resource_profile=PROFILE,policy=ALIAS_POLICY,exact_arithmetic_policy=OLD)
class IntegrityTests(unittest.TestCase):
 def receipt(self):
  graph,inputs,declared=capture(module("operator",[.5,2.,3.]),[[2,3]])
  normalized,record=normalize(graph,inputs,declared)
  from pytorch_fused_matrix_normalization import inventory,_state
  expected=source_identity(inputs,inventory(graph),_state(graph),str(graph.graph))
  return record,inputs,declared,{n:state_payload(graph,n)for n in declared},expected
 def rejected(self,change):
  r,i,d,p,expected=self.receipt();change(r);rebind(r)
  with self.assertRaises(ValueError):check(r,i,d,p,expected_source_identity=expected)
 def test_reciprocal_coefficient_hash(self):self.rejected(lambda r:r["entries"][0]["plan"].update(reciprocal_sha256="0"*64))
 def test_changed_shape(self):self.rejected(lambda r:r["entries"][0]["plan"].update(result_shape=[6]))
 def test_missing_rewrite(self):self.rejected(lambda r:r["entries"].clear())
 def test_divisor_origin_substitution(self):self.rejected(lambda r:r["entries"][0].update(state_name="other"))
 def test_normalized_operand_substitution(self):
  self.rejected(lambda r:r["normalized_inventory"][-2]["args"]["items"].reverse())
 def test_derived_weight_substitution(self):
  self.rejected(lambda r:r["normalized_state"][r["entries"][0]["reciprocal_state_name"]].update(sha256="0"*64))
 def test_original_state_substitution(self):self.rejected(lambda r:r["public_state"]["std"].update(sha256="0"*64))
 def test_source_hash(self):self.rejected(lambda r:r.update(source_fx_text="changed"))
 def test_normalized_hash(self):self.rejected(lambda r:r.update(normalized_fx_text="changed"))
 def test_version_substitution(self):self.rejected(lambda r:r.update(format="production"))
 def test_activation_claim(self):self.rejected(lambda r:r.update(production_activation=True))
 def test_implementation_hash(self):self.rejected(lambda r:r.update(implementation_sha256="0"*64))
 def test_duplicate_source_names(self):
  self.rejected(lambda r:r["source_inventory"][-1].update(name=r["source_inventory"][0]["name"]))
 def test_payload_substitution(self):
  r,i,d,p,expected=self.receipt();p["std"]=raw([.5,2.,4.])
  with self.assertRaisesRegex(ValueError,"payload substitution"):check(r,i,d,p,expected_source_identity=expected)
 def test_unknown_public_state(self):
  graph,i,d=capture(module("operator",[.5,2.,3.]),[[2,3]]);d["std"]["sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"state binding"):normalize(graph,i,d)
 def test_explicit_policy_required(self):
  graph,i,d=capture(module("operator",[.5,2.,3.]),[[2,3]]);graph.meta.clear()
  with self.assertRaisesRegex(ValueError,"isolated division policy"):normalize(graph,i,d)
 def test_wrong_signature(self):
  graph,i,d=capture(module("operator",[.5,2.,3.]),[[2,3]]);i[0]["shape"]=[6]
  with self.assertRaisesRegex(ValueError,"signature"):normalize(graph,i,d)
 def test_float32_state_rejected(self):
  graph,i,d=capture(module("operator",[.5,2.,3.]),[[2,3]]);graph.std=graph.std.float()
  with self.assertRaisesRegex(ValueError,"state binding"):normalize(graph,i,d)
class SourceIdentityTests(unittest.TestCase):
 receipt=IntegrityTests.receipt
 rejected=IntegrityTests.rejected
 def test_original_input_shape_substitution(self):
  self.rejected(lambda r:r["source_inventory"][0]["tensor"].update(shape=[3,2]))
 def test_external_identity_required(self):
  r,i,d,p,expected=self.receipt()
  with self.assertRaises(TypeError):check(r,i,d,p)
 def test_external_identity_mismatch(self):
  r,i,d,p,expected=self.receipt()
  with self.assertRaisesRegex(ValueError,"Authenticated source"):
   check(r,i,d,p,expected_source_identity="0"*64)
 def test_redigested_source_identity_cannot_replace_authority(self):
  r,i,d,p,expected=self.receipt();r["source_fx_text"]+=" unauthorized"
  r["source_fx_sha256"]=hashlib.sha256(r["source_fx_text"].encode()).hexdigest()
  r["source_identity"]=source_identity(i,r["source_inventory"],r["public_state"],r["source_fx_text"])
  rebind(r)
  with self.assertRaisesRegex(ValueError,"Authenticated source"):check(r,i,d,p,expected_source_identity=expected)
 def test_duplicate_source_input_rejected(self):
  self.rejected(lambda r:r["source_inventory"].append(copy.deepcopy(r["source_inventory"][0])))
 def test_missing_output_source_rejected(self):
  self.rejected(lambda r:r["source_inventory"].pop())
if __name__=="__main__":unittest.main(verbosity=2)
