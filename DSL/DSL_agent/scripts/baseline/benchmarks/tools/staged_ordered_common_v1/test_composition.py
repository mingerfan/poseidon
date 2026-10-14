"""Compose registered common calls with ordered writes, using independent references."""
import copy,json,operator,unittest,math,hashlib,io
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from benchmark_graph import digest
from pytorch_static_trace import trace,ALIAS_POLICY,COMMON_POLICY
from pytorch_arithmetic_normalization import normalize_common,check_common_record,ORDERED_COMMON_POLICY
from pytorch_ordered_capture import prepare_module,COMMON_FORMAT
from pytorch_ordered_reference import validate
from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
PROFILE="general-tensors-v1"
APPROX=dict(format="poseidon-pytorch-approximations-v1",
 defaults=dict(exp=dict(method="taylor",domain=[-8.,8.],degree=2),
  reciprocal=dict(method="goldschmidt",domain=[.5,100.],iterations=1)),
 modules=[],nodes=[])
# New SDPA fixture profile, independent of the retained narrow-domain fixtures.
# Probes use [-1,1]; |QK/sqrt(2)| <= sqrt(2)*1.125 and additive mask <= .2.
# At least one allowed key per row: exp(-limit)>.125 and 2*exp(limit)<16.
VALID_APPROX=copy.deepcopy(APPROX)
VALID_APPROX["defaults"]["reciprocal"]["domain"]=[.125,16.]
EMPTY_APPROX=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])
REFERENCES=[]
def prepared(model,signatures,names,*,approximations=None,**kwargs):
 return prepare_module(model.double().eval(),signatures,names,approximations=EMPTY_APPROX if approximations is None else approximations,
  resource_profile=PROFILE,approximation_policy="node-module-default-v1",
  matrix_policy="static-tensor-matmul-v1",functional_policy=ADAPTIVE_FUNCTION_POLICY,**kwargs)
def check_case(case,model,signatures,names,*,approximations=None):
 value=prepared(model,signatures,names,common_calls=True,approximations=approximations)
 high,low,binding,_=value
 alias=binding["pytorch_capture"]["ordered_alias_capture"]
 adapter=alias["capture_adapter"]
 assert adapter["format"]==COMMON_FORMAT
 record=adapter["common_normalization"]
 check_common_record(record,alias["capture_emissions"],adapter["arithmetic"])
 assert record["format"]==ORDERED_COMMON_POLICY and record["ordered_effects"] is True
 reference=validate(value,signatures,names,resource_profile=PROFILE,count=16)
 REFERENCES.append(dict(case=case,graph_sha256=digest(high),lowered_sha256=digest(low),
  common_rules=sorted({e["rule"] for e in record["entries"]}),reference=reference,
  approximation_configuration_sha256=digest(EMPTY_APPROX if approximations is None else approximations)))
 return value
class Weighted(torch.nn.Module):
 def __init__(self,kind,style):
  super().__init__();self.kind,self.style=kind,style
  if kind=="public_tensor":self.register_buffer("bias",torch.tensor([.125,-.25,.75],dtype=torch.float64))
 def forward(self,x,y):
  previous=x.clone()
  other=y if self.kind=="cipher" else self.bias if self.kind=="public_tensor" else .25
  if self.style==0:weighted=torch.add(x,other,alpha=1.25)
  elif self.style==1:weighted=x.sub(other,alpha=-.5)
  else:weighted=torch.subtract(x,other,alpha=2.)
  x.view(-1).add_(.125)
  return weighted,x,previous
class WeightedComposition(Weighted):
 def __init__(self,kind,context):
  super().__init__(kind,0);self.context=context
 def forward(self,x,y):
  previous=x.clone()
  other=y if self.kind=="cipher" else self.bias if self.kind=="public_tensor" else .25
  if self.context=="reduction":
   x.narrow(-1,0,2).mul_(.75)
   weighted=x.add(other,alpha=1.25)
   result=weighted.mean(dim=-1,keepdim=True)
  else:
   weighted=torch.add(x,other,alpha=1.25)
   result=weighted*x+previous
  x.view(-1).add_(.125)
  return result,x,previous
class InferenceDropout(torch.nn.Module):
 def __init__(self,kind,style,as_module=False):
  super().__init__();self.kind,self.style,self.as_module=kind,style,as_module
  if as_module:self.layer=getattr(torch.nn,kind)(p=.3,inplace=False)
 def forward(self,x):
  previous=x.clone()
  target=x.transpose(-1,-2) if self.style==1 else x
  if self.as_module:y=self.layer(target)
  else:y=getattr(F,self.kind.lower())(target,p=.3,training=False,inplace=False)
  if self.style==2:y=y.clone()
  x.add_(.25)
  return y,x,previous
class Attention(torch.nn.Module):
 def __init__(self,mask,style):
  super().__init__();self.mask_kind,self.style=mask,style
  if mask=="boolean":self.register_buffer("mask",torch.tensor([[True,False],[True,True]]))
  if mask=="additive":self.register_buffer("mask",torch.tensor([[0.,-.2],[.1,0.]],dtype=torch.float64))
 def forward(self,q,k,v):
  previous=v.clone()
  if self.style==0:q.view(-1).add_(.125)
  elif self.style==1:k.transpose(-1,-2).mul_(.75)
  else:v.reshape(-1).add_(.125)
  value=F.dropout(v,p=.2,training=False)
  mask=self.mask if self.mask_kind in ("boolean","additive") else None
  y=F.scaled_dot_product_attention(q,k,value,attn_mask=mask,dropout_p=0.,is_causal=self.mask_kind=="causal")
  v.add_(.05)
  return dict(attention=y,current=v,previous=previous)
class EffectWitness(torch.nn.Module):
 def forward(self,x,y):
  alias=F.dropout(x,p=.2,training=False)
  x.add_(y)
  return alias
class CompositionTests(unittest.TestCase):
 def test_dead_mutation_retained_and_legacy_counterexample(self):
  signature=[dict(name=n,shape=[2,3]) for n in ("x","y")]
  source=EffectWitness().double().eval();raw=trace(source,signature,resource_profile=PROFILE,policy=ALIAS_POLICY)
  naive,_=normalize_common(raw,signature,resource_profile=PROFILE)
  fixed,record=normalize_common(raw,signature,resource_profile=PROFILE,ordered_effects=True)
  x=torch.arange(6,dtype=torch.float64).reshape(2,3)/4;y=torch.ones_like(x)*.375
  expected=source(x.clone(),y.clone())
  self.assertFalse(torch.equal(naive(x.clone(),y.clone()),expected))
  torch.testing.assert_close(fixed(x.clone(),y.clone()),expected,atol=0,rtol=0)
  self.assertTrue(any(n.op=="call_method" and n.target=="add_" for n in fixed.graph.nodes))
  self.assertEqual(record["dead_code_policy"],"retain-source-effects-and-order")
  check_case("unused_add_write_witness",source,signature,["result"])
 def test_public_scalar_alpha_three_contexts(self):
  for style in range(3):
   with self.subTest(style=style):
    check_case("alpha_scalar_"+str(style),Weighted("public_scalar",style),
     [dict(name="x",shape=[2,3]),dict(name="y",shape=[3])],["weighted","current","previous"])
 def test_public_tensor_alpha_three_contexts(self):
  for style in range(3):
   with self.subTest(style=style):
    check_case("alpha_public_tensor_"+str(style),Weighted("public_tensor",style),
     [dict(name="x",shape=[2,3]),dict(name="y",shape=[3])],["weighted","current","previous"])
 def test_cipher_alpha_three_contexts(self):
  for style in range(3):
   with self.subTest(style=style):
    check_case("alpha_cipher_"+str(style),Weighted("cipher",style),
     [dict(name="x",shape=[2,3]),dict(name="y",shape=[3])],["weighted","current","previous"])
 def test_functional_dropout_three_contexts_each(self):
  for kind,shape in (("Dropout",[2,3]),("Dropout1d",[1,2,3]),("Dropout2d",[1,1,2,3]),
    ("Dropout3d",[1,1,2,2,2]),("Alpha_Dropout",[2,3]),("Feature_Alpha_Dropout",[2,3])):
   for style in range(3):
    with self.subTest(kind=kind,style=style):
     check_case("functional_"+kind+"_"+str(style),InferenceDropout(kind,style),
      [dict(name="x",shape=shape)],["result","current","previous"])
 def test_module_dropout_three_contexts_each(self):
  for kind,shape in (("Dropout",[2,3]),("Dropout1d",[1,2,3]),("Dropout2d",[1,1,2,3]),
    ("Dropout3d",[1,1,2,2,2]),("AlphaDropout",[2,3]),("FeatureAlphaDropout",[2,3])):
   for style in range(3):
    with self.subTest(kind=kind,style=style):
     check_case("module_"+kind+"_"+str(style),InferenceDropout(kind,style,as_module=True),
      [dict(name="x",shape=shape)],["result","current","previous"])
 def test_sdpa_no_mask_three_contexts(self):self.sdpa("none")
 def test_sdpa_boolean_mask_three_contexts(self):self.sdpa("boolean")
 def test_sdpa_additive_mask_three_contexts(self):self.sdpa("additive")
 def test_sdpa_causal_mask_three_contexts(self):self.sdpa("causal")
 def sdpa(self,kind):
  for style,shape in enumerate(([2,2],[1,2,2],[1,2,2,2])):
   with self.subTest(kind=kind,style=style):
    check_case("sdpa_"+kind+"_bounded_domain_"+str(style),Attention(kind,style),
     [dict(name=n,shape=shape) for n in ("q","k","v")],["attention","current","previous"],
     approximations=VALID_APPROX)
 def test_common_calls_opt_in_and_old_capture_default(self):
  model=Weighted("cipher",0);sig=[dict(name="x",shape=[2,3]),dict(name="y",shape=[3])]
  with self.assertRaises(ValueError):prepared(model,sig,["weighted","current","previous"])
  with self.assertRaisesRegex(ValueError,"Explicit ordered common calls"):
   prepared(model,sig,["weighted","current","previous"],common_calls="enabled")
 def test_wrong_source_policy_rejected(self):
  sig=[dict(name=n,shape=[2,3]) for n in ("x","y")]
  raw=trace(EffectWitness(),sig,resource_profile=PROFILE,policy=COMMON_POLICY)
  with self.assertRaisesRegex(ValueError,"Ordered common source policy"):
   normalize_common(raw,sig,resource_profile=PROFILE,ordered_effects=True)
 def test_ordered_record_policy_tamper_rejected(self):
  value=check_case("record_tamper_context",EffectWitness(),
   [dict(name=n,shape=[2,3]) for n in ("x","y")],["result"])
  alias=value[2]["pytorch_capture"]["ordered_alias_capture"];adapter=alias["capture_adapter"]
  record=copy.deepcopy(adapter["common_normalization"]);record["ordered_effects"]=False
  record.pop("binding_sha256");record["binding_sha256"]=digest(record)
  with self.assertRaisesRegex(ValueError,"Ordered common capture policy"):
   check_common_record(record,alias["capture_emissions"],adapter["arithmetic"])
 def test_generated_public_constant_tamper_rejected(self):
  value=check_case("generated_constant_tamper_context",Weighted("public_tensor",0),
   [dict(name="x",shape=[2,3]),dict(name="y",shape=[3])],["weighted","current","previous"])
  alias=value[2]["pytorch_capture"]["ordered_alias_capture"];adapter=alias["capture_adapter"]
  record=copy.deepcopy(adapter["common_normalization"]);record["generated_constants"][0]["float64_sha256"]="0"*64
  record.pop("binding_sha256");record["binding_sha256"]=digest(record)
  with self.assertRaisesRegex(ValueError,"Common generated mask binding"):
   check_common_record(record,alias["capture_emissions"],adapter["arithmetic"])
 def test_training_dropout_and_missing_approximation_rejected(self):
  class Training(torch.nn.Module):
   def forward(self,x):return F.dropout(x,training=True)
  with self.assertRaises(ValueError):prepared(Training(),[dict(name="x",shape=[2,3])],["result"],common_calls=True)
  with self.assertRaises(ValueError):
   prepare_module(Attention("none",0),[dict(name=n,shape=[2,2]) for n in ("q","k","v")],
    ["attention","current","previous"],resource_profile=PROFILE,common_calls=True,
    approximation_policy="node-module-default-v1",matrix_policy="static-tensor-matmul-v1",
    functional_policy=ADAPTIVE_FUNCTION_POLICY)
 def test_original_causal_domain_is_still_rejected(self):
  for style,shape in enumerate(([2,2],[1,2,2],[1,2,2,2])):
   with self.subTest(style=style):
    signatures=[dict(name=n,shape=shape) for n in ("q","k","v")]
    value=prepared(Attention("causal",style),signatures,["attention","current","previous"],
     common_calls=True,approximations=APPROX)
    with self.assertRaisesRegex(ValueError,"outside explicit approximation domain"):
     validate(value,signatures,["attention","current","previous"],resource_profile=PROFILE,count=16)
 def test_new_declared_domain_has_input_bound_basis(self):
  limit=math.sqrt(2)*1.125+.2
  self.assertGreater(math.exp(-limit),.125)
  self.assertLess(2*math.exp(limit),16.)
  self.assertLess(2*(1+limit+.5*limit**2),16.)
 def test_unmatched_approximation_default_remains_rejected(self):
  with self.assertRaisesRegex(ValueError,"Unmatched approximation default"):
   prepared(EffectWitness(),[dict(name=n,shape=[2,3]) for n in ("x","y")],["result"],
    common_calls=True,approximations=APPROX)
 def test_alpha_residual_and_reduction_compositions(self):
  for kind in ("public_scalar","public_tensor","cipher"):
   for context in ("residual","reduction"):
    with self.subTest(kind=kind,context=context):
     check_case("alpha_"+kind+"_"+context,WeightedComposition(kind,context),
      [dict(name="x",shape=[2,3]),dict(name="y",shape=[3])],["result","current","previous"])


def state_records(module):
 arrays={k:v.detach().cpu().numpy().copy() for k,v in module.state_dict().items()}
 records={}
 for i,(name,value) in enumerate(arrays.items()):
  stream=io.BytesIO();np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
  records[name]=dict(file="state_"+str(i)+".npy",sha256=hashlib.sha256(raw).hexdigest(),
   bytes=len(raw),shape=list(value.shape),dtype=value.dtype.str)
 from pytorch_weight_package import binding
 return records,binding(arrays,records)

def check_state(value,records,weights):
 from pytorch_common_constant_provenance import check
 alias=value[2]["pytorch_capture"]["ordered_alias_capture"]
 adapter=alias["capture_adapter"];common=adapter["common_normalization"]
 check_common_record(common,alias["capture_emissions"],adapter["arithmetic"])
 return check(alias["capture_emissions"],common,alias["capture_emission_state_constant_sources"],
              records,weights,resource_profile=PROFILE)

class ConstantProvenanceTests(unittest.TestCase):
 def build(self,kind):
  if kind=="alpha":
   model=Weighted("public_tensor",0)
   signature=[dict(name="x",shape=[2,3]),dict(name="y",shape=[3])]
   names=["weighted","current","previous"];approx=EMPTY_APPROX
  else:
   model=Attention(kind,0)
   signature=[dict(name=n,shape=[2,2]) for n in ("q","k","v")]
   names=["attention","current","previous"];approx=VALID_APPROX
  records,weights=state_records(model)
  value=prepared(model,signature,names,common_calls=True,approximations=approx)
  check_state(value,records,weights)
  return value,records,weights
 def test_alpha_boolean_and_causal_original_derivations(self):
  for kind in ("alpha","boolean","causal"):
   with self.subTest(kind=kind):
    value,records,weights=self.build(kind)
    receipt=check_state(value,records,weights)
    self.assertEqual(receipt["generated_state"],1)
 def test_wrong_public_weight_hash_is_rejected(self):
  value,records,weights=self.build("alpha")
  changed=copy.deepcopy(weights)
  changed["normalized_float64_sha256"]["bias"]="0"*64
  with self.assertRaisesRegex(ValueError,"loaded state"):check_state(value,records,changed)
 def test_rehashed_wrong_alpha_constant_does_not_pass(self):
  value,records,weights=self.build("alpha");changed=copy.deepcopy(value)
  alias=changed[2]["pytorch_capture"]["ordered_alias_capture"]
  common=alias["capture_adapter"]["common_normalization"]
  item=common["generated_constants"][0]
  array=np.asarray(alias["capture_emissions"]["constants"][item["captured_constant"]],dtype="<f8")+.25
  alias["capture_emissions"]["constants"][item["captured_constant"]]=array.tolist()
  item["float64_sha256"]=hashlib.sha256(array.tobytes()).hexdigest()
  common["captured_graph_sha256"]=digest(alias["capture_emissions"])
  common.pop("binding_sha256");common["binding_sha256"]=digest(common)
  with self.assertRaisesRegex(ValueError,"original derivation"):check_state(changed,records,weights)
 def test_alpha_selector_and_coefficient_tamper_rejected(self):
  for field,new in (("source","unknown"),("alpha",2.)):
   with self.subTest(field=field):
    value,records,weights=self.build("alpha");changed=copy.deepcopy(value)
    common=changed[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]["common_normalization"]
    common["generated_constants"][0]["derivation"][field]=new
    common.pop("binding_sha256");common["binding_sha256"]=digest(common)
    with self.assertRaisesRegex(ValueError,"Alpha derivation"):check_state(changed,records,weights)
 def test_boolean_mask_cannot_be_rebound_to_altered_weights(self):
  value,records,weights=self.build("boolean");changed=copy.deepcopy(weights)
  changed["normalized_float64_sha256"]["mask"]="0"*64
  with self.assertRaisesRegex(ValueError,"original state"):check_state(value,records,changed)
 def test_causal_shape_rehash_tamper_rejected(self):
  value,records,weights=self.build("causal");changed=copy.deepcopy(value)
  common=changed[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]["common_normalization"]
  common["generated_constants"][0]["derivation"]["query_length"]=3
  common.pop("binding_sha256");common["binding_sha256"]=digest(common)
  with self.assertRaisesRegex(ValueError,"Causal mask derivation"):check_state(changed,records,weights)
 def test_undeclared_state_rejected(self):
  value,records,weights=self.build("alpha");changed=copy.deepcopy(value)
  alias=changed[2]["pytorch_capture"]["ordered_alias_capture"]
  source=next(n for n,s in alias["capture_emission_state_constant_sources"].items() if s=="bias")
  alias["capture_emission_state_constant_sources"][source]="undeclared"
  with self.assertRaisesRegex(ValueError,"Undeclared emission"):check_state(changed,records,weights)
