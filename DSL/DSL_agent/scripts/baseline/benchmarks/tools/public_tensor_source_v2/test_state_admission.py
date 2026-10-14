"""Pure-Python admission tests, independent of Torch and the occupied Nix launcher."""
import copy,json,unittest
from public_state_binding import check_original,check_trace,direct_divisors,validate_state
def state(dtype="<f8",shape=None):
 return dict(dtype=dtype,shape=[3] if shape is None else shape,sha256="a"*64,normalized_float64_sha256="b"*64)
def graph(rhs="std"):
 def node(name,op,target,args=()):
  return dict(name=name,op=op,target=target,args=dict(kind="tuple",items=list(args)),
   kwargs=dict(kind="dict",items=[]))
 def ref(name):return dict(kind="fx_value",name=name)
 return [node("x","placeholder","x"),node("std","get_attr",rhs),
  node("div","call_function","_operator.truediv",[ref("x"),ref("std")])]
class StateAdmissionTests(unittest.TestCase):
 def test_mixed_dtype_manifest_preserved(self):
  s={"std":state(),"index":state("<i8"),"mask":state("|b1"),"counter":state("<i8",[])}
  declarations={n:dict(dtype=v["dtype"],shape=v["shape"])for n,v in s.items()}
  self.assertEqual(check_original(s,declarations),s)
 def test_int32_state_is_admissible(self):
  self.assertEqual(validate_state({"index":state("<i4")})["index"]["dtype"],"<i4")
 def test_float32_source_requires_other_explicit_policy(self):
  with self.assertRaisesRegex(ValueError,"exact raw type"):validate_state({"std":state("<f4")})
 def test_bool_metadata_cannot_be_changed_to_float64(self):
  with self.assertRaisesRegex(ValueError,"raw dtype/shape"):
   check_original({"mask":state("<f8")},{"mask":dict(dtype="|b1",shape=[3])})
 def test_integer_metadata_cannot_be_changed_to_float64(self):
  with self.assertRaisesRegex(ValueError,"raw dtype/shape"):
   check_original({"index":state("<f8")},{"index":dict(dtype="<i8",shape=[3])})
 def test_large_integer_raw_hash_is_separate_from_float64_identity(self):
  a=state("<i8");b=copy.deepcopy(a);b["sha256"]="c"*64
  with self.assertRaisesRegex(ValueError,"Trace state subset"):check_trace({"index":a},{"index":b},[])
 def test_unused_state_remains_authenticated(self):
  s={"std":state(),"counter":state("<i8")}
  self.assertEqual(check_trace(s,{"std":s["std"]},["counter"]),{"std":s["std"]})
 def test_unused_state_inventory_is_complete(self):
  with self.assertRaisesRegex(ValueError,"Trace state subset"):
   check_trace({"std":state(),"counter":state("<i8")},{"std":state()},[])
 def test_traced_state_cannot_add_values(self):
  with self.assertRaisesRegex(ValueError,"Trace state subset"):
   check_trace({"std":state()},{"std":state(),"new":state()},[])
 def test_traced_raw_dtype_cannot_change(self):
  with self.assertRaisesRegex(ValueError,"Trace state subset"):
   check_trace({"index":state("<i8")},{"index":state("<f8")},[])
 def test_complete_source_names_required(self):
  with self.assertRaisesRegex(ValueError,"state names"):
   check_original({"std":state()},{})
 def test_shape_binding_preserved(self):
  with self.assertRaisesRegex(ValueError,"raw dtype/shape"):
   check_original({"std":state()},{"std":dict(dtype="<f8",shape=[1,3])})
 def test_zero_axis_rejected(self):
  with self.assertRaisesRegex(ValueError,"static shape"):validate_state({"std":state(shape=[0])})
 def test_boolean_axis_is_not_integer_size(self):
  with self.assertRaisesRegex(ValueError,"static shape"):validate_state({"std":state(shape=[True])})
 def test_hash_is_strict_hex(self):
  s=state();s["sha256"]="g"*64
  with self.assertRaisesRegex(ValueError,"digest"):validate_state({"std":s})
 def test_state_unknown_field_rejected(self):
  s=state();s["fake"]="x"
  with self.assertRaisesRegex(ValueError,"exact raw type"):validate_state({"std":s})
 def test_divisor_inventory_ignores_nondivisor_public_types(self):
  s={"std":state(),"index":state("<i8"),"mask":state("|b1")}
  self.assertEqual(direct_divisors(graph(),s),{"std":s["std"]})
 def test_integer_divisor_rejected(self):
  with self.assertRaisesRegex(ValueError,"exact float64"):direct_divisors(graph(),{"std":state("<i8")})
 def test_boolean_divisor_rejected(self):
  with self.assertRaisesRegex(ValueError,"exact float64"):direct_divisors(graph(),{"std":state("|b1")})
 def test_encrypted_divisor_rejected(self):
  g=graph();g[1].update(op="placeholder",target="y")
  with self.assertRaisesRegex(ValueError,"Direct declared"):direct_divisors(g,{"std":state()})
 def test_encrypted_dividend_required(self):
  g=graph();g[0].update(op="get_attr",target="other")
  with self.assertRaisesRegex(ValueError,"Encrypted tensor dividend"):direct_divisors(g,{"std":state(),"other":state()})
 def test_derived_divisor_rejected(self):
  g=graph();g[1].update(op="call_function",target="torch.square")
  with self.assertRaisesRegex(ValueError,"Direct declared"):direct_divisors(g,{"std":state()})
 def test_divisor_state_required(self):
  with self.assertRaisesRegex(ValueError,"Direct declared"):direct_divisors(graph(),{"other":state()})
 def test_scalar_division_does_not_claim_tensor_coverage(self):
  g=graph();g[-1]["args"]["items"][1]=dict(kind="float",value=2.0)
  with self.assertRaisesRegex(ValueError,"requires a declared divisor"):direct_divisors(g,{"std":state()})
 def test_duplicate_source_names_rejected(self):
  g=graph();g[1]["name"]="x"
  with self.assertRaisesRegex(ValueError,"Unique public division source"):direct_divisors(g,{"std":state()})
 def test_forward_dependency_rejected(self):
  g=graph();g[0],g[2]=g[2],g[0]
  with self.assertRaisesRegex(ValueError,"FX dependency order"):direct_divisors(g,{"std":state()})
 def test_rounding_mode_rejected(self):
  g=graph();g[-1]["target"]="torch.div";g[-1]["kwargs"]["items"]=[["rounding_mode",dict(kind="str",value="floor")]]
  with self.assertRaisesRegex(ValueError,"rounding/out"):direct_divisors(g,{"std":state()})
 def test_duplicate_keyword_rejected(self):
  g=graph();g[-1]["target"]="torch.div";g[-1]["kwargs"]["items"]=[["out",dict(kind="none")],["out",dict(kind="none")]]
  with self.assertRaisesRegex(ValueError,"duplicate keyword"):direct_divisors(g,{"std":state()})
 def test_explicit_none_rounding_out_allowed(self):
  g=graph();g[-1]["target"]="torch.div";g[-1]["kwargs"]["items"]=[["out",dict(kind="none")],["rounding_mode",dict(kind="none")]]
  self.assertEqual(set(direct_divisors(g,{"std":state()})),{"std"})

class ApproximationAdmissionTests(unittest.TestCase):
 def test_registered_nonlinear_profiles_preserved(self):
  from pytorch_approximation import validate
  profiles={"relu":[.25,.5,.25],"gelu":[0.,.5,.3989422804014327],"sigmoid":[.5,.25,0.,-1./48.]}
  for function,coefficients in profiles.items():
   value=dict(format="poseidon-pytorch-approximations-v1",defaults={function:dict(method="polynomial",
    basis="power",domain=[-4.,4.],coefficients=coefficients)},modules=[],nodes=[])
   frozen=copy.deepcopy(value)
   self.assertEqual(validate(value),frozen);self.assertEqual(value,frozen)
 def test_approximation_domain_and_coefficients_still_checked(self):
  from pytorch_approximation import validate
  for invalid in (dict(domain=[4,-4],coefficients=[.5]),
   dict(domain=[-4,4],coefficients=[float("inf")])):
   value=dict(format="poseidon-pytorch-approximations-v1",defaults={"gelu":dict(method="polynomial",basis="power",**invalid)},
    modules=[],nodes=[])
   with self.assertRaises(ValueError):validate(value)
 def test_duplicate_node_approximation_selectors_rejected(self):
  from pytorch_approximation import validate
  selector=dict(name="gelu",function="gelu",profile=dict(method="polynomial",basis="power",
   domain=[-4,4],coefficients=[0,.5,.4]))
  value=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[selector,copy.deepcopy(selector)])
  with self.assertRaisesRegex(ValueError,"Conflicting approximation selectors"):validate(value)

class FloatNormalizationAdmissionTests(unittest.TestCase):
 def actual_state(self,dtype,values):
  import struct,hashlib
  formats={"<f4":"f","<f8":"d","<i8":"q","<i4":"i","|b1":"?"}
  raw=struct.pack("<"+formats[dtype]*len(values),*values)
  decoded=struct.unpack("<"+formats[dtype]*len(values),raw)
  normal=struct.pack("<"+"d"*len(values),*decoded)
  return dict(dtype=dtype,shape=[len(values)],sha256=hashlib.sha256(raw).hexdigest(),
   normalized_float64_sha256=hashlib.sha256(normal).hexdigest())
 def converted(self,s):
  from public_state_binding import float64_state
  return float64_state(s,float_policy="convert-float64-v1")
 def test_float32_requires_explicit_convert_policy(self):
  from public_state_binding import validate_state
  s={"std":self.actual_state("<f4",[.7,-1.3,2.1])}
  self.assertEqual(validate_state(s,float_policy="convert-float64-v1"),s)
  with self.assertRaisesRegex(ValueError,"exact raw type"):validate_state(s)
 def test_float32_conversion_matches_exact_promoted_value_bytes(self):
  import struct
  a=self.actual_state("<f4",[.7,-1.3,2.1])
  f32=struct.unpack("<fff",struct.pack("<fff",.7,-1.3,2.1))
  expected=self.actual_state("<f8",f32)
  self.assertEqual(self.converted({"std":a}),{"std":expected})
 def test_float32_conversion_preserves_the_raw_source_binding(self):
  s={"std":self.actual_state("<f4",[.7,-1.3,2.1])};original=copy.deepcopy(s)
  normalized=self.converted(s)
  self.assertEqual(s,original);self.assertNotEqual(s,normalized)
 def test_float64_state_remains_exact(self):
  s={"std":self.actual_state("<f8",[.7,-1.3,2.1])}
  self.assertEqual(self.converted(s),s)
 def test_integer_index_stays_integer(self):
  s={"index":self.actual_state("<i8",[2,0])}
  self.assertEqual(self.converted(s),s)
 def test_int32_index_stays_int32(self):
  s={"index":self.actual_state("<i4",[1,2])}
  self.assertEqual(self.converted(s),s)
 def test_boolean_mask_stays_boolean(self):
  s={"mask":self.actual_state("|b1",[True,False])}
  self.assertEqual(self.converted(s),s)
 def test_mixed_state_conversion_is_selective(self):
  s={"std":self.actual_state("<f4",[.7]),"weight":self.actual_state("<f8",[.25]),
   "index":self.actual_state("<i8",[2]),"mask":self.actual_state("|b1",[True])}
  n=self.converted(s)
  self.assertEqual(n["std"]["dtype"],"<f8")
  self.assertTrue(all(n[k]==s[k]for k in ("weight","index","mask")))
 def test_raw_float32_manifest_dtype_is_authenticated(self):
  from public_state_binding import check_original
  s={"std":self.actual_state("<f4",[.7])}
  check_original(s,{"std":dict(dtype="<f4",shape=[1])},float_policy="convert-float64-v1")
  with self.assertRaisesRegex(ValueError,"raw dtype/shape"):
   check_original(s,{"std":dict(dtype="<f8",shape=[1])},float_policy="convert-float64-v1")
 def test_conversion_rejects_changed_values(self):
  from public_state_binding import check_normalization
  s={"std":self.actual_state("<f4",[.7])};n=self.converted(s);n["std"]=self.actual_state("<f8",[.7])
  with self.assertRaisesRegex(ValueError,"normalization state"):
   check_normalization(s,n,float_policy="convert-float64-v1")
 def test_conversion_rejects_accidental_integer_to_float_cast(self):
  from public_state_binding import check_normalization
  s={"index":self.actual_state("<i8",[2])};n={"index":self.actual_state("<f8",[2.])}
  with self.assertRaisesRegex(ValueError,"normalization state"):
   check_normalization(s,n,float_policy="convert-float64-v1")
 def test_conversion_rejects_accidental_bool_to_float_cast(self):
  from public_state_binding import check_normalization
  s={"mask":self.actual_state("|b1",[True])};n={"mask":self.actual_state("<f8",[1.])}
  with self.assertRaisesRegex(ValueError,"normalization state"):
   check_normalization(s,n,float_policy="convert-float64-v1")
 def test_conversion_rejects_extra_state(self):
  from public_state_binding import check_normalization
  s={"std":self.actual_state("<f4",[.7])};n=self.converted(s);n["new"]=self.actual_state("<f8",[1.])
  with self.assertRaisesRegex(ValueError,"normalization state"):
   check_normalization(s,n,float_policy="convert-float64-v1")
 def test_conversion_rejects_missing_state(self):
  from public_state_binding import check_normalization
  s={"std":self.actual_state("<f4",[.7])}
  with self.assertRaisesRegex(ValueError,"normalization state"):
   check_normalization(s,{},float_policy="convert-float64-v1")
 def test_unsupported_float_policy_fails_closed(self):
  from public_state_binding import float64_state
  for policy in ("any",None,False,[]):
   with self.subTest(policy=policy),self.assertRaisesRegex(ValueError,"float policy"):
    float64_state({"std":self.actual_state("<f4",[.7])},float_policy=policy)
 def test_float16_stays_unsupported_under_convert_policy(self):
  s=state("<f2")
  with self.assertRaisesRegex(ValueError,"exact raw type"):self.converted({"std":s})
 def test_raw_float32_divisor_cannot_skip_conversion(self):
  with self.assertRaisesRegex(ValueError,"exact raw type"):
   direct_divisors(graph(),{"std":self.actual_state("<f4",[.7])})
 def test_normalized_float32_divisor_is_authenticated(self):
  s=self.converted({"std":self.actual_state("<f4",[.7])})
  self.assertEqual(direct_divisors(graph(),s),s)
 def test_negative_zero_sign_survives_conversion(self):
  negative=self.actual_state("<f4",[-0.]);positive=self.actual_state("<f4",[0.])
  self.assertNotEqual(self.converted({"std":negative})["std"]["sha256"],
   self.converted({"std":positive})["std"]["sha256"])
 def test_large_int64_raw_bytes_never_pass_through_float_conversion(self):
  a=self.actual_state("<i8",[2**53]);b=self.actual_state("<i8",[2**53+1])
  self.assertEqual(a["normalized_float64_sha256"],b["normalized_float64_sha256"])
  self.assertNotEqual(a["sha256"],b["sha256"])
  self.assertNotEqual(self.converted({"counter":a}),self.converted({"counter":b}))
