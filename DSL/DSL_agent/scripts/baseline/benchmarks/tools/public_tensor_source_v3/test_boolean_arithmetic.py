"""Boundaries and tamper rejection for exact public boolean multiplication."""
import copy,hashlib,struct,unittest
from pathlib import Path
import public_boolean_arithmetic as m
from benchmark_graph import digest
def value(name):return dict(kind="fx_value",name=name)
def node(name,op,target,items=(),dtype="float64",shape=(2,3)):
 return dict(name=name,op=op,target=target,args=dict(kind="tuple",items=list(items)),
  kwargs=dict(kind="dict",items=[]),tensor=dict(policy=m.metadata([])["policy"],shape=list(shape),dtype=dtype))
def fixture():
 payload=bytes([1,0,1]);numeric=struct.pack("<3d",1.,0.,1.)
 state={"mask":dict(shape=[3],dtype="|b1",sha256=m.sha(payload),normalized_float64_sha256=m.sha(numeric))}
 source=[node("x","placeholder","x"),node("mask","get_attr","mask",dtype="bool",shape=(3,)),
  node("product","call_function","_operator.mul",[value("x"),value("mask")]),
  node("output","output","output",[value("product")])]
 entry=dict(source_node="product",input="x",mask_node="mask",mask_state_name="mask",
  derived_state_name="_poseidon_public_arithmetic_mask_product",operand_order=1,inplace=False,
  plan=m.mask_plan([2,3],[3],payload)[0],derived_fx_node="numeric_mask")
 normalized=[source[0],node("numeric_mask","get_attr",entry["derived_state_name"],shape=(3,)),
  node("product","call_function","_operator.mul",[value("x"),value("numeric_mask")]),source[3]]
 source_text="authenticated static source";target_text="explicit bool-to-zero-or-one public multiplication"
 record=dict(format=m.POLICY,source_inventory=source,normalized_inventory=copy.deepcopy(normalized),
  source_fx_text=source_text,source_fx_sha256=m.sha(source_text.encode()),normalized_fx_text=target_text,
  normalized_fx_sha256=m.sha(target_text.encode()),public_state=state,
  normalized_state=copy.deepcopy({**state,entry["derived_state_name"]:dict(shape=[3],dtype="<f8",sha256=m.sha(numeric),normalized_float64_sha256=m.sha(numeric))}),
  entries=[entry],removed_get_attrs=["mask"],resource_profile=m.PROFILE,production_activation=False,
  original_state_preserved=True,implementation_sha256=m.sha(Path(m.__file__).read_bytes()))
 record["source_identity"]=digest(dict(source_inventory=source,public_state=state,source_fx_sha256=record["source_fx_sha256"],resource_profile=m.PROFILE))
 record["binding"]=digest(record)
 return record,{"mask":payload}
class NumericMaskTests(unittest.TestCase):
 def test_zero_one_numeric_values(self):
  p,raw=m.mask_plan([2,3],[3],bytes([0,1,1]));self.assertEqual(struct.unpack("<3d",raw),(0.,1.,1.))
  self.assertEqual(p["approximation"],"none");self.assertEqual(p["result_shape"],[2,3])
 def test_nonzero_bool_storage_represents_true(self):
  _,raw=m.mask_plan([3],[3],bytes([0,2,255]));self.assertEqual(struct.unpack("<3d",raw),(0.,1.,1.))
 def test_column_broadcast(self):
  p,raw=m.mask_plan([2,3],[2,1],bytes([1,0]));self.assertEqual(p["result_shape"],[2,3]);self.assertEqual(len(raw),16)
 def test_scalar_boolean_broadcast(self):
  p,raw=m.mask_plan([2,3],[],b"\1");self.assertEqual(p["result_shape"],[2,3]);self.assertEqual(struct.unpack("<d",raw),(1.,))
 def test_zero_or_negative_shape_rejected(self):
  for shape in([0],[2,-1]):
   with self.subTest(shape=shape),self.assertRaises(ValueError):m.mask_plan([2,3],shape,b"")
 def test_bad_payload_size_rejected(self):
  for raw in(b"",b"\1\0",b"\1\0\1\0"):
   with self.subTest(raw=raw),self.assertRaisesRegex(ValueError,"boolean mask payload"):m.mask_plan([2,3],[3],raw)
 def test_list_payload_rejected(self):
  with self.assertRaisesRegex(ValueError,"boolean mask payload"):m.mask_plan([2,3],[3],[True,False,True])
 def test_incompatible_broadcast_rejected(self):
  with self.assertRaisesRegex(ValueError,"broadcast"):m.mask_plan([2,3],[4],b"\1"*4)
 def test_inplace_broadcast_expansion_rejected(self):
  with self.assertRaisesRegex(ValueError,"cannot expand"):m.mask_plan([2,1],[3],b"\1"*3,inplace=True)
 def test_resource_rank_rejected(self):
  with self.assertRaisesRegex(ValueError,"static shape"):m.mask_plan([1]*33,[1],b"\1")
class GraphBindingTests(unittest.TestCase):
 def test_exact_complete_graph_and_raw_state(self):
  r,p=fixture();self.assertIs(m.check(r,p,expected_source_identity=r["source_identity"]),r)
  self.assertEqual(r["public_state"]["mask"]["dtype"],"|b1")
  self.assertEqual(len(r["normalized_state"]),2)
 def reject(self,modify,pattern=None):
  r,p=fixture();identity=r["source_identity"];modify(r,p)
  r["binding"]=digest({k:v for k,v in r.items()if k!="binding"})
  if pattern:
   with self.assertRaisesRegex(ValueError,pattern):m.check(r,p,expected_source_identity=identity)
  else:
   with self.assertRaises(ValueError):m.check(r,p,expected_source_identity=identity)
 def test_numeric_value_substitution(self):
  self.reject(lambda r,p:r["normalized_state"][r["entries"][0]["derived_state_name"]].update(sha256="0"*64),"Derived numeric mask")
 def test_raw_mask_substitution(self):self.reject(lambda r,p:p.update(mask=b"\0\1\0"),"state bytes")
 def test_state_dtype_substitution(self):self.reject(lambda r,p:r["public_state"]["mask"].update(dtype="<f8"),"Authenticated boolean source")
 def test_removed_computation_is_rejected(self):self.reject(lambda r,p:r["normalized_inventory"].pop(2),"Complete boolean graph")
 def test_extra_computation_is_rejected(self):
  self.reject(lambda r,p:r["normalized_inventory"].insert(2,node("extra","call_function","_operator.neg",[value("x")])),"Complete boolean graph")
 def test_output_binding_cannot_change(self):
  self.reject(lambda r,p:r["normalized_inventory"][-1]["args"].update(items=[value("x")]),"Complete boolean graph")
 def test_wrong_input_is_rejected(self):self.reject(lambda r,p:r["entries"][0].update(input="mask"),"source entry")
 def test_missing_registered_use_is_rejected(self):self.reject(lambda r,p:r.update(entries=[]),"use inventory")
 def test_duplicate_registered_use_is_rejected(self):self.reject(lambda r,p:r["entries"].append(copy.deepcopy(r["entries"][0])),"use inventory")
 def test_fresh_node_cannot_overwrite_input(self):self.reject(lambda r,p:r["entries"][0].update(derived_fx_node="x"),"Fresh boolean FX")
 def test_original_bool_array_cannot_be_replaced(self):self.reject(lambda r,p:r["normalized_state"]["mask"].update(dtype="<f8"),"Derived numeric mask")
 def test_source_identity_cannot_be_self_rebound(self):
  def alter(r,p):
   r["source_inventory"][2]["target"]="_operator.add"
   r["source_identity"]=digest(dict(source_inventory=r["source_inventory"],public_state=r["public_state"],source_fx_sha256=r["source_fx_sha256"],resource_profile=m.PROFILE))
  self.reject(alter,"Authenticated boolean source")
 def test_plaintext_only_mask_arithmetic_is_not_this_partition(self):
  r,p=fixture();source=r["source_inventory"];source[0]=node("x","get_attr","weight")
  state=dict(r["public_state"],weight=dict(shape=[2,3],dtype="<f8",sha256="0"*64,normalized_float64_sha256="0"*64))
  with self.assertRaisesRegex(ValueError,"encrypted float input"):m.selected_entries(source,state,p)
 def test_boolean_inplace_receiver_cannot_mutate_public_state(self):
  r,p=fixture();source=r["source_inventory"];source[2]=node("product","call_method","mul_",[value("mask"),value("x")])
  with self.assertRaisesRegex(ValueError,"cannot be mutated"):m.selected_entries(source,r["public_state"],p)
 def test_public_mask_with_another_use_is_retained(self):
  r,p=fixture();source=r["source_inventory"];source[-1]["args"]["items"].append(value("mask"))
  projected,removed=m.projected(source,r["entries"]);self.assertEqual(removed,[])
  self.assertTrue(any(n["name"]=="mask"for n in projected))
 def test_extra_keyword_is_rejected(self):
  n=node("product","call_method","mul",[value("x"),value("mask")]);n["kwargs"]["items"]=[["alpha",dict(kind="scalar",value=2)]]
  with self.assertRaisesRegex(ValueError,"keyword"):m.operands(n)
 def test_torch_mul_out_rejected(self):
  n=node("product","call_function","torch.mul",[value("x"),value("mask")]);n["kwargs"]["items"]=[["out",value("x")]]
  with self.assertRaisesRegex(ValueError,"out is forbidden"):m.operands(n)
 def test_reversed_public_operand_is_registered(self):
  r,p=fixture();source=r["source_inventory"];source[2]["args"]["items"].reverse()
  entries=m.selected_entries(source,r["public_state"],p);self.assertEqual(entries[0]["input"],"x");self.assertEqual(entries[0]["operand_order"],0)
 def test_wrong_mask_tensor_metadata_rejected(self):
  r,p=fixture();r["source_inventory"][1]["tensor"]["dtype"]="int64"
  with self.assertRaisesRegex(ValueError,"typed shape"):m.selected_entries(r["source_inventory"],r["public_state"],p)
