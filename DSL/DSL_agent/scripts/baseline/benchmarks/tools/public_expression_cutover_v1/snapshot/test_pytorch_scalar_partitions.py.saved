"""Scalar partition source capture, alias effects and transport rejection tests."""
import copy,unittest
from benchmark_graph import digest
from pytorch_partition_normalization import POLICY,SCALAR_POLICY,plan,normalize,check_transport
from test_pytorch_einsum_views import model,PROFILE,APPROX
def prepare(module,shapes,outputs,partition=SCALAR_POLICY):
 from pytorch_ordered_capture import prepare_module
 sig=[dict(name="x"if i==0 else"y",shape=s)for i,s in enumerate(shapes)]
 result=prepare_module(module,sig,outputs,resource_profile=PROFILE,approximations=APPROX,
  dtype_policy="strict-float64-v1",matrix_policy="static-tensor-matmul-v1",
  approximation_policy="node-module-default-v1",functional_policy="static-adaptive-average-v1",
  common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
  gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
  partition_source_policy=partition,einsum_source_policy="poseidon-pytorch-static-einsum-v2")
 return result,sig
class SourceTests(unittest.TestCase):
 def source(self,body,shapes,outputs=["out"],buffer=False):
  result,sig=prepare(model(body,buffer=buffer),shapes,outputs)
  from pytorch_mha_ordered_reference import validate
  ref=validate(result,sig,outputs,resource_profile=PROFILE,count=4)
  self.assertEqual(ref["status"],"passed",ref)
  if not buffer:
   from pytorch_ordered_contract import capture_binding,SCALAR_PARTITION_SOURCE_FORMAT
   from pytorch_weight_package import binding
   spec=dict(format=SCALAR_PARTITION_SOURCE_FORMAT,entry="model.make",files={"model.py":"0"*64},
    factory=dict(args=[],kwargs={}),weights={},dtype_policy="strict-float64-v1",
    inputs=sig,outputs=outputs,approximations=APPROX)
   high,lowered,bound,_=result
   bound["pytorch_capture"]["weight_binding"]=binding({},{})
   checked,_=capture_binding(high,spec,resource_profile=PROFILE,capture_metadata=bound["pytorch_capture"],
    expected_weight_binding=binding({},{}))
   self.assertEqual(checked,lowered)
  return result
 def test_singleton_scalar_output(self):self.source("return x.unbind()[0]",[[1]])
 def test_function_keywords_scalar_arithmetic(self):self.source("a,b=torch.unbind(input=x,dim=-1)\nreturn a+b*.25",[[2]])
 def test_method_tuple_output(self):self.source("return x.unbind(0)",[[2]],["a","b"])
 def test_direct_stack(self):self.source("return torch.stack(torch.unbind(x),dim=0)",[[4]])
 def test_partial_negative_ports(self):self.source("p=x.unbind()\nreturn p[-1]+p[0]*.25",[[4]])
 def test_repeated_ports(self):self.source("p=x.unbind()\nreturn torch.stack([p[1],p[1],p[-1]])",[[3]])
 def test_tuple_slice(self):self.source("return torch.stack(x.unbind()[::2])",[[4]])
 def test_dictionary_list_output(self):self.source("p=x.unbind()\nreturn dict(first=[p[-1]],second=p[0])",[[2]],["last","first"])
 def test_scalar_view_write_updates_source(self):self.source("a,b=x.unbind()\na.add_(b*.125)\nreturn a,x",[[2]],["scalar","updated"])
 def test_source_write_updates_views(self):self.source("a,b=x.unbind()\nx.mul_(.25)\nreturn a+b",[[2]])
 def test_snapshot_breaks_alias(self):self.source("a,b=x.unbind()\ns=a.clone()\na.mul_(.25)\nreturn s,x",[[2]],["snapshot","updated"])
 def test_detach_keeps_alias(self):self.source("a,b=x.unbind()\na.detach().add_(.125)\nreturn x,b",[[2]],["updated","tail"])
 def test_repeated_port_write(self):self.source("p=x.unbind()\na,b=p[1],p[-1]\na.add_(.125)\nreturn b,x",[[2]],["alias","updated"])
 def test_nested_chunk_then_scalar_unbind(self):self.source("a,b=x.chunk(2)\nu,v=a.unbind()\nu.mul_(.25)\nreturn torch.cat([a,b]),v",[[4]],["merged","scalar"])
 def test_scalar_einsum_factor(self):self.source("a,b=x.unbind()\nreturn torch.einsum(',i->i',a,y)+b",[[2],[2]])
 def test_scalar_reshape_write(self):self.source("a,b=x.unbind()\na.reshape(()).add_(.125)\nreturn x",[[2]])
 def test_public_scalar_gain(self):self.source("a,b=x.unbind()\nreturn a*self.gain+b",[[2]],buffer=True)
 def test_transposed_tensor_unbind(self):self.source("return torch.stack(x.transpose(0,1).unbind(-1),dim=-1)",[[2,3]])
 def test_public_unbind_read(self):
  import torch
  m=model("a,b=self.weight.unbind()\nreturn x*a+b")
  m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
  result,sig=prepare(m,[[2]],["out"])
  from pytorch_mha_ordered_reference import validate
  self.assertEqual(validate(result,sig,["out"],resource_profile=PROFILE,count=4)["status"],"passed")
 def test_public_chunk_affine(self):
  import torch
  m=model("a,b=self.weight.chunk(2)"+chr(10)+"return x*a+b")
  m.register_buffer("weight",torch.tensor([.2,.3,.4,.5],dtype=torch.float64))
  result,sig=prepare(m,[[2]],["out"])
  from pytorch_mha_ordered_reference import validate
  self.assertEqual(validate(result,sig,["out"],resource_profile=PROFILE,count=4)["status"],"passed")
 def test_public_scalar_left_subtraction(self):
  import torch
  m=model("a,b=self.weight.unbind()"+chr(10)+"return a-x*b")
  m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
  result,sig=prepare(m,[[2]],["out"])
  from pytorch_mha_ordered_reference import validate
  self.assertEqual(validate(result,sig,["out"],resource_profile=PROFILE,count=4)["status"],"passed")
 def test_public_view_operand_substitution_rejected(self):
  import torch
  from pytorch_arithmetic_normalization import check_record
  m=model("a,b=self.weight.unbind()"+chr(10)+"return x*a+b")
  m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
  result,_=prepare(m,[[2]],["out"])
  alias=result[2]["pytorch_capture"]["ordered_alias_capture"]
  record=copy.deepcopy(alias["capture_adapter"]["arithmetic"])
  graph=copy.deepcopy(alias["capture_emissions"])
  add=next(n for n in graph["nodes"]if n["id"]=="fx_add");add["inputs"][1]="value_getitem"
  record["captured_graph_sha256"]=digest(graph);record["binding_sha256"]=digest({k:v for k,v in record.items()if k!="binding_sha256"})
  with self.assertRaisesRegex(ValueError,"public view source binding"):check_record(record,graph)
 def test_public_view_root_content_substitution_rejected(self):
  import torch
  from pytorch_arithmetic_normalization import check_record
  m=model("a,b=self.weight.unbind()"+chr(10)+"return x*a+b")
  m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
  result,_=prepare(m,[[2]],["out"])
  alias=result[2]["pytorch_capture"]["ordered_alias_capture"]
  record=copy.deepcopy(alias["capture_adapter"]["arithmetic"]);graph=copy.deepcopy(alias["capture_emissions"])
  graph["constants"]["weight_0"][0]=.8;record["captured_graph_sha256"]=digest(graph)
  record["binding_sha256"]=digest({k:v for k,v in record.items()if k!="binding_sha256"})
  with self.assertRaisesRegex(ValueError,"view root binding"):check_record(record,graph)
 def test_public_unbind_write_rejected(self):
  import torch
  m=model("a,b=self.weight.unbind()\na.add_(.2)\nreturn x*a+b")
  m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
  with self.assertRaises(ValueError):prepare(m,[[2]],["out"])
 def test_old_version_scalar_unbind_rejected(self):
  with self.assertRaisesRegex(ValueError,"Scalar unbind"):prepare(model("return x.unbind()[0]"),[[2]],["out"],partition=POLICY)
 def test_legacy_partial_ports_are_already_supported(self):
  import torch
  from pytorch_static_trace import trace,ALIAS_POLICY
  m=model("return x.unbind(0)[-1]")
  sig=[dict(name="x",shape=[2,3])]
  traced=trace(m,sig,resource_profile=PROFILE,policy=ALIAS_POLICY,partition_source_policy=POLICY)
  normalized,record=normalize(traced,sig,resource_profile=PROFILE)
  self.assertEqual([p["port"]for p in record["entries"][0]["ports"]],[0,1])
  self.assertTrue(torch.equal(m(torch.arange(6).reshape(2,3)),normalized(torch.arange(6).reshape(2,3))))
 def test_dynamic_port_rejected(self):
  with self.assertRaises((TypeError,ValueError,IndexError)):
   prepare(model("return x.unbind()[x.sum()]"),[[2]],["out"])
 def test_v2_requires_scalar_alias_contract(self):
  from pytorch_ordered_capture import prepare_module
  with self.assertRaisesRegex(ValueError,"scalar view contract"):
   prepare_module(model("return x"),[dict(name="x",shape=[2])],["out"],resource_profile=PROFILE,
    common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
    gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
    partition_source_policy=SCALAR_POLICY,einsum_source_policy="poseidon-pytorch-static-einsum-v1")
class TransportTests(unittest.TestCase):
 def record(self):
  from pytorch_static_trace import trace,ALIAS_POLICY
  sig=[dict(name="x",shape=[2])]
  traced=trace(model("return torch.stack(x.unbind())"),sig,resource_profile=PROFILE,policy=ALIAS_POLICY,
   partition_source_policy=SCALAR_POLICY)
  _,record=normalize(traced,sig,resource_profile=PROFILE,source_policy=SCALAR_POLICY)
  return record,sig
 def test_explicit_scalar_plan(self):
  p=plan([4],"unbind",axis=-1,source_policy=SCALAR_POLICY)
  self.assertEqual(p["policy"],SCALAR_POLICY);self.assertEqual([v["result_shape"]for v in p["ports"]],[[]]*4)
 def test_registered_policy_only(self):
  with self.assertRaises(ValueError):plan([2],"unbind",source_policy="unchecked")
 def test_empty_shape_rejected(self):
  for shape in ([],[0],[True]):
   with self.subTest(shape=shape),self.assertRaises(ValueError):plan(shape,"unbind",source_policy=SCALAR_POLICY)
 def test_axis_and_budget_rejected(self):
  for shape,axis in (([2],True),([2],1),([1025],0),([1]*33,0)):
   with self.subTest(shape=shape,axis=axis),self.assertRaises(ValueError):plan(shape,"unbind",axis=axis,source_policy=SCALAR_POLICY)
 def check_bad(self,edit):
  record,sig=self.record();edit(record)
  record["binding"]=digest({k:v for k,v in record.items()if k!="binding"})
  with self.assertRaises(ValueError):check_transport(record,inputs=sig,resource_profile=PROFILE,source_policy=SCALAR_POLICY)
 def test_rank_one_ports_must_retain_logical_scalar(self):
  self.check_bad(lambda r:r["entries"][0]["plan"]["ports"][0].update(result_shape=[1]))
 def test_wrong_index_rejected(self):
  def edit(r):
   n=next(n for n in r["normalized_inventory"]if n["name"]==r["entries"][0]["ports"][0]["name"])
   n["args"]["items"][1]["items"][0]["value"]=1
  self.check_bad(edit)
 def test_missing_port_rejected(self):self.check_bad(lambda r:r["entries"][0]["ports"].pop())
 def test_missing_entry_rejected(self):self.check_bad(lambda r:r.update(entries=[]))
 def test_policy_downgrade_rejected(self):self.check_bad(lambda r:r.update(format=POLICY))
 def test_old_checker_does_not_accept_v2(self):
  record,sig=self.record()
  with self.assertRaises(ValueError):check_transport(record,inputs=sig,resource_profile=PROFILE)
 def test_versions_and_capability(self):
  from pytorch_model_intake import SCALAR_PARTITION_FORMAT,SOURCE_FORMATS,PACKAGES,REFERENCES
  from application_capabilities import input_sources
  self.assertIn(SCALAR_PARTITION_FORMAT,SOURCE_FORMATS)
  self.assertEqual(PACKAGES[SCALAR_PARTITION_FORMAT],"poseidon-trusted-pytorch-provenance-v20")
  self.assertEqual(REFERENCES[SCALAR_PARTITION_FORMAT],"poseidon-trusted-pytorch-reference-v20")
  cap=input_sources()["trusted_pytorch"]["ordered_scalar_partitions"]
  self.assertEqual(cap["version"],SCALAR_PARTITION_FORMAT);self.assertEqual(cap["agent_status"],"unverified")
if __name__=="__main__":unittest.main()
