"""Exact powers/public division, source effects and immutable provenance tests."""
import copy,io,math,tempfile,unittest
from benchmark_graph import digest
from pytorch_exact_arithmetic import POLICY,plan,arguments,classification,check_transport
from test_pytorch_einsum_views import model,PROFILE,APPROX
def prepare(module,shapes,outputs,*,policy=POLICY):
 from pytorch_ordered_capture import prepare_module
 signature=[dict(name="x"if i==0 else"y",shape=s)for i,s in enumerate(shapes)]
 prepared=prepare_module(module,signature,outputs,resource_profile=PROFILE,approximations=APPROX,
  dtype_policy="strict-float64-v1",matrix_policy="static-tensor-matmul-v1",
  approximation_policy="node-module-default-v1",functional_policy="static-adaptive-average-v1",
  common_calls=True,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",
  gelu_source_policy="poseidon-gelu-source-variants-v1",mha_source_policy="poseidon-mha-ordered-source-v1",
  partition_source_policy="poseidon-pytorch-static-partition-v2",einsum_source_policy="poseidon-pytorch-static-einsum-v2",
  **({"exact_arithmetic_policy":policy}if policy is not None else {}))
 return prepared,signature
def checked(module,shapes,outputs=["out"]):
 from pytorch_mha_ordered_reference import validate
 from pytorch_ordered_contract import capture_binding,EXACT_ARITHMETIC_SOURCE_FORMAT
 from pytorch_weight_package import binding,export_state
 prepared,signature=prepare(module,shapes,outputs)
 receipt=validate(prepared,signature,outputs,resource_profile=PROFILE,count=4)
 if receipt["status"]!="passed":raise AssertionError(receipt)
 arrays={n:v.detach().cpu().numpy()for n,v in module.state_dict().items()}
 with tempfile.TemporaryDirectory()as directory:
  weights=export_state(module,directory,resource_profile=PROFILE)
 spec=dict(format=EXACT_ARITHMETIC_SOURCE_FORMAT,entry="model.make",files={"model.py":"0"*64},
  factory=dict(args=[],kwargs={}),weights=weights,dtype_policy="strict-float64-v1",
  inputs=signature,outputs=outputs,approximations=APPROX)
 high,lowered,captured,_=prepared;bound=binding(arrays,weights)
 captured["pytorch_capture"]["weight_binding"]=bound
 checked_lower,_=capture_binding(high,spec,resource_profile=PROFILE,
  capture_metadata=captured["pytorch_capture"],expected_weight_binding=bound)
 if checked_lower!=lowered:raise AssertionError("normalized graph differs")
 return prepared,signature
class SourceTests(unittest.TestCase):
 def source(self,body,shapes=[[2]],outputs=["out"],buffer=False):
  return checked(model(body,buffer=buffer),shapes,outputs)[0]
 def test_operator_square(self):self.source("return x ** 2")
 def test_operator_fourth_power(self):self.source("return (x*.25) ** 4",[[2,2]])
 def test_function_keyword_power(self):self.source("return torch.pow(input=x,exponent=2,out=None)")
 def test_method_keyword_power(self):self.source("return x.pow(exponent=4.0)",[[2,1]])
 def test_power_in_multi_input_fanout(self):self.source("a=x.pow(2)\nreturn a+y,a+x",[[2],[2]],["a","b"])
 def test_power_then_static_shape(self):self.source("a=torch.pow(x,2)\nreturn a.reshape(a.numel())",[[2,2]])
 def test_power_then_linear(self):
  import torch
  m=model("return torch.nn.functional.linear(x.pow(2),self.weight,self.bias)")
  m.register_buffer("weight",torch.tensor([[.17,-.09],[.08,.11]],dtype=torch.float64))
  m.register_buffer("bias",torch.tensor([.01,-.02],dtype=torch.float64))
  checked(m,[[1,2]])
 def test_power_with_einsum(self):self.source('return torch.einsum("ij->i",x.pow(2))',[[2,2]])
 def test_scalar_power_output(self):self.source("return x.unbind()[0].pow(2)",[[2]])
 def test_scalar_fourth_power(self):self.source("a,b=x.unbind()\nreturn torch.stack([a.pow(4),b.pow(2)])",[[2]])
 def test_inplace_square_view(self):self.source("a=x.transpose(0,1)\na.pow_(2)\nreturn x,a",[[2,2]],["base","view"])
 def test_inplace_fourth_power_snapshot(self):self.source("a=x.clone()\nx.pow_(4)\nreturn a,x",[[2]],["snapshot","updated"])
 def test_inplace_power_scalar(self):self.source("a,b=x.unbind()\na.pow_(4)\nreturn x,b",[[2]],["base","tail"])
 def test_inplace_power_detached_view(self):self.source("a=x.detach()\na.pow_(2)\nreturn x")
 def test_operator_division(self):self.source("return x/2")
 def test_negative_public_division(self):self.source("return x/-2",[[2,2]])
 def test_division_float_constant(self):self.source("return x/.375")
 def test_function_division_keywords(self):self.source("return torch.div(input=x,other=2,rounding_mode=None,out=None)")
 def test_function_divide_alias(self):self.source("return torch.divide(x,2)")
 def test_function_true_divide_alias(self):self.source("return torch.true_divide(input=x,other=2)")
 def test_method_division_keywords(self):self.source("return x.div(other=2,rounding_mode=None)")
 def test_method_divide_alias(self):self.source("return x.divide(2)")
 def test_method_true_divide_alias(self):self.source("return x.true_divide(2)")
 def test_inplace_division_view(self):self.source("a=x.transpose(0,1)\na.div_(2)\nreturn x",[[2,2]])
 def test_inplace_divide_alias(self):self.source("x.divide_(2)\nreturn x")
 def test_inplace_true_divide_alias(self):self.source("x.true_divide_(2)\nreturn x")
 def test_inplace_division_snapshot(self):self.source("a=x.clone()\nx.div_(2)\nreturn a,x",[[2]],["snapshot","updated"])
 def test_scalar_inplace_division(self):self.source("a,b=x.unbind()\na.div_(2)\nreturn torch.stack([a,b])",[[2]])
 def test_scalar_buffer_divisor(self):self.source("return x/self.gain",buffer=True)
 def test_scalar_buffer_method_division(self):self.source("return x.div(self.gain)",[[2,2]],buffer=True)
 def test_scalar_buffer_inplace_division(self):self.source("x.div_(self.gain)\nreturn x",buffer=True)
 def test_scalar_buffer_declared_hash_changes(self):
  import torch
  m=model("return x/self.gain",buffer=True)
  a=checked(m,[[2]])[0][2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]
  m.gain.fill_(.5)
  b=checked(m,[[2]])[0][2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]
  self.assertNotEqual(a["exact_arithmetic_normalization"]["binding"],b["exact_arithmetic_normalization"]["binding"])
 def test_division_after_partition(self):self.source("a,b=x.chunk(2)\nreturn torch.cat([a/2,b/-2])",[[4]])
 def test_rank32_power_and_division(self):self.source("return x.pow(2)/2",[[1]*31+[2]])
 def reject(self,body,shapes=[[2]],buffer=False):
  with self.assertRaises((ValueError,TypeError,RuntimeError,AttributeError)):
   prepare(model(body,buffer=buffer),shapes,["out"])
 def test_cipher_divisor_rejected(self):self.reject("return x/y",[[2],[2]])
 def test_dynamic_exponent_rejected(self):self.reject("return x.pow(x.sum())")
 def test_tensor_exponent_rejected(self):self.reject("return x.pow(self.gain)",buffer=True)
 def test_fractional_power_rejected(self):self.reject("return x.pow(.5)")
 def test_negative_power_rejected(self):self.reject("return x.pow(-1)")
 def test_power_three_rejected(self):self.reject("return x ** 3")
 def test_bool_exponent_rejected(self):self.reject("return x.pow(True)")
 def test_zero_divisor_rejected(self):self.reject("return x/0")
 def test_cipher_reverse_division_rejected(self):self.reject("return 2/x")
 def test_rounding_division_rejected(self):self.reject('return torch.div(x,2,rounding_mode="floor")')
 def test_out_mutation_rejected(self):self.reject("return torch.pow(x,2,out=x)")
 def test_reciprocal_budget_rejected(self):self.reject("return x/.00001")
 def test_divisor_budget_rejected(self):self.reject("return x/2048")
 def test_public_tensor_vector_divisor_rejected(self):
  import torch
  m=model("return x/self.weight");m.register_buffer("weight",torch.tensor([2.,3.],dtype=torch.float64))
  with self.assertRaisesRegex(ValueError,"scalar"):prepare(m,[[2]],["out"])
 def test_public_computed_tensor_divisor_rejected(self):self.reject("return x/(self.gain+1)",buffer=True)
 def test_public_state_write_rejected(self):self.reject("self.gain.pow_(2)\nreturn x/self.gain",buffer=True)
 def test_old_v20_rejects_new_source(self):
  with self.assertRaises(ValueError):prepare(model("return x.pow(2)"),[[2]],["out"],policy=None)
 def test_policy_requires_ordered_contract(self):
  from pytorch_static_trace import trace,POLICY as STATIC
  with self.assertRaisesRegex(ValueError,"Explicit exact arithmetic"):
   trace(model("return x/2"),[dict(name="x",shape=[2])],resource_profile=PROFILE,policy=STATIC,exact_arithmetic_policy=POLICY)
class TransportTests(unittest.TestCase):
 def record(self,body="return x.pow(4)/2",buffer=False):
  result,signature=prepare(model(body,buffer=buffer),[[2]],["out"])
  return copy.deepcopy(result[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]["exact_arithmetic_normalization"]),signature
 def reject(self,change,body="return x.pow(4)/2",buffer=False):
  record,signature=self.record(body,buffer);change(record)
  record["binding"]=digest({k:v for k,v in record.items()if k!="binding"})
  with self.assertRaises(ValueError):check_transport(record,inputs=signature,resource_profile=PROFILE)
 def test_missing_entry(self):self.reject(lambda r:r["entries"].pop())
 def test_substituted_power(self):self.reject(lambda r:r["entries"][0]["plan"].update(parameter=2))
 def test_substituted_reciprocal(self):self.reject(lambda r:r["entries"][-1]["plan"]["steps"][0].update(coefficient=.75))
 def test_missing_expansion(self):self.reject(lambda r:r["normalized_inventory"].pop(1))
 def test_substituted_input(self):self.reject(lambda r:r["entries"][0].update(input="missing"))
 def test_wrong_metadata(self):self.reject(lambda r:r["normalized_inventory"][-2]["tensor"].update(dtype="float32"))
 def test_reordered_steps(self):self.reject(lambda r:r["entries"][0]["emitted_nodes"].reverse())
 def test_duplicate_names(self):self.reject(lambda r:r["source_inventory"][0].update(name=r["source_inventory"][1]["name"]))
 def test_scalar_divisor_substitution(self):
  self.reject(lambda r:r["entries"][0]["scalar_divisor"].update(value=.5),body="return x/self.gain",buffer=True)
 def test_scalar_divisor_state_hash(self):
  self.reject(lambda r:r["public_state"]["gain"].update(normalized_float64_sha256="0"*64),body="return x/self.gain",buffer=True)
 def test_implementation_substitution(self):self.reject(lambda r:r.update(implementation_sha256="0"*64))
 def test_source_text_substitution(self):self.reject(lambda r:r.update(source_fx_text="different"))
class PlanTests(unittest.TestCase):
 def test_square_single_step(self):self.assertEqual(len(plan([2],"power",2)["steps"]),1)
 def test_fourth_power_two_steps(self):self.assertEqual(len(plan([2],"power",4,inplace=True)["steps"]),2)
 def test_divisor_boundary(self):self.assertEqual(plan([],"divide",1/1024)["steps"][0]["coefficient"],1024)
 def test_divisor_upper_boundary(self):self.assertEqual(plan([2],"divide",1024)["steps"][0]["coefficient"],1/1024)
 def test_nonfinite_parameters(self):
  for value in [float("nan"),float("inf"),-float("inf"),True,None]:
   with self.subTest(value=value),self.assertRaises(ValueError):plan([2],"divide",value)
 def test_shape_budget(self):
  for shape in [[2]*33,[0],[65537]]:
   with self.subTest(shape=shape),self.assertRaises(ValueError):plan(shape,"power",2)
 def test_duplicate_keywords(self):
  with self.assertRaises(ValueError):arguments((1,2),{"exponent":2},classification("call_function","torch.pow"))
 def test_true_divide_signature(self):
  with self.assertRaises(ValueError):arguments((1,2),{"rounding_mode":None},classification("call_method","true_divide"))
if __name__=="__main__":unittest.main()
