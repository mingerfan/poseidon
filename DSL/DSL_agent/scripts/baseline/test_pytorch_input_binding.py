"""Input binding, public specialization, output restoration and replay gates."""
import copy, inspect, json, math, tempfile, unittest
from pathlib import Path
import torch
from benchmark_graph import digest
from pytorch_input_binding import (CALL_FORMAT,SOURCE_FORMAT,TensorRef,validate_call,
    public_tree,plan,signature_record,normalize,check_flow,invoke,reference_module)
from pytorch_static_trace import trace
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_public_value_ops import ARITHMETIC_POLICY
def tensor(name):return dict(kind="tensor",name=name)
def literal(value):return dict(kind="literal",value=value)
def container(kind,*items):return dict(kind=kind,items=list(items))
def call(args,kwargs=None):return dict(format=CALL_FORMAT,args=args,kwargs=kwargs or {})
def spec(binding):return dict(format=SOURCE_FORMAT,inputs=[dict(name=n,shape=[2,3])for n in("y","x")],call=binding)
class Dictionary(torch.nn.Module):
    def forward(self,batch):return {"kind":batch["x"]+batch["y"]*2,"extras":[None,batch["x"].square()]}
class Nested(torch.nn.Module):
    def forward(self,values,config):
        return (values[0]+values[1][0]*config["scale"],[values[1][0],None])
class Keywords(torch.nn.Module):
    def forward(self,x,/,*,other,scale=2.0,bias=None,mode="linear",aux=False):
        y=x+other*scale
        if bias is not None:y=y+bias
        return {"y":y,"aux":x}if aux else y
class KeywordOnly(torch.nn.Module):
    def forward(self,*,left,right):return left-right
class Defaults(torch.nn.Module):
    def forward(self,x,y,settings={"scale":0.5,"shape":[2,3]}):
        return (x+y*settings["scale"]).reshape(settings["shape"])
class InputBindingTests(unittest.TestCase):
    def setUp(self):
        self.x=torch.arange(6,dtype=torch.float64).reshape(2,3)
        self.y=self.x*.25-1
        self.dcall=call([container("dict",["x",tensor("x")],["y",tensor("y")])])
    def capture(self,module,s):
        r=plan(s,signature_record(module))
        fx=trace(module,s["inputs"],resource_profile="general-tensors-v1",**TRACE_OPTIONS,
                 public_expression_policy=ARITHMETIC_POLICY,input_plan=r)
        n,f=normalize(fx,r);check_flow(f,s)
        torch.testing.assert_close(invoke(module,r,dict(x=self.x,y=self.y)),n(self.y,self.x),atol=0,rtol=0)
        return fx,n,f
    def test_dictionary_reorders_declared_abi_and_retains_kind_key(self):
        fx,n,f=self.capture(Dictionary(),spec(self.dcall))
        self.assertEqual([x.target for x in n.graph.nodes if x.op=="placeholder"],["y","x"])
        self.assertEqual(f["removed_guards"],[])
    def test_module_parameter_projection_keeps_random_linear_weights(self):
        from pytorch_public_expression_flow import normalize as public_normalize,_arrays,state_identity
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(3,3).double()
            def forward(self,batch):return self.linear(batch["x"])+batch["y"]
        module=Linear();fx,n,f=self.capture(module,spec(self.dcall))
        arrays=_arrays(n);effective,flow=public_normalize(n,arrays,expected_state=state_identity(arrays,resource_profile="general-tensors-v1"),
            resource_profile="general-tensors-v1",policy=ARITHMETIC_POLICY)
        self.assertEqual(set(flow["effective_state"]),{"linear.weight","linear.bias"})
        self.assertEqual(state_identity(_arrays(effective),resource_profile="general-tensors-v1"),flow["effective_state"])
    def test_nested_tuple_list_and_public_dictionary(self):
        s=spec(call([container("tuple",tensor("x"),container("list",tensor("y"))),
                     container("dict",["scale",literal(.25)])]))
        self.capture(Nested(),s)
    def test_positional_only_keywords_and_frozen_none_bool_string_defaults(self):
        _,_,f=self.capture(Keywords(),spec(call([tensor("x")],dict(other=tensor("y")))))
        self.assertEqual(len(f["removed_guards"]),7)
    def test_all_keyword_only_inputs(self):
        self.capture(KeywordOnly(),spec(call([],dict(left=tensor("x"),right=tensor("y")))))
    def test_explicit_public_value_overrides_default(self):
        s=spec(call([tensor("x")],dict(other=tensor("y"),scale=literal(.5),aux=literal(True))))
        self.capture(Keywords(),s)
    def test_nested_public_defaults(self):
        self.capture(Defaults(),spec(call([tensor("x"),tensor("y")])))
    def test_flat_tensor_output_restored(self):
        _,n,_=self.capture(Dictionary(),spec(self.dcall))
        self.assertIsInstance(n(self.y,self.x)["kind"],torch.Tensor)
    def test_default_plan_is_reproducible(self):
        s=spec(call([tensor("x"),tensor("y")]))
        self.assertEqual(plan(s,signature_record(Defaults())),plan(s,signature_record(Defaults())))
    def test_wrong_shape_rejected_by_replay(self):
        s=spec(self.dcall);_,_,f=self.capture(Dictionary(),s)
        f["source_inventory"][0]["tensor"]["shape"]=[3,2];f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"placeholder_shape"):check_flow(f,s)
    def test_changed_public_guard_value_rejected(self):
        s=spec(call([tensor("x")],dict(other=tensor("y"))));_,_,f=self.capture(Keywords(),s)
        eq=next(n for n in f["source_inventory"]if n["target"]=="_operator.eq")
        eq["args"]["items"][1]["value"]=.5;f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"specialization_guard"):check_flow(f,s)
    def test_changed_guard_message_rejected(self):
        s=spec(call([tensor("x")],dict(other=tensor("y"))));_,_,f=self.capture(Keywords(),s)
        guard=next(n for n in f["source_inventory"]if n["target"]=="torch._assert")
        guard["args"]["items"][1]["value"]="anything";f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"specialization_message"):check_flow(f,s)
    def test_guard_used_by_output_rejected(self):
        s=spec(call([tensor("x")],dict(other=tensor("y"))));_,_,f=self.capture(Keywords(),s)
        guard=next(n for n in f["source_inventory"]if n["target"]=="_operator.eq")
        f["source_inventory"][-1]["args"]["items"][0]=dict(kind="fx_value",name=guard["name"])
        f["output_template"]=None;f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"outside_guard"):check_flow(f,s)
    def test_output_tree_tampering_rejected(self):
        s=spec(self.dcall);_,_,f=self.capture(Dictionary(),s)
        f["output_template"]["items"][0][1]["index"]=1
        f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaises(ValueError):check_flow(f,s)
    def test_normalized_abi_tampering_rejected(self):
        s=spec(self.dcall);_,_,f=self.capture(Dictionary(),s)
        f["normalized_inventory"][0]["target"]="fake"
        f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"flow_changed"):check_flow(f,s)
    def test_missing_specialization_guard_rejected(self):
        s=spec(call([tensor("x")],dict(other=tensor("y"))));_,_,f=self.capture(Keywords(),s)
        f["source_inventory"]=[n for n in f["source_inventory"]if n["name"]not in f["removed_guards"]]
        f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"missing_or_duplicate_guard"):check_flow(f,s)
    def test_changed_default_projection_rejected(self):
        s=spec(call([tensor("x"),tensor("y")]));_,_,f=self.capture(Defaults(),s)
        f["placeholder_defaults"]["settings_1"]=[];f["binding"]=digest({k:v for k,v in f.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"placeholder_defaults"):check_flow(f,s)
    def test_manifest_call_changes_identity(self):
        a=spec(call([tensor("x")],dict(other=tensor("y"))))
        b=copy.deepcopy(a);b["call"]["kwargs"]["scale"]=literal(.5)
        self.assertNotEqual(plan(a,signature_record(Keywords()))["binding"],plan(b,signature_record(Keywords()))["binding"])
    def test_missing_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing_or_duplicate"):validate_call(call([tensor("x")]),spec(self.dcall)["inputs"])
    def test_duplicate_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"missing_or_duplicate"):validate_call(call([tensor("x"),tensor("x")]),spec(self.dcall)["inputs"])
    def test_unknown_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"tensor_binding"):validate_call(call([tensor("secret")]),spec(self.dcall)["inputs"])
    def test_duplicate_dictionary_key_rejected(self):
        c=call([container("dict",["x",tensor("x")],["x",tensor("y")])])
        with self.assertRaisesRegex(ValueError,"duplicate_key"):validate_call(c,spec(c)["inputs"])
    def test_nan_literal_rejected(self):
        c=call([tensor("x"),tensor("y"),literal(float("nan"))])
        with self.assertRaisesRegex(ValueError,"finite_scalar"):validate_call(c,spec(c)["inputs"])
    def test_integer_precision_rejected(self):
        c=call([tensor("x"),tensor("y"),literal(2**53+1)])
        with self.assertRaisesRegex(ValueError,"integer_precision"):validate_call(c,spec(c)["inputs"])
    def test_extra_call_field_rejected(self):
        c=copy.deepcopy(self.dcall);c["private"]=True
        with self.assertRaisesRegex(ValueError,"fields"):validate_call(c,spec(c)["inputs"])
    def test_wrong_kind_rejected(self):
        c=call([dict(kind="pickle",value="x"),tensor("y")])
        with self.assertRaisesRegex(ValueError,"tree_kind"):validate_call(c,spec(c)["inputs"])
    def test_nested_depth_rejected(self):
        tree=tensor("x")
        for _ in range(9):tree=container("list",tree)
        c=call([tree,tensor("y")])
        with self.assertRaisesRegex(ValueError,"capacity"):validate_call(c,spec(c)["inputs"])
    def test_invalid_forward_keyword_rejected(self):
        c=call([tensor("x")],{"other":tensor("y"),"wrong":literal(1)})
        with self.assertRaisesRegex(ValueError,"signature_mismatch"):plan(spec(c),signature_record(Keywords()))
    def test_positional_only_keyword_rejected(self):
        c=call([],dict(x=tensor("x"),other=tensor("y")))
        with self.assertRaisesRegex(ValueError,"signature_mismatch"):plan(spec(c),signature_record(Keywords()))
    def test_missing_required_keyword_rejected(self):
        c=call([tensor("x"),tensor("y")])
        with self.assertRaisesRegex(ValueError,"signature_mismatch"):plan(spec(c),signature_record(Keywords()))
    def test_variadic_forward_rejected(self):
        class Var(torch.nn.Module):
            def forward(self,*args):return args[0]
        with self.assertRaisesRegex(ValueError,"variadic"):signature_record(Var())
    def test_non_data_default_rejected(self):
        class Bad(torch.nn.Module):
            def forward(self,x,y,config=object()):return x+y
        with self.assertRaisesRegex(ValueError,"literal_type"):signature_record(Bad())
    def test_dynamic_encrypted_control_rejected(self):
        class Dynamic(torch.nn.Module):
            def forward(self,batch):
                if batch["x"].sum()>0:return batch["x"]
                return batch["y"]
        with self.assertRaises(Exception):self.capture(Dynamic(),spec(self.dcall))
    def test_tensor_input_mutation_rejected(self):
        class Mutation(torch.nn.Module):
            def forward(self,x,y):x.add_(1);return x+y
        r=plan(spec(call([tensor("x"),tensor("y")])),signature_record(Mutation()))
        with self.assertRaisesRegex(ValueError,"tensor_mutated"):
            reference_module(Mutation(),r)(self.y,self.x)
    def test_container_input_mutation_rejected(self):
        class Mutation(torch.nn.Module):
            def forward(self,batch):batch["z"]=3;return batch["x"]+batch["y"]
        r=plan(spec(self.dcall),signature_record(Mutation()))
        with self.assertRaisesRegex(ValueError,"container_mutated"):invoke(Mutation(),r,dict(x=self.x,y=self.y))
    def test_runtime_dictionary_and_declared_order(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(self.dcall),signature_record(Dictionary()))
        result=flatten_samples(r,[dict(args=[{"x":self.x,"y":self.y}],kwargs={})])
        self.assertIs(result[0]["x"],self.x);self.assertIs(result[0]["y"],self.y)
    def test_runtime_public_defaults_are_frozen(self):
        from pytorch_input_binding import flatten_samples
        c=call([tensor("x")],dict(other=tensor("y")));r=plan(spec(c),signature_record(Keywords()))
        sample=dict(args=[self.x],kwargs=dict(other=self.y))
        self.assertEqual(set(flatten_samples(r,[sample])[0]),{"x","y"})
        sample["kwargs"]["scale"]=.25
        with self.assertRaisesRegex(ValueError,"public_value_changed"):flatten_samples(r,[sample])
    def test_runtime_tuple_list_mismatch_rejected(self):
        from pytorch_input_binding import flatten_samples
        c=call([container("tuple",tensor("x"),container("list",tensor("y"))),
                container("dict",["scale",literal(.25)])])
        r=plan(spec(c),signature_record(Nested()))
        with self.assertRaisesRegex(ValueError,"runtime_container"):
            flatten_samples(r,[dict(args=[[self.x,[self.y]],{"scale":.25}],kwargs={})])
    def test_runtime_extra_key_rejected(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(self.dcall),signature_record(Dictionary()))
        with self.assertRaisesRegex(ValueError,"runtime_dictionary"):
            flatten_samples(r,[dict(args=[{"x":self.x,"y":self.y,"z":3}],kwargs={})])
    def test_runtime_bool_integer_confusion_rejected(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(call([tensor("x")],dict(other=tensor("y")))),signature_record(Keywords()))
        with self.assertRaisesRegex(ValueError,"public_value_changed"):
            flatten_samples(r,[dict(args=[self.x],kwargs=dict(other=self.y,aux=0))])
    def test_runtime_positional_only_rejected(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(call([tensor("x")],dict(other=tensor("y")))),signature_record(Keywords()))
        with self.assertRaisesRegex(ValueError,"runtime_signature"):
            flatten_samples(r,[dict(args=[],kwargs=dict(x=self.x,other=self.y))])
    def test_runtime_sample_count_17_rejected(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(self.dcall),signature_record(Dictionary()))
        with self.assertRaisesRegex(ValueError,"sample_count"):flatten_samples(r,[{}]*17)
    def test_runtime_defaults_container_value_rejected(self):
        from pytorch_input_binding import flatten_samples
        r=plan(spec(call([tensor("x"),tensor("y")])),signature_record(Defaults()))
        with self.assertRaisesRegex(ValueError,"public_value_changed"):
            flatten_samples(r,[dict(args=[self.x,self.y],kwargs=dict(settings={"scale":.5,"shape":[3,2]}))])
    def test_application_invalid_sample_format_is_terminal(self):
        from application_component import Application
        with tempfile.TemporaryDirectory()as folder:
            app=Application(Path(folder),object())
            result=app.execute("unused",[],sample_format="arbitrary")
            self.assertEqual(result["status"],"unsupported");self.assertFalse(result["encrypted_execution"])
    def test_backend_forwards_only_explicit_structured_format(self):
        from unittest.mock import patch
        from application_backend import PoseidonBackend
        with patch("program_execution.execute_program",return_value={"status":"test_only"})as execute:
            PoseidonBackend().execute(Path("/tmp/test-only"),[],sample_format="forward_call")
            self.assertEqual(execute.call_args.kwargs,{"sample_format":"forward_call"})
            execute.reset_mock()
            PoseidonBackend().execute(Path("/tmp/test-only"),[])
            self.assertEqual(execute.call_args.kwargs,{})
    def test_old_v25_rejects_structured_call_field(self):
        from pytorch_ordered_contract import shim
        from pytorch_source_contract import validate_manifest
        v=spec(self.dcall);v["format"]="poseidon-trusted-pytorch-v25"
        with self.assertRaisesRegex(ValueError,"fields"):validate_manifest(shim(v))
    def test_v26_shim_retains_original_call(self):
        from pytorch_ordered_contract import shim
        v=spec(self.dcall);before=copy.deepcopy(v);base=shim(v)
        self.assertNotIn("call",base);self.assertEqual(v,before)
    def test_capability_uses_explicit_version_and_no_success_claim(self):
        from application_capabilities import input_sources
        c=input_sources()["trusted_pytorch"]["input_call"]
        self.assertEqual(c["version"],SOURCE_FORMAT)
        self.assertEqual(c["encrypted_status"],"unverified")
if __name__=="__main__":unittest.main()
