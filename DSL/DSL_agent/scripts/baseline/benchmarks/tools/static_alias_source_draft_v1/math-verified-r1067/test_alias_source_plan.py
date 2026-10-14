import copy, unittest
import numpy as np
import torch
from benchmark_graph import digest
from benchmark_math import evaluate
from alias_source_plan import plan,verify,AliasError,SOURCE_FORMAT
from fx_alias_source_plan import from_fx
from fixtures import CASES,APPROX,capture

def one_program(shape=(2,3),nodes=None,outputs=None,constants=None):
    return dict(format=SOURCE_FORMAT,inputs=[dict(name="x",shape=list(shape))],constants=constants or {},
                nodes=nodes or [dict(id="alias",op="alias",inputs=["x"],attrs={},output="a")],
                outputs=outputs or [dict(name="out",value="a")])
def node(op,refs,attrs=None,output="a"):
    return dict(id=output,op=op,inputs=refs,attrs=attrs or {},output=output)

class AliasTests(unittest.TestCase):
    def reject(self,code,func,*args,**kwargs):
        with self.assertRaises(AliasError) as cm:func(*args,**kwargs)
        self.assertEqual(cm.exception.code,code)
        feedback=cm.exception.feedback()
        self.assertEqual(feedback["retry_policy"],"never_automatic")
        self.assertNotIn("private_source",str(feedback))
        self.assertIs(type(feedback["location"]["source_position"]),int)
    def test_all_24_known_fx_contexts(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                fixture,traced,value,meta=capture(case)
                self.assertEqual(meta["source_ir_sha256"],digest(value["source_request"]["source"]))
                self.assertFalse(value["production_intake_integrated"])
    def test_shared_tensor_uses_updated_storage(self):
        _,_,value,_=capture(CASES[1])
        inputs=dict(x=np.asarray([[-.15,.1,.15],[.05,-.1,0.]]))
        actual=evaluate(value["lowered_graph"],inputs)
        polynomial=.5*inputs["x"]+.25*inputs["x"]**2
        np.testing.assert_allclose(actual["output_0"],1.5*inputs["x"]+polynomial,rtol=1e-12,atol=1e-12)
    def test_prior_immutable_value_is_preserved(self):
        _,_,value,_=capture(CASES[22])
        inputs=dict(x=np.linspace(-.15,.15,6).reshape(2,3))
        actual=evaluate(value["lowered_graph"],inputs)
        np.testing.assert_allclose(actual["output_0"],inputs["x"]**2,rtol=1e-12,atol=1e-12)
        versions=[e["new_version"] for e in value["events"] if e["rule"]=="mutation_ssa"]
        self.assertEqual(versions,[1,2,3])
    def test_reshape_copy_separates_storage(self):
        _,_,value,_=capture(CASES[19])
        event=next(e for e in value["events"] if e["rule"]=="reshape_copy")
        self.assertNotEqual(event["input_storage"],event["output_storage"])
    def test_reshape_view_preserves_storage(self):
        _,_,value,_=capture(CASES[20])
        event=next(e for e in value["events"] if e["rule"]=="reshape_alias")
        self.assertEqual(event["input_storage"],event["output_storage"])
    def test_contiguous_alias_and_copy_are_distinguished(self):
        _,_,value,_=capture(CASES[21])
        events={e["rule"]:e for e in value["events"] if e["rule"].startswith("contiguous")}
        self.assertEqual(events["contiguous_alias"]["input_storage"],events["contiguous_alias"]["output_storage"])
        self.assertNotEqual(events["contiguous_copy"]["input_storage"],events["contiguous_copy"]["output_storage"])
    def test_clone_preserves_dense_layout_and_separates_storage(self):
        _,_,value,_=capture(CASES[18])
        event=next(e for e in value["events"] if e["rule"]=="clone_preserve_dense_layout")
        self.assertNotEqual(event["input_storage"],event["output_storage"])
    def test_partial_write_updates_only_selected_elements(self):
        _,_,value,_=capture(CASES[10])
        x=np.linspace(-.15,.15,6).reshape(2,3);other=np.full((2,2),.125)
        actual=evaluate(value["lowered_graph"],dict(x=x,other=other))
        expected=x.copy();expected[:,1:3]+=.5*other
        np.testing.assert_allclose(actual["output_1"],expected,rtol=1e-12,atol=1e-12)
    def test_split_alias_writeback_changes_original_tensor(self):
        _,_,value,_=capture(CASES[23]);x=np.linspace(-.15,.15,6).reshape(2,3)
        actual=evaluate(value["lowered_graph"],dict(x=x));expected=x.copy();expected[:,1:]+=.25
        np.testing.assert_allclose(actual["output_0"],expected,rtol=1e-12,atol=1e-12)
    def test_incompatible_view_matches_native_rejection(self):
        class Invalid(torch.nn.Module):
            def forward(self,x):return x.transpose(0,1).view(6)
        m=Invalid().eval();x=torch.arange(6,dtype=torch.float64).reshape(2,3)
        with self.assertRaises(RuntimeError):m(x)
        self.reject("alias_view_rejected",from_fx,torch.fx.symbolic_trace(m),[dict(name="x",shape=[2,3])],APPROX)
    def test_overlapping_expansion_matches_native_rejection(self):
        class Invalid(torch.nn.Module):
            def forward(self,x):return x.expand(4).add_(.125)
        m=Invalid().eval();x=torch.tensor([.15],dtype=torch.float64)
        with self.assertRaises(RuntimeError):m(x)
        self.reject("alias_overlap_rejected",from_fx,torch.fx.symbolic_trace(m),[dict(name="x",shape=[1])],APPROX)
    def test_partially_overlapping_operands_rejected(self):
        class Invalid(torch.nn.Module):
            def forward(self,x):
                x[1:].add_(x[:-1]);return x
        m=Invalid().eval();x=torch.arange(4,dtype=torch.float64)
        with self.assertRaises(RuntimeError):m(x)
        self.reject("alias_overlap_rejected",from_fx,torch.fx.symbolic_trace(m),[dict(name="x",shape=[4])],APPROX)
    def test_same_alias_operand_is_valid(self):
        class Valid(torch.nn.Module):
            def forward(self,x):
                x.add_(x);return x
        traced=torch.fx.symbolic_trace(Valid().eval())
        value,_=from_fx(traced,[dict(name="x",shape=[2,3])],APPROX)
        x=np.linspace(-.15,.15,6).reshape(2,3)
        np.testing.assert_allclose(evaluate(value["lowered_graph"],dict(x=x))["output_0"],2*x,rtol=1e-12,atol=1e-12)
    def test_empty_slice_rejected(self):
        src=one_program(nodes=[node("slice",["x"],dict(index=[dict(kind="slice",start=0,stop=0,step=None)]))])
        self.reject("alias_shape_rejected",plan,src,APPROX)
    def test_negative_slice_step_rejected(self):
        src=one_program(nodes=[node("slice",["x"],dict(index=[dict(kind="slice",start=None,stop=None,step=-1)]))])
        self.reject("alias_shape_rejected",plan,src,APPROX)
    def test_public_weight_mutation_rejected(self):
        src=one_program(constants=dict(w=[[.1,.2,.3],[.3,.2,.1]]),
                        nodes=[node("add",["w","x"],dict(inplace=True,alpha=1))])
        self.reject("alias_public_mutation_rejected",plan,src,APPROX)
    def test_arithmetic_broadcast_cannot_resize_mutated_target(self):
        src=one_program(shape=(1,3),constants=dict(w=[[.1,.2,.3],[.3,.2,.1]]),
                        nodes=[node("add",["x","w"],dict(inplace=True,alpha=1))])
        self.reject("alias_shape_rejected",plan,src,APPROX)
    def test_missing_relu_approximation_rejected(self):
        src=one_program(nodes=[node("relu",["x"],dict(inplace=True))])
        self.reject("alias_approximation_rejected",plan,src,{})
    def test_malformed_relu_profile_rejected(self):
        p=copy.deepcopy(APPROX);p["relu"]["method"]="implicit"
        src=one_program(nodes=[node("relu",["x"],dict(inplace=True))])
        self.reject("alias_approximation_rejected",plan,src,p)
    def test_polynomial_basis_is_bound(self):
        src=one_program(nodes=[node("relu",["x"],dict(inplace=True))])
        config=dict(relu=dict(method="polynomial",basis="chebyshev",domain=[-1.,1.],coefficients=[.125,.5,.125]))
        value=plan(src,config);x=np.linspace(-.15,.15,6).reshape(2,3)
        expected=np.polynomial.chebyshev.chebval(x,config["relu"]["coefficients"])
        np.testing.assert_allclose(evaluate(value["lowered_graph"],dict(x=x))["out"],expected,rtol=1e-12,atol=1e-12)
    def test_duplicate_input_names_rejected(self):
        src=one_program();src["inputs"].append(copy.deepcopy(src["inputs"][0]))
        self.reject("alias_source_rejected",plan,src,APPROX)
    def test_rank_above_profile_rejected(self):
        self.reject("alias_shape_rejected",plan,one_program(shape=(1,)*33),APPROX)
    def test_unsupported_mutation_call_rejected(self):
        for method in ("copy_","resize_","set_","as_strided"):
            with self.subTest(method=method):
                graph=torch.fx.Graph();x=graph.placeholder("x");y=graph.call_method(method,(x,x));graph.output(y)
                self.reject("alias_call_rejected",from_fx,torch.fx.GraphModule({},graph),[dict(name="x",shape=[2,3])],APPROX)
    def test_original_data_is_not_mutated(self):
        src=one_program(nodes=[node("relu",["x"],dict(inplace=True))]);before=copy.deepcopy(src);a=copy.deepcopy(APPROX)
        plan(src,APPROX);self.assertEqual(src,before);self.assertEqual(APPROX,a)
    def test_source_objects_and_cycles_are_rejected_without_hooks(self):
        class Unsafe:
            def __deepcopy__(self,memo):raise AssertionError("Object hook")
        src=one_program();src["private_source"]=Unsafe()
        self.reject("alias_source_rejected",plan,src,APPROX)
        src=one_program();src["private_source"]=src
        self.reject("alias_resource_rejected",plan,src,APPROX)
    def test_missing_source_fields_are_structured(self):
        self.reject("alias_source_rejected",plan,{},APPROX)
    def test_resigned_source_tampering_rejected(self):
        value=plan(one_program(),APPROX);expected=value["source_request_sha256"]
        value["source_request"]["source"]["inputs"][0]["shape"]=[3,2]
        value["source_request_sha256"]=digest(value["source_request"]);value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        self.reject("alias_integrity_rejected",verify,value,expected_source_request_sha256=expected)
    def test_resigned_ssa_output_tampering_rejected(self):
        value=plan(one_program(),APPROX);expected=value["source_request_sha256"]
        value["events"]=[]
        value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        self.reject("alias_integrity_rejected",verify,value,expected_source_request_sha256=expected)
    def test_unverified_layers_are_not_marked_passed(self):
        value=plan(one_program(),APPROX)
        for key in ("physical_layout_status","compiler_status","encrypted_status","agent_status"):self.assertEqual(value[key],"unverified")
        self.assertFalse(value["original_function_accuracy_certified"])
    def test_tuple_output_structure_is_kept(self):
        _,_,value,metadata=capture(CASES[2])
        self.assertEqual(metadata["output_schema"]["kind"],"tuple")
        self.assertEqual(len(metadata["output_schema"]["items"]),4)

    def test_fx_immutable_static_dimensions(self):
        class Valid(torch.nn.Module):
            def forward(self,x):
                a=x.reshape([3,2]);a.add_(.125);return x,a
        value,_=from_fx(torch.fx.symbolic_trace(Valid().eval()),[dict(name="x",shape=[2,3])],APPROX)
        x=np.linspace(-.15,.15,6).reshape(2,3)
        np.testing.assert_allclose(evaluate(value["lowered_graph"],dict(x=x))["output_0"],x+.125,rtol=1e-12,atol=1e-12)
    def test_fx_immutable_structured_output(self):
        class Valid(torch.nn.Module):
            def forward(self,x):
                y=x.clone();y.add_(.125);return {"base":x,"values":[y,None]}
        value,metadata=from_fx(torch.fx.symbolic_trace(Valid().eval()),[dict(name="x",shape=[2,3])],APPROX)
        self.assertEqual(metadata["output_schema"]["kind"],"dict")
        self.assertEqual(metadata["output_schema"]["items"][1]["value"]["kind"],"list")
        self.assertEqual(metadata["output_schema"]["items"][1]["value"]["items"][1],{"kind":"none"})
        self.assertEqual(len(value["mathematical_graph"]["outputs"]),2)

    def test_view_layout_admission_and_values_match_fixed_torch(self):
        global LAYOUT_COUNTS
        LAYOUT_COUNTS=dict(attempted=0,native_admitted=0,native_rejected=0)
        class Layout(torch.nn.Module):
            def __init__(self,transform,strategy,wanted,rank):
                super().__init__();self.transform=transform;self.strategy=strategy
                self.wanted=wanted;self.rank=rank
            def forward(self,x):
                if self.transform=="identity":y=x
                elif self.transform=="transpose":y=x.transpose(0,-1)
                elif self.transform=="permute":y=x.permute(tuple(reversed(range(self.rank))))
                elif self.transform=="step":y=x[...,::2]
                elif self.transform=="leading_slice":y=x[0:1,...]
                else:y=x[...,1:]
                if self.strategy=="clone_view":v=y.clone().view(self.wanted)
                elif self.strategy=="contiguous_view":v=y.contiguous().view(self.wanted)
                elif self.strategy=="view":v=y.view(self.wanted)
                else:v=y.reshape(self.wanted)
                v.add_(.125);return x,y,v
        for shape in ((1,3),(2,3),(2,1,3),(2,3,2)):
            sample=torch.linspace(-.15,.15,int(np.prod(shape)),dtype=torch.float64).reshape(shape)
            for transform in ("identity","transpose","permute","step","leading_slice","inner_slice"):
                if transform=="step":selected=sample[...,::2]
                elif transform=="leading_slice":selected=sample[0:1,...]
                elif transform=="inner_slice":selected=sample[...,1:]
                else:selected=sample
                wanted=(selected.numel(),)
                for strategy in ("view","reshape","clone_view","contiguous_view"):
                    with self.subTest(shape=shape,transform=transform,strategy=strategy):
                        LAYOUT_COUNTS["attempted"]+=1
                        module=Layout(transform,strategy,wanted,len(shape)).eval()
                        traced=torch.fx.symbolic_trace(module)
                        try:native=module(sample.clone());admitted=True
                        except RuntimeError:admitted=False
                        if not admitted:
                            LAYOUT_COUNTS["native_rejected"]+=1
                            self.reject("alias_view_rejected",from_fx,traced,[dict(name="x",shape=list(shape))],APPROX)
                            continue
                        LAYOUT_COUNTS["native_admitted"]+=1
                        value,_=from_fx(traced,[dict(name="x",shape=list(shape))],APPROX)
                        actual=evaluate(value["lowered_graph"],dict(x=sample.numpy()))
                        for index,expected in enumerate(native):
                            np.testing.assert_allclose(actual["output_"+str(index)],expected.numpy(),rtol=1e-12,atol=1e-12)

if __name__=="__main__":unittest.main()
