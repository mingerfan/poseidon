"""Closed public values are independently compared with Torch before DSL."""
import copy,math,operator,unittest
import numpy as np
import torch
from benchmark_graph import digest
from model_resources import resolve
from pytorch_public_value_ops import POLICY,CAST_POLICY,ARITHMETIC_POLICY,evaluate as unary
from pytorch_public_arithmetic_values import describe,evaluate,plan
from pytorch_public_expression_flow import normalize,check,state_identity,_arrays
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_static_trace import trace
from benchmarks.tools.public_arithmetic_expression_v1.models import Model,input_shape
PROFILE="general-tensors-v1"
class ClosedPublicArithmeticTests(unittest.TestCase):
    def test_arithmetic_independent_torch_and_multiple_call_styles(self):
        a=np.asarray([[.21,-.31,.61],[.17,.83,-.37]],dtype="<f8")
        b=np.asarray([.47,.69,1.13],dtype="<f8")
        for operation in("add","subtract","multiply","divide","negate"):
            for alpha in(.25,1.0):
                attrs=dict(alpha=alpha)if operation in("add","subtract")else{}
                values=[a]if operation=="negate"else[a,b]
                actual=evaluate(values,operation,attrs,resource_profile=PROFILE)
                ta=torch.from_numpy(a.copy());tb=torch.from_numpy(b.copy())
                expected={"add":lambda:torch.add(ta,tb,alpha=alpha),
                  "subtract":lambda:torch.sub(ta,tb,alpha=alpha),"multiply":lambda:torch.mul(ta,tb),
                  "divide":lambda:torch.div(ta,tb),"negate":lambda:torch.neg(ta)}[operation]().numpy()
                np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)
                self.assertFalse(actual.flags.writeable)
    def test_descriptors_and_keyword_operands(self):
        for kind,target,args,kw,op in[
            ("call_function","_operator.sub",(1.0,"p"),{},"subtract"),
            ("call_function","torch.add",(),dict(input="p",other=.1,alpha=.5),"add"),
            ("call_method","multiply",("p",.5),{},"multiply"),
            ("call_function","torch.true_divide",("p",2),dict(rounding_mode=None),"divide"),
            ("call_method","neg",("p",),{},"negate")]:
            actual,attrs,operands=describe(kind,target,args,kw);self.assertEqual(actual,op);self.assertTrue(operands)
    def test_literal_and_two_parent_proofs_bind_values_order_and_attributes(self):
        arrays={"a":np.asarray([.2,.7],dtype="<f8"),"b":np.asarray([.5,.9],dtype="<f8")}
        resolver=lambda value:(value,arrays[value])if type(value)is str and value in arrays else None
        _,_,proof=plan("call_function","torch.add",("a","b"),dict(alpha=.5),resolve_value=resolver,resource_profile=PROFILE)
        self.assertEqual([o["name"]for o in proof["operands"]],["a","b"])
        self.assertEqual(proof["binding"],digest({k:v for k,v in proof.items()if k!="binding"}))
        arrays["b"]=arrays["b"]+.001
        _,_,changed=plan("call_function","torch.add",("a","b"),dict(alpha=.5),resolve_value=resolver,resource_profile=PROFILE)
        self.assertNotEqual(proof["binding"],changed["binding"])
    def test_scalar_rank_zero_and_rank_32_are_bounded(self):
        a=np.ones([1]*32,dtype="<f8")*.3
        out=evaluate([a,2.0],"multiply",{},resource_profile=PROFILE)
        self.assertEqual(out.ndim,32)
        self.assertEqual(evaluate([np.asarray(.2,dtype="<f8")],"negate",{},resource_profile=PROFILE).shape,())
    def test_broadcast_capacity_is_checked_before_result_materialization(self):
        limits=resolve(PROFILE);n=int(limits.constant_elements**(1/3))+1
        while n**3<=limits.constant_elements:n+=1
        self.assertLessEqual(n,limits.axis_size)
        self.assertLessEqual(n*n,limits.constant_elements)
        with self.assertRaisesRegex(ValueError,"capacity"):
            evaluate([np.ones((n,n,1)),np.ones((1,1,n))],"multiply",{},resource_profile=PROFILE)
    def test_wrong_dtype_nonfinite_and_out_of_bound_values_rejected(self):
        for a in[np.asarray([.3],dtype="<f4"),np.asarray([1],dtype="<i4"),np.asarray([True],dtype="bool"),
                  np.asarray([float("inf")]),np.asarray([1025.])]:
            with self.assertRaises(ValueError):evaluate([a],"negate",{},resource_profile=PROFILE)
        for v in[True,None,".3",float("nan"),1025]:
            with self.assertRaises(ValueError):evaluate([np.asarray([.3]),v],"multiply",{},resource_profile=PROFILE)
    def test_zero_small_divisors_and_rounding_rejected(self):
        for v in[0.,1e-5,-1e-5]:
            with self.assertRaisesRegex(ValueError,"divisor"):
                evaluate([np.asarray([.3]),v],"divide",{},resource_profile=PROFILE)
        for mode in["floor","trunc",False]:
            with self.assertRaisesRegex(ValueError,"rounding"):
                describe("call_function","torch.div",("a",2.0),dict(rounding_mode=mode))
    def test_result_and_alpha_budgets_rejected(self):
        with self.assertRaisesRegex(ValueError,"result_limit"):
            evaluate([np.asarray([1024.]),2.0],"multiply",{},resource_profile=PROFILE)
        for alpha in[True,float("inf"),1025,"x"]:
            with self.assertRaises(ValueError):describe("call_method","add",("a",.1),dict(alpha=alpha))
    def test_inplace_out_and_device_calls_stay_rejected(self):
        for target in["add_","mul_","div_","copy_"]:
            with self.assertRaises(ValueError):describe("call_method",target,("a",.1),{})
        with self.assertRaises(ValueError):describe("call_function","torch.add",("a",.1),dict(out="b"))
        with self.assertRaises(ValueError):describe("call_method","add",("a",.1),dict(device="cuda"))
    def test_unbound_dependency_and_scalar_only_proof_rejected(self):
        for args in[("private",.1),(1.,2.)]:
            with self.assertRaises(ValueError):
                plan("call_function","operator.add",args,{},resolve_value=lambda _:None,resource_profile=PROFILE)
    def test_existing_cast_versions_keep_records_and_reject_new_math(self):
        for policy in(POLICY,CAST_POLICY):
            with self.assertRaises(ValueError):unary(np.asarray([.3]),"add",{},policy=policy)
            with self.assertRaisesRegex(ValueError,"not_registered"):
                trace(Model("add",0),[dict(name="x",shape=[2,3])],
                      resource_profile=PROFILE,public_expression_policy=policy,**TRACE_OPTIONS)
    def test_all_fifteen_contexts_preserve_cipher_ops_and_independent_reference(self):
        for operation in("add","subtract","multiply","divide","negate"):
            for context in range(3):
                with self.subTest(operation=operation,context=context):
                    module=Model(operation,context).double().eval();original=_arrays(module)
                    traced=trace(module,[dict(name="x",shape=input_shape(operation,context))],
                        resource_profile=PROFILE,public_expression_policy=ARITHMETIC_POLICY,**TRACE_OPTIONS)
                    attrs=_arrays(traced);frozen=state_identity(attrs,resource_profile=PROFILE)
                    lowered,flow=normalize(traced,attrs,expected_state=frozen,resource_profile=PROFILE,policy=ARITHMETIC_POLICY)
                    check(flow,attrs,expected_binding=flow["binding"],expected_source_inventory=flow["source_inventory"],
                          resource_profile=PROFILE,policy=ARITHMETIC_POLICY)
                    self.assertEqual(state_identity(_arrays(traced),resource_profile=PROFILE),frozen)
                    self.assertEqual(set(_arrays(module)),set(original))
                    self.assertTrue(flow["cipher_operations_preserved"]);self.assertGreaterEqual(len(flow["entries"]),1)
                    size=math.prod(input_shape(operation,context))
                    x=torch.linspace(-.63,.91,size,dtype=torch.float64).reshape(input_shape(operation,context))
                    with torch.no_grad():
                        np.testing.assert_allclose(module(x.clone()).numpy(),lowered(x.clone()).numpy(),rtol=1e-12,atol=1e-12)
    def test_derived_proof_and_source_roots_cannot_be_relabelled(self):
        module=Model("add",1).double()
        traced=trace(module,[dict(name="x",shape=[2,3])],resource_profile=PROFILE,
            public_expression_policy=ARITHMETIC_POLICY,**TRACE_OPTIONS)
        attrs=_arrays(traced)
        lowered,flow=normalize(traced,attrs,expected_state=state_identity(attrs,resource_profile=PROFILE),
             resource_profile=PROFILE,policy=ARITHMETIC_POLICY)
        entry=next(v for v in flow["entries"]if v["plan"]["operation"]=="add")
        self.assertEqual(set(entry["roots"]),{"a","b"})
        damaged=copy.deepcopy(flow);damaged["entries"][0]["roots"]=["a"];damaged.pop("binding");damaged["binding"]=digest(damaged)
        with self.assertRaises(ValueError):check(damaged,attrs,expected_binding=damaged["binding"],
            expected_source_inventory=flow["source_inventory"],resource_profile=PROFILE,policy=ARITHMETIC_POLICY)
    def test_public_state_mutation_still_rejected_before_folding(self):
        class Bad(Model):
            def forward(self,x):return x*self.a.add_(.1)
        with self.assertRaisesRegex(ValueError,"not_registered"):
            trace(Bad("multiply",1),[dict(name="x",shape=[2,3])],
                  resource_profile=PROFILE,public_expression_policy=ARITHMETIC_POLICY,**TRACE_OPTIONS)
    def test_cipher_plus_public_is_not_folded(self):
        class Mixed(Model):
            def forward(self,x):return self.a+x
        module=Mixed("multiply",1)
        traced=trace(module,[dict(name="x",shape=[2,3])],resource_profile=PROFILE,
            public_expression_policy=ARITHMETIC_POLICY,**TRACE_OPTIONS)
        attrs=_arrays(traced)
        result,flow=normalize(traced,attrs,expected_state=state_identity(attrs,resource_profile=PROFILE),
            resource_profile=PROFILE,policy=ARITHMETIC_POLICY)
        self.assertEqual(flow["entries"],[])
        self.assertTrue(any(node.op=="call_function"and node.target is operator.add for node in result.graph.nodes))
    def test_format_dispatch_and_capability_are_explicit(self):
        from pytorch_model_intake import PUBLIC_ARITHMETIC_FORMAT,SOURCE_FORMATS,PACKAGES,REFERENCES,ordered
        from pytorch_public_expression_capture import source_policy
        from application_capabilities import input_sources
        spec=dict(format=PUBLIC_ARITHMETIC_FORMAT)
        self.assertTrue(ordered(spec));self.assertIn(spec["format"],SOURCE_FORMATS)
        self.assertEqual(PACKAGES[spec["format"]],"poseidon-trusted-pytorch-provenance-v25")
        self.assertEqual(REFERENCES[spec["format"]],"poseidon-trusted-pytorch-reference-v25")
        self.assertEqual(source_policy(spec),ARITHMETIC_POLICY)
        feature=input_sources()["trusted_pytorch"]["ordered_closed_public_arithmetic"]
        self.assertEqual(feature["encrypted_status"],"unverified")
