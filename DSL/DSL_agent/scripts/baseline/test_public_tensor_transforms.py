"""Public-state transform math, metadata and historical-policy controls."""
import copy
import unittest
from unittest import mock
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_public_value_ops import (POLICY, CAST_POLICY, ARITHMETIC_POLICY, TRANSFORM_POLICY,
    plan as public_plan)
from pytorch_public_tensor_transforms import plan, registered
from pytorch_public_expression_flow import normalize, check, _arrays, state_identity
from pytorch_public_arithmetic_capture import TRACE_OPTIONS
from pytorch_static_trace import trace

PROFILE = "general-tensors-v1"

class PublicTransformMathTests(unittest.TestCase):
    def check(self, value, op, args, kwargs, reference, kind="call_method"):
        before=value.copy()
        target=op if kind=="call_method"else"torch."+op
        logical,physical,proof=plan(value,kind,target,("public",*args),kwargs,resource_profile=PROFILE)
        np.testing.assert_array_equal(logical,reference)
        np.testing.assert_array_equal(value,before)
        self.assertFalse(logical.flags.writeable)
        self.assertFalse(np.shares_memory(value,logical))
        self.assertEqual(proof["binding"],digest({k:v for k,v in proof.items()if k!="binding"}))
        np.testing.assert_array_equal(physical,logical.astype("<f8")if value.dtype.str=="<f4"else logical)
        self.assertEqual(logical.dtype,value.dtype)
        return proof

    def test_three_contexts_per_operation(self):
        for shape in ((3,), (2,3), (1,2,2)):
            v=np.arange(np.prod(shape),dtype="<f8").reshape(shape)-.37
            t=torch.from_numpy(v.copy())
            repeats=(2,)+(1,)*(len(shape)-1)
            with self.subTest(shape=shape,op="repeat"):
                self.check(v,"repeat",(repeats,),{},t.repeat(repeats).numpy())
            with self.subTest(shape=shape,op="tile"):
                self.check(v,"tile",((2,),),{},torch.tile(t,(2,)).numpy(),kind="call_function")
            with self.subTest(shape=shape,op="repeat_interleave"):
                self.check(v,"repeat_interleave",(2,),dict(dim=-1,output_size=shape[-1]*2),
                    torch.repeat_interleave(t,2,dim=-1,output_size=shape[-1]*2).numpy())
            with self.subTest(shape=shape,op="flip"):
                self.check(v,"flip",((-1,),),{},torch.flip(t,(-1,)).numpy(),kind="call_function")
            with self.subTest(shape=shape,op="roll"):
                self.check(v,"roll",(-3,),dict(dims=-1),t.roll(-3,-1).numpy())

    def test_dtype_retained_before_exact_materialization(self):
        for dtype in ("<f8","<f4","<i4","<i8","|b1"):
            v=np.array([[1,0,-2],[3,-4,0]],dtype=dtype);t=torch.from_numpy(v.copy())
            with self.subTest(dtype=dtype):
                self.check(v,"repeat",((2,1),),{},t.repeat(2,1).numpy())
                self.check(v,"tile",((2,),),{},t.tile((2,)).numpy())
                self.check(v,"repeat_interleave",(2,),{"dim":0},t.repeat_interleave(2,dim=0).numpy())
                self.check(v,"flip",((0,-1),),{},t.flip((0,-1)).numpy())
                self.check(v,"roll",((1,-2),),dict(dims=(0,-1)),t.roll((1,-2),(0,-1)).numpy())

    def test_flattened_interleave_and_roll(self):
        v=np.array([[.1,-.2,.3],[.4,-.5,.6]],dtype="<f8");t=torch.from_numpy(v.copy())
        self.check(v,"repeat_interleave",(3,),{},t.repeat_interleave(3).numpy())
        self.check(v,"roll",(10**30,),{},t.roll((10**30)%v.size).numpy())

    def test_scalar_public_state_and_identity_copy(self):
        v=np.array(-.31,dtype="<f8");t=torch.from_numpy(v.copy())
        self.check(v,"repeat",((3,),),{},t.repeat(3).numpy())
        self.check(v,"tile",((),),{},torch.tile(t,()).numpy())
        self.check(v,"repeat_interleave",(2,),{},t.repeat_interleave(2).numpy())
        self.check(v,"flip",((),),{},t.flip(()).numpy())
        self.check(v,"roll",(0,),{},t.roll(0).numpy())

    def test_repeated_roll_axes_follow_framework_semantics(self):
        v=np.arange(6,dtype="<f8").reshape(2,3);t=torch.from_numpy(v.copy())
        self.check(v,"roll",((1,2,-4),),dict(dims=(-1,-1,0)),t.roll((1,2,-4),(-1,-1,0)).numpy())

    def test_constant_budget_is_distinct_from_cipher_tensor_budget(self):
        v=np.ones((2,32768),dtype="<f8")
        logical,_,p=plan(v,"call_method","repeat",("public",2,1),{},resource_profile=PROFILE)
        self.assertEqual(logical.size,131072)
        self.assertEqual(p["allocation_bound_elements"],131072)

    def test_rank_boundary(self):
        v=np.ones((1,)*32,dtype="<f8")
        self.check(v,"roll",(5,),dict(dims=-1),v)
        with self.assertRaisesRegex(ValueError,"public_transform_repeats"):
            plan(v,"call_method","repeat",("public",*(1,)*33),{},resource_profile=PROFILE)

    def test_allocation_rejected_before_numpy_work(self):
        v=np.array([1.,2.])
        with mock.patch("pytorch_public_tensor_transforms.np.tile")as allocate:
            with self.assertRaisesRegex(ValueError,"public_transform_output_capacity"):
                plan(v,"call_method","repeat",("public",10**12),{},resource_profile=PROFILE)
            allocate.assert_not_called()

    def test_invalid_counts_are_closed(self):
        for count in (0,-1,True,2.,np.array([1,2]),[1,2]):
            with self.subTest(count_type=type(count).__name__),self.assertRaises(ValueError):
                plan(np.array([.1,.2]),"call_method","repeat_interleave",("public",count),{},resource_profile=PROFILE)

    def test_invalid_axes_sizes_and_signatures(self):
        v=np.ones((2,3))
        for op,args,kwargs in (
            ("flip",((0,-2),),{}),("flip",((2,),),{}),
            ("roll",((1,2),),dict(dims=(0,))),("roll",(1,),dict(dims=True)),
            ("repeat",((0,1),),{}),("repeat",((2,1),),dict(repeats=(2,1))),
            ("repeat_interleave",(2,),dict(dim=0,output_size=7)),
            ("repeat_interleave",(2,),dict(dim=0,output_size=True))):
            with self.subTest(op=op,args=args),self.assertRaises(ValueError):
                plan(v,"call_method",op,("public",*args),kwargs,resource_profile=PROFILE)

    def test_nonfinite_and_unsafe_dtype(self):
        for value in (np.array([np.nan]),np.array([np.inf]),np.array(["bad"],dtype=object)):
            with self.assertRaises(ValueError):
                plan(value,"call_method","roll",("public",1),{},resource_profile=PROFILE)

    def test_legacy_policies_keep_rejection(self):
        for policy in (POLICY,CAST_POLICY,ARITHMETIC_POLICY):
            with self.subTest(policy=policy),self.assertRaisesRegex(ValueError,"not_registered"):
                public_plan(np.array([.1,.2]),"call_method","repeat",("public",2),{},
                            resource_profile=PROFILE,policy=policy)
        self.assertEqual(public_plan(np.array([.1,.2]),"call_method","repeat",("public",2),{},
            resource_profile=PROFILE,policy=TRANSFORM_POLICY)[0].tolist(),[.1,.2,.1,.2])

    def test_only_registered_targets(self):
        self.assertTrue(registered("call_function","torch.flip"))
        for kind,target in (("call_method","eval"),("call_function","numpy.tile"),
                            ("call_module","repeat"),("call_function","torch.repeat")):
            self.assertFalse(registered(kind,target))


class PublicTransformFlowTests(unittest.TestCase):
    class Model(torch.nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            self.register_buffer("public",torch.tensor([[.19,-.31]],dtype=torch.float64))
            self.register_buffer("mask",torch.tensor([[True,False]],dtype=torch.bool))
        def forward(self,x):
            if self.kind=="repeat":return x+self.public.repeat(2,1)
            if self.kind=="tile":return x*self.public.tile((2,1))
            if self.kind=="repeat_interleave":return x+self.public.repeat_interleave(2,dim=0)
            if self.kind=="flip":return x*self.mask.flip((-1,)).double()
            return x+self.public.roll(1,-1)

    def prepared(self,kind):
        module=self.Model(kind).eval()
        inputs=[dict(name="x",shape=[2,2]if kind in("repeat","tile","repeat_interleave")else[1,2])]
        traced=trace(module,inputs,resource_profile=PROFILE,**TRACE_OPTIONS,
                     public_expression_policy=TRANSFORM_POLICY)
        arrays=_arrays(traced)
        result,flow=normalize(traced,arrays,expected_state=state_identity(arrays,resource_profile=PROFILE),
                              resource_profile=PROFILE,policy=TRANSFORM_POLICY)
        return module,traced,arrays,result,flow,inputs

    def test_public_transform_flow_matches_original_framework(self):
        for kind in ("repeat","tile","repeat_interleave","flip","roll"):
            with self.subTest(kind=kind):
                module,traced,arrays,result,flow,inputs=self.prepared(kind)
                x=torch.arange(np.prod(inputs[0]["shape"]),dtype=torch.float64).reshape(inputs[0]["shape"])-.23
                torch.testing.assert_close(module(x),result(x),rtol=0,atol=0)
                self.assertEqual(flow["format"],"poseidon-public-expression-flow-v4")
                self.assertTrue(flow["cipher_operations_preserved"])
                self.assertTrue(flow["original_state_preserved"])
                check(flow,arrays,expected_binding=flow["binding"],expected_source_inventory=flow["source_inventory"],
                      resource_profile=PROFILE,policy=TRANSFORM_POLICY)

    def test_source_order_and_roots_retained(self):
        _,_,_,_,flow,_=self.prepared("flip")
        self.assertEqual(flow["entries"][0]["source_target"],"flip")
        self.assertEqual(flow["entries"][0]["roots"],["mask"])
        cipher=[n["name"]for n in flow["source_inventory"]if n["op"]=="placeholder"]
        self.assertEqual(cipher,[n["name"]for n in flow["normalized_inventory"]if n["op"]=="placeholder"])

    def test_forged_plan_rejected_even_after_rebinding(self):
        _,_,arrays,_,flow,_=self.prepared("repeat")
        bad=copy.deepcopy(flow);bad["entries"][0]["plan"]["attributes"]["repeats"]=[1,2]
        bad["binding"]=digest({k:v for k,v in bad.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"record_changed"):
            check(bad,arrays,expected_binding=bad["binding"],expected_source_inventory=bad["source_inventory"],
                  resource_profile=PROFILE,policy=TRANSFORM_POLICY)

    def test_runtime_count_tensor_remains_rejected(self):
        class Dynamic(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("public",torch.tensor([.2],dtype=torch.float64))
            def forward(self,x):return self.public.repeat_interleave(x)
        with self.assertRaises(ValueError):
            trace(Dynamic().eval(),[dict(name="x",shape=[1])],resource_profile=PROFILE,
                  **TRACE_OPTIONS,public_expression_policy=TRANSFORM_POLICY)

    def test_historical_public_policy_remains_closed(self):
        with self.assertRaisesRegex(ValueError,"operation_not_registered"):
            trace(self.Model("repeat").eval(),[dict(name="x",shape=[2,2])],resource_profile=PROFILE,
                  **TRACE_OPTIONS,public_expression_policy=ARITHMETIC_POLICY)

if __name__=="__main__":unittest.main()
