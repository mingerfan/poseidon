import copy,hashlib,unittest
from pathlib import Path
import numpy as np
import torch
from pytorch_public_value_ops import *
COUNTERS=dict(contexts=0,reference_groups=0,context_signatures={})
class Controls(unittest.TestCase):
    def fail(self,fn,code):
        with self.assertRaisesRegex(ValueError,code):fn()
    def context(self,op,context):
        rng=np.random.default_rng(8700+context)
        source_shape=[(6,),(2,3),(2,1,3)][context]if op!="permute"else[(2,3),(2,3,2),(2,1,3,2)][context]
        elements=math.prod(source_shape)
        for probe in range(4):
            a=np.zeros(source_shape)if probe==0 else np.linspace(-1,1,elements).reshape(source_shape)if probe==1 else rng.normal(size=source_shape)if probe==2 else np.full(source_shape,3.)
            tensor=torch.from_numpy(a.copy())
            if op=="reshape":
                target_shape=[(3,2),(1,6),(2,3,1)][context]
                kind,target,args,kw="call_method","view",("parent",list(target_shape)),{}
                expected=tensor.view(target_shape)
            elif op=="permute":
                dimensions=tuple(reversed(range(tensor.ndim)))
                kind,target,args,kw="call_method","permute",("parent",*dimensions),{}
                expected=tensor.permute(dimensions)
            elif op=="slice":
                kind,target,args,kw="call_function","operator.getitem",("parent",(None,Ellipsis,slice(0,3,2))),{}
                expected=tensor[None,...,0:3:2]
            elif op=="bool_cast":
                a=(a>0);tensor=torch.from_numpy(a.copy())
                kind,target,args,kw="call_method","to",("parent",),{"dtype":"torch.float32"if context%2 else"torch.float64"}
                expected=tensor.to(dtype=torch.float32 if context%2 else torch.float64)
            else:
                kind,target,args,kw="call_method","contiguous",("parent",),{}
                expected=tensor.contiguous()
            logical,physical,record=plan(a,kind,target,args,kw,resource_profile="general-tensors-v1")
            np.testing.assert_array_equal(logical,expected.numpy())
            np.testing.assert_array_equal(physical,expected.numpy().astype("<f8"))
            self.assertFalse(logical.flags.writeable);self.assertFalse(physical.flags.writeable)
            check_plan(record,a,kind,target,args,kw,expected_binding=record["binding"])
            COUNTERS["reference_groups"]+=1
        context_signature=digest(dict(kind=kind,target=target,source_shape=list(source_shape),operation=record["operation"],attributes=record["attributes"]))
        signatures=COUNTERS["context_signatures"].setdefault(op,[])
        self.assertNotIn(context_signature,signatures)
        signatures.append(context_signature)
        COUNTERS["contexts"]+=1
    def test_five_operations_three_contexts(self):
        for op in("reshape","permute","slice","bool_cast","identity"):
            for index in range(3):
                with self.subTest(op=op,context=index):self.context(op,index)
    def test_registered_view_spellings(self):
        a=np.arange(6,dtype="<f8").reshape(2,3);t=torch.from_numpy(a)
        cases=[("call_method","reshape",("parent",3,2),{},t.reshape(3,2)),
               ("call_function","torch.reshape",("parent",),{"shape":(3,2)},torch.reshape(t,(3,2))),
               ("call_method","unsqueeze",("parent",-1),{},t.unsqueeze(-1)),
               ("call_function","torch.unsqueeze",("parent",),{"dim":0},torch.unsqueeze(t,0)),
               ("call_method","flatten",("parent",),{"start_dim":0,"end_dim":1},t.flatten(0,1)),
               ("call_function","torch.flatten",("parent",0),{},torch.flatten(t,0)),
               ("call_method","squeeze",("parent",),{},t.squeeze())]
        for kind,target,args,kw,reference in cases:
            logical,_,_=plan(a,kind,target,args,kw)
            np.testing.assert_array_equal(logical,reference.numpy())
    def test_registered_transpose_spellings(self):
        a=np.arange(6,dtype="<f8").reshape(2,3);t=torch.from_numpy(a)
        for kind,target,args,kw,ref in [
            ("call_method","transpose",("parent",),{"dim0":0,"dim1":1},t.transpose(0,1)),
            ("call_function","torch.transpose",("parent",0,-1),{},torch.transpose(t,0,-1)),
            ("call_method","t",("parent",),{},t.t()),
            ("call_function","torch.t",("parent",),{},torch.t(t)),
            ("call_function","builtins.getattr",("parent","T"),{},t.T)]:
            np.testing.assert_array_equal(plan(a,kind,target,args,kw)[0],ref.numpy())
    def test_rank32_and_scalar(self):
        a=np.ones((1,)*32,dtype="<f8")
        self.assertEqual(evaluate(a,"permute",dict(dims=list(reversed(range(32))))).ndim,32)
        scalar=np.array(4,dtype="<f8")
        self.assertEqual(evaluate(scalar,"reshape",dict(shape=[])).shape,())
        self.assertEqual(plan(scalar,"call_method","flatten",("parent",),{})[0].shape,(1,))
    def test_large_profile_is_used(self):
        a=np.arange(1000000,dtype="<f8").reshape(1000,1000)
        logical,physical,record=plan(a,"call_method","view",("parent",[100,100,100]),{})
        self.assertEqual(logical.size,1000000);self.assertEqual(record["logical_output"]["elements"],1000000)
        self.fail(lambda:plan(a,"call_method","view",("parent",[100,100,100]),{},resource_profile="legacy-v1"),"value_capacity")
    def test_medium_profile_boundary(self):
        a=np.ones((256,256),dtype="<f8")
        self.assertEqual(plan(a,"call_method","view",("parent",[256,256]),{},resource_profile="medium-v1")[0].size,65536)
        self.fail(lambda:checked_array(np.ones(65537),resource_profile="medium-v1"),"value_capacity")
    def test_scalar_slice_and_ellipsis(self):
        a=np.arange(6,dtype="<f8").reshape(2,3)
        for key in((1,-1),(Ellipsis,1),(None,1,slice(None)),(slice(-2,None),slice(1,None))):
            logical,_,_=plan(a,"call_function","operator.getitem",("parent",key),{})
            np.testing.assert_array_equal(logical,torch.from_numpy(a)[key].numpy())
    def test_float32_widening_preserves_data(self):
        a=np.array([0.1,-0.25,0.3],dtype="<f4")
        _,physical,record=plan(a,"call_method","view",("parent",[1,3]),{})
        np.testing.assert_array_equal(physical,a.reshape(1,3).astype("<f8"))
        self.assertTrue(record["exact_float64_materialization"])
        self.assertEqual(record["logical_output"]["dtype"],"<f4");self.assertEqual(record["materialized_output"]["dtype"],"<f8")
    def test_integer_geometry_preserves_type(self):
        a=np.array([1,3,2],dtype="<i8")
        value,physical,_=plan(a,"call_method","reshape",("parent",[1,3]),{})
        self.assertEqual(value.dtype,physical.dtype);self.assertEqual(value.dtype.str,"<i8")
    def test_identity_does_not_alias(self):
        a=np.arange(3,dtype="<f8");value=evaluate(a,"identity",{})
        a[:]=8;np.testing.assert_array_equal(value,[0,1,2])
    def test_plan_metadata_no_alias(self):
        a=np.arange(6,dtype="<f8");shape=[2,3]
        _,_,record=plan(a,"call_method","view",("parent",shape),{})
        shape[0]=3;self.assertEqual(record["attributes"]["shape"],[2,3])
    def test_plan_content_and_dtype_bindings(self):
        a=np.arange(3,dtype="<f8");_,_,record=plan(a,"call_method","view",("parent",[1,3]),{})
        changed=a.copy();changed[0]=4
        self.fail(lambda:check_plan(record,changed,"call_method","view",("parent",[1,3]),{},expected_binding=record["binding"]),"plan_changed")
        self.fail(lambda:check_plan(record,a.astype("<f4"),"call_method","view",("parent",[1,3]),{},expected_binding=record["binding"]),"plan_changed")
    def test_plan_external_authority(self):
        a=np.arange(3,dtype="<f8");_,_,record=plan(a,"call_method","view",("parent",[1,3]),{})
        binding=record["binding"];record["attributes"]["shape"]=[3,1]
        self.fail(lambda:check_plan(record,a,"call_method","view",("parent",[1,3]),{},expected_binding=binding),"plan_binding")
    def test_wrong_profile_does_not_share_plan(self):
        a=np.arange(3,dtype="<f8");_,_,record=plan(a,"call_method","view",("parent",[1,3]),{})
        self.fail(lambda:check_plan(record,a,"call_method","view",("parent",[1,3]),{},expected_binding=record["binding"],resource_profile="medium-v1"),"plan_changed")
    def test_resource_rank_rejected(self):
        self.fail(lambda:checked_array(np.ones((1,)*5),resource_profile="legacy-v1"),"value_capacity")
    def test_zero_element_and_nonfinite_rejected(self):
        for a,code in ((np.empty((0,3)),"value_capacity"),(np.array([np.nan]),"nonfinite"),(np.array([np.inf]),"nonfinite")):
            self.fail(lambda:evaluate(a,"identity",{}),code)
    def test_unsupported_dtype_and_subclass(self):
        for a in(np.array([1],dtype="<i2"),np.array(["text"]),np.array([1],dtype=object)):
            self.fail(lambda:checked_array(a,resource_profile="general-tensors-v1"),"array_type")
        class Derived(np.ndarray):pass
        self.fail(lambda:checked_array(np.ones(1).view(Derived),resource_profile="general-tensors-v1"),"array_type")
    def test_invalid_reshape(self):
        a=np.arange(6,dtype="<f8")
        for dims in([4,2],[-1,-1],[0,6],[True,6]):
            self.fail(lambda:evaluate(a,"reshape",dict(shape=dims)),"shape")
    def test_inferred_axis_bound_rejected(self):
        a=np.ones((256,256))
        self.fail(lambda:evaluate(a,"reshape",dict(shape=[-1]),resource_profile="legacy-v1"),"value_capacity")
        b=np.ones((256,256))
        self.fail(lambda:evaluate(b,"reshape",dict(shape=[-1]),resource_profile="medium-v1"),"shape_capacity")
    def test_duplicate_or_bad_permutation(self):
        for dims in([0,0],[0,2],[True,1]):
            self.fail(lambda:evaluate(np.ones((2,3)),"permute",dict(dims=dims)),"permutation|axis")
    def test_negative_or_zero_step_rejected(self):
        for step in(0,-1):
            self.fail(lambda:plan(np.arange(3,dtype="<f8"),"call_function","operator.getitem",("parent",slice(None,None,step)),{}),"slice_step")
    def test_empty_slice_rejected(self):
        self.fail(lambda:plan(np.arange(3,dtype="<f8"),"call_function","operator.getitem",("parent",slice(1,1)),{}),"value_capacity")
    def test_bool_and_dynamic_index_rejected(self):
        for key in(True,[0,1],"private"):
            self.fail(lambda:plan(np.arange(3,dtype="<f8"),"call_function","operator.getitem",("parent",key),{}),"static_slice")
    def test_ellipsis_count_and_axis_capacity(self):
        for key in((Ellipsis,Ellipsis),(0,0)):
            self.fail(lambda:plan(np.arange(3,dtype="<f8"),"call_function","operator.getitem",("parent",key),{}),"slice")
    def test_non_boolean_cast_not_silent(self):
        self.fail(lambda:plan(np.ones(3),"call_method","float",("parent",),{}),"bool_cast")
    def test_cast_arguments_and_dtype(self):
        for args,kwargs in((("parent","torch.int64"),{}),(("parent",),{"dtype":"torch.float64","device":"cuda"}),(("parent","torch.float64"),{"dtype":"torch.float64"})):
            self.fail(lambda:plan(np.array([True]),"call_method","to",args,kwargs),"cast")
    def test_unknown_op(self):
        self.fail(lambda:evaluate(np.ones(1),"square",{}),"operation")
    def test_arbitrary_object_not_compared_or_executed(self):
        class Evil:
            def __eq__(self,other):raise AssertionError("custom equality executed")
        self.fail(lambda:describe("call_function","builtins.getattr",("parent",Evil()),{},parent_shape=[1]),"attribute")
        self.fail(lambda:describe("call_method","flatten",("parent",Evil()),{},parent_shape=[]),"flatten_axis")
    def test_duplicate_arguments(self):
        self.fail(lambda:describe("call_function","torch.reshape",("parent",[1]),{"shape":[1]},parent_shape=[1]),"duplicate")
    def test_transpose_rank(self):
        self.fail(lambda:describe("call_method","t",("parent",),{},parent_shape=[1,1,1]),"matrix_transpose")
    def test_axis_bools(self):
        self.fail(lambda:describe("call_method","unsqueeze",("parent",True),{},parent_shape=[1]),"axis")

if __name__=="__main__":unittest.main()
