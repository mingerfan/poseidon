"""Independent closed constructor math, conversion and preallocation boundaries."""
import unittest,copy
from unittest.mock import patch
import numpy as np
import torch
from pytorch_public_constants import plan,OPERATIONS
from pytorch_static_trace import ALIAS_POLICY
from model_resources import resolve

class PublicConstantTests(unittest.TestCase):
    def construct(self,operation,shape=(2,3),dtype=None,fill=.375,parent_dtype="float64",extra=None):
        kind="call_method" if operation.startswith("new_") else "call_function"
        target=operation if kind=="call_method" else "torch."+operation
        args=["x"]
        if operation.startswith("new_"):args.append(tuple(shape))
        if operation in ("full_like","new_full"):args.append(fill)
        kwargs={} if dtype is None else dict(dtype="torch."+dtype)
        kwargs.update(extra or {})
        return plan(kind,target,args,kwargs,parent_name="x",
            parent_metadata=dict(policy=ALIAS_POLICY,shape=list(shape),dtype=parent_dtype))
    def test_six_operations_three_shapes_against_pinned_torch(self):
        for op in OPERATIONS:
            for shape in ((4,),(2,3),(1,2,1,3)):
                with self.subTest(operation=op,shape=shape):
                    a,p,record=self.construct(op,shape)
                    x=torch.empty(shape,dtype=torch.float64)
                    actual=(getattr(x,op)(shape,.375) if op=="new_full" else getattr(x,op)(shape)
                        if op.startswith("new_") else getattr(torch,op)(x,.375)
                        if op=="full_like" else getattr(torch,op)(x))
                    np.testing.assert_array_equal(a,actual.numpy())
                    self.assertEqual(record["dependency"],"static_shape_and_declared_dtype_only")
                    self.assertFalse(a.flags.writeable);self.assertFalse(p.flags.writeable)
                    self.assertFalse(np.shares_memory(a,p))
    def test_five_logical_dtypes_have_exact_torch_conversion(self):
        for op in OPERATIONS:
            for dtype in ("float32","float64","int32","int64","bool"):
                for fill in (-1.9,.125000017,False):
                    with self.subTest(operation=op,dtype=dtype,fill=fill):
                        a,p,proof=self.construct(op,dtype=dtype,fill=fill)
                        x=torch.empty((2,3),dtype=torch.float64);target=getattr(torch,dtype)
                        actual=(getattr(x,op)((2,3),fill,dtype=target) if op=="new_full" else
                            getattr(x,op)((2,3),dtype=target) if op.startswith("new_") else
                            getattr(torch,op)(x,fill,dtype=target) if op=="full_like" else
                            getattr(torch,op)(x,dtype=target))
                        np.testing.assert_array_equal(a,actual.numpy())
                        np.testing.assert_array_equal(p,actual.numpy().astype(np.float64))
    def test_default_dtype_inherits_declared_public_source(self):
        for dtype in ("float32","float64","int32","int64","bool"):
            a,_,_=self.construct("full_like",parent_dtype=dtype,fill=.75)
            self.assertEqual(str(a.dtype),dtype)
    def test_values_do_not_depend_on_input_and_each_call_copies(self):
        a,p,_=self.construct("ones_like");b,q,_=self.construct("ones_like")
        self.assertFalse(np.shares_memory(a,b));self.assertFalse(np.shares_memory(p,q))
    def test_preallocation_rank_axis_and_elements(self):
        limits=resolve("general-tensors-v1")
        bad=[(1,)*(limits.rank+1),(limits.axis_size+1,),(1001,1001),(),(0,),(-1,),(True,)]
        # Scalar rank zero is intentionally supported.
        bad.remove(())
        for shape in bad:
            with self.subTest(shape=shape),patch("pytorch_public_constants.np.full",side_effect=AssertionError("allocated")):
                with self.assertRaises(ValueError):self.construct("new_full",shape)
        a,_,_=self.construct("new_ones",(1,)*32);self.assertEqual(a.ndim,32)
        a,_,_=self.construct("new_full",());self.assertEqual(a.shape,())
    def test_nonfinite_overflow_and_dynamic_fills_reject_before_allocation(self):
        for dtype,fill in [(None,float("nan")),(None,float("inf")),("float32",1e39),
                           ("int32",2**31),("int64",2**63),(None,["x"]),(None,"x")]:
            with self.subTest(dtype=dtype,fill=fill),patch("pytorch_public_constants.np.full",side_effect=AssertionError("allocated")):
                with self.assertRaises(ValueError):self.construct("full_like",dtype=dtype,fill=fill)
    def test_cpu_and_side_effect_options_are_closed(self):
        for kwargs in (dict(device="cuda"),dict(device="meta"),dict(requires_grad=True),
                       dict(out="x"),dict(pin_memory=True),dict(layout="strided"),
                       dict(memory_format="channels_last"),dict(dtype="torch.complex64")):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.construct("new_full" if "pin_memory"in kwargs else"full_like",extra=kwargs)
    def test_named_input_and_size_duplicate_reject(self):
        meta=dict(policy=ALIAS_POLICY,shape=[2],dtype="float64")
        a,_,_=plan("call_function","torch.ones_like",[],dict(input="x",device="cpu",requires_grad=False),
                  parent_name="x",parent_metadata=meta)
        np.testing.assert_array_equal(a,[1.,1.])
        with self.assertRaises(ValueError):
            plan("call_function","torch.ones_like",["x"],dict(input="x"),parent_name="x",parent_metadata=meta)

    def test_global_element_limit_is_checked_before_allocation(self):
        meta=dict(policy=ALIAS_POLICY,shape=[2,3],dtype="float64")
        for remaining in (-1,0,5,True):
            with self.subTest(remaining=remaining),patch("pytorch_public_constants.np.full",side_effect=AssertionError("allocated")):
                with self.assertRaises(ValueError):
                    plan("call_function","torch.ones_like",["x"],{},parent_name="x",
                         parent_metadata=meta,remaining_elements=remaining)
    def test_new_shape_is_independent_of_source_shape_and_dtype_source_remains_bound(self):
        meta=dict(policy=ALIAS_POLICY,shape=[2,3],dtype="float32")
        for shape in ((),(1,),(1,2,1)):
            a,p,proof=plan("call_method","new_full",["x",shape,.125000017],{},
                parent_name="x",parent_metadata=meta)
            actual=torch.empty((2,3),dtype=torch.float32).new_full(shape,.125000017)
            np.testing.assert_array_equal(a,actual.numpy())
            np.testing.assert_array_equal(p,actual.numpy().astype(np.float64))
            self.assertEqual(proof["source_metadata"]["shape"],[2,3])
