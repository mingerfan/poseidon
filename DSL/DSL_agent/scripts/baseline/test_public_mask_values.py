"""Independent public mask math, PyTorch promotion and resource boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pytorch_public_mask_values import public_plan,arguments,result_shape
from pytorch_public_value_ops import MASK_POLICY
from pytorch_public_expression_flow import Ref

class PublicMaskValues(unittest.TestCase):
    def select(self,kind,target,args,values,kwargs=None,**options):
        def resolve(value):
            return (value.name,values[value.name])if type(value)is Ref and value.name in values else None
        return public_plan(kind,target,args,kwargs or{},resolve_value=resolve,**options)
    def test_public_where_promotions(self):
        types=(np.bool_,np.int32,np.int64,np.float32,np.float64)
        for adtype in types:
            for bdtype in types:
                for ashape,bshape in(((2,),()),((),(2,)),((2,),(2,))):
                    with self.subTest(a=str(adtype),b=str(bdtype),ashape=ashape,bshape=bshape):
                        a=np.array(1,dtype=adtype)if not ashape else np.array([1,0],dtype=adtype)
                        bb=np.array(-1 if bdtype!=np.bool_ else True,dtype=bdtype)if not bshape else np.array([0,1],dtype=bdtype)
                        values={"m":np.array([True,False]),"a":a,"b":bb}
                        logical,physical,proof=self.select("call_function","torch.where",(Ref("m"),Ref("a"),Ref("b")),values)
                        expected=torch.where(torch.from_numpy(values["m"]),torch.from_numpy(a),torch.from_numpy(bb)).numpy()
                        self.assertEqual(logical.dtype,expected.dtype);np.testing.assert_array_equal(logical,expected)
                        self.assertFalse(logical.flags.writeable);self.assertFalse(physical.flags.writeable)
                        self.assertFalse(proof["encrypted_condition"])
    def test_weak_scalar_promotions(self):
        for dtype in(np.bool_,np.int32,np.int64,np.float32,np.float64):
            for value in(False,2,-.125,.125000017):
                for swap in(False,True):
                    with self.subTest(dtype=str(dtype),value=value,swap=swap):
                        a=np.array([0,1],dtype=dtype);mask=np.array([True,False])
                        operands=(value,Ref("a"))if swap else(Ref("a"),value)
                        logical,_,_=self.select("call_function","torch.where",(Ref("m"),)+operands,{"m":mask,"a":a})
                        tensor=torch.from_numpy(a)
                        expected=torch.where(torch.from_numpy(mask),value if swap else tensor,tensor if swap else value).numpy()
                        self.assertEqual(logical.dtype,expected.dtype);np.testing.assert_array_equal(logical,expected)
    def test_scalar_only_and_method(self):
        values={"m":np.array([[True],[False]]),"a":np.array([[.1,-.2]],dtype="<f8")}
        logical,_,_=self.select("call_method","where",(Ref("a"),Ref("m"),.375),values)
        expected=torch.from_numpy(values["a"]).where(torch.from_numpy(values["m"]),.375).numpy()
        np.testing.assert_array_equal(logical,expected)
        logical,_,_=self.select("call_function","torch.where",(Ref("m"),.125,False),values)
        expected=torch.where(torch.from_numpy(values["m"]),.125,False).numpy()
        self.assertEqual(logical.dtype,expected.dtype);np.testing.assert_array_equal(logical,expected)
    def test_masked_fill_dtypes_and_broadcast(self):
        for dtype in(np.bool_,np.int32,np.int64,np.float32,np.float64):
            values={"m":np.array([[True],[False]]),"a":np.array([[1,0]],dtype=dtype)}
            for value in(-.125,0,1,True):
                logical,_,_=self.select("call_method","masked_fill",(Ref("a"),Ref("m"),value),values)
                expected=torch.from_numpy(values["a"]).masked_fill(torch.from_numpy(values["m"]),value).numpy()
                self.assertEqual(logical.dtype,expected.dtype);np.testing.assert_array_equal(logical,expected)
    def test_public_scalar_tensor_fill(self):
        values={"m":np.array([True,False]),"a":np.array([1.,2.]),"f":np.array(.125,dtype="<f4")}
        logical,_,_=self.select("call_method","masked_fill",(Ref("a"),Ref("m"),Ref("f")),values)
        expected=torch.from_numpy(values["a"]).masked_fill(torch.from_numpy(values["m"]),torch.from_numpy(values["f"])).numpy()
        np.testing.assert_array_equal(logical,expected)
    def test_finite_mask_and_fill_rejections(self):
        for mask in(np.array([1,0],dtype="<i8"),np.array([1.,0.]),True,Ref("private")):
            args=(mask if type(mask)is not np.ndarray else Ref("m"),Ref("a"),0.)
            with self.subTest(mask=str(type(mask))),self.assertRaises(ValueError):
                self.select("call_function","torch.where",args,{"m":mask,"a":np.array([1.,2.])})
        for fill in(float("nan"),float("inf"),2**100,object(),[1.]):
            with self.subTest(fill=str(type(fill))),self.assertRaises(ValueError):
                self.select("call_method","masked_fill",(Ref("a"),Ref("m"),fill),
                    {"a":np.array([1.,2.]),"m":np.array([True,False])})
    def test_signatures_and_inplace_public_rejected(self):
        for args,kwargs in(((Ref("m"),Ref("a"),0.),{"input":Ref("a")}),((Ref("m"),),{}),
                            ((Ref("m"),Ref("a"),0.),{"out":Ref("a")})):
            with self.assertRaises(ValueError):arguments("call_function","torch.where",args,kwargs)
        with self.assertRaisesRegex(ValueError,"public_mask_public_inplace"):
            self.select("call_method","masked_fill_",(Ref("a"),Ref("m"),0.),{"a":np.array([1.,2.]),"m":np.array([True,False])})
    def test_output_budget_checked_before_selection(self):
        values={"m":np.ones((1024,1),dtype=np.bool_),"a":np.ones((1,2048),dtype="<f8")}
        with patch("pytorch_public_mask_values.np.where",side_effect=AssertionError("allocated")),self.assertRaisesRegex(ValueError,"budget"):
            self.select("call_function","torch.where",(Ref("m"),Ref("a"),0.),values)
        values={"m":np.array([True,False]),"a":np.array([1.,2.])}
        with patch("pytorch_public_mask_values.np.where",side_effect=AssertionError("allocated")),self.assertRaisesRegex(ValueError,"total_constant_budget"):
            self.select("call_function","torch.where",(Ref("m"),Ref("a"),0.),values,remaining=1)
    def test_rank_and_inplace_shape_boundaries(self):
        self.assertEqual(result_shape([],[],[],resource_profile="general-tensors-v1"),[])
        self.assertEqual(result_shape([1]*32,[1]*32,[],resource_profile="general-tensors-v1"),[1]*32)
        with self.assertRaises(ValueError):result_shape([1]*33,[],[],resource_profile="general-tensors-v1")
        with self.assertRaisesRegex(ValueError,"inplace_shape"):
            result_shape([2,2],[1,2],[],resource_profile="general-tensors-v1",inplace=True)
        with self.assertRaisesRegex(ValueError,"broadcast"):
            result_shape([2,3],[2,2],[],resource_profile="general-tensors-v1")
    def test_overflow_and_nonfinite_public_result(self):
        values={"m":np.array([True,False]),"a":np.array([1.,2.],dtype="<f4")}
        with self.assertRaisesRegex(ValueError,"fill_range"):
            self.select("call_method","masked_fill",(Ref("a"),Ref("m"),1e39),values)
if __name__=="__main__":unittest.main()
