"""Static indexing is compared with actual Torch indexing, not DSL output."""
import copy,unittest
import numpy as np,torch
from pytorch_static_index import plan,axis_selection,encode,decode

class StaticIndexPlanTests(unittest.TestCase):
    def check(self,dims,index):
        p=plan(dims,index,resource_profile="general-tensors-v1")
        x=np.arange(np.prod(dims),dtype=np.float64).reshape(dims)
        value=x
        for step in p["steps"]:
            a=step["attrs"]
            if step["op"]=="slice":
                selectors=[slice(None)]*value.ndim;selectors[a["axis"]]=slice(a["start"],a["stop"],a["step"])
                value=value[tuple(selectors)]
            else:value=np.take(value,a["indices"],axis=a["axis"])
        value=value.reshape(p["result_shape"])
        np.testing.assert_array_equal(value,torch.from_numpy(x)[index].numpy())
        self.assertEqual(tuple(p["result_shape"]),value.shape)
    def test_basic_axes_slices_ellipsis_and_newaxes(self):
        for dims,index in (
            ((2,3,4),(1,slice(None),slice(1,None,2))),
            ((3,4),(None,slice(-2,None),None,slice(None,None,2))),
            ((2,3,4),(Ellipsis,-1)),
            ((4,),slice(-100,100,2)),
            ((2,3),(slice(None),-1)),
            ((2,3),()),
            ((2,3),(Ellipsis,None)),
            ((),(None,Ellipsis))):
            with self.subTest(dims=dims,index=index):self.check(dims,index)
    def test_public_vector_order_duplicates_negatives_and_newaxes(self):
        for dims,index in (((2,4),(slice(None),[3,0,3])),
            ((3,2),([2,0,2],slice(None),None)),
            ((2,3,2),(slice(None),None,[2,0],slice(None))),
            ((2,3,2),(None,slice(None),[-1,0],None,slice(None)))):
            with self.subTest(dims=dims,index=index):self.check(dims,index)
    def test_static_axis_selection_negative_axes(self):
        p=axis_selection([2,4],-1,[3,0,3],kind="index_select",resource_profile="general-tensors-v1")
        self.assertEqual(p["result_shape"],[2,3])
        with self.assertRaisesRegex(ValueError,"nonnegative"):
            axis_selection([2,4],1,[-1],kind="index_select",resource_profile="general-tensors-v1")
    def test_boolean_float_or_custom_selector_rejected_without_coercion(self):
        class Index:
            def __index__(self):raise AssertionError("Untrusted coercion")
        for key in (True,1.,[True],[1.],Index()):
            with self.subTest(key=type(key)),self.assertRaises(ValueError):
                plan([4],key,resource_profile="general-tensors-v1")
    def test_empty_out_of_bounds_negative_step_and_multiple_vectors_rejected(self):
        for dims,index in (((4,),4),((4,),-5),((4,),slice(2,2)),((4,),slice(None,None,-1)),
            ((4,),slice(None,None,0)),((4,),[]),((2,3),([0],[1])),((2,3),(0,[1])),
            ((2,3),(Ellipsis,Ellipsis)),((4,),slice(None,True))):
            with self.subTest(index=index),self.assertRaises(ValueError):plan(dims,index,resource_profile="general-tensors-v1")
    def test_rank_32_is_profile_specific(self):
        self.check((1,)*32,(Ellipsis,))
        with self.assertRaisesRegex(ValueError,"result shape budget"):plan([1]*32,(None,Ellipsis),resource_profile="general-tensors-v1")
        with self.assertRaisesRegex(ValueError,"source shape budget"):plan([1]*5,Ellipsis,resource_profile="legacy-v1")
    def test_selector_roundtrip_and_exact_fields(self):
        for key in (1,None,Ellipsis,slice(-2,None,2),[3,0,-1]):
            self.assertEqual(encode(decode(encode(key))),encode(key))
            bad=copy.deepcopy(encode(key));bad["extra"]=True
            with self.assertRaises(ValueError):decode(bad)
    def test_expansion_shape_budget_checked_before_graph_creation(self):
        with self.assertRaises(ValueError):plan([2],list(range(8192)),resource_profile="general-tensors-v1")
        with self.assertRaises(ValueError):plan([1],tuple([None]*66),resource_profile="general-tensors-v1")

if __name__=="__main__":unittest.main()
