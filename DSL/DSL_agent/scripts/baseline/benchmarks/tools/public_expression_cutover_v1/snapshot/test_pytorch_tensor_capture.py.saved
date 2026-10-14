"""PyTorch tensor mappings checked against independent numerical references."""
import unittest
import numpy as np
import torch
from pytorch_model_capture import capture
from benchmark_graph import samples
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference

class Reduction(torch.nn.Module):
    def __init__(self,op,variant):
        super().__init__();self.op=op;self.variant=variant
    def forward(self,x):
        if self.variant==0:
            return torch.sum(input=x,dim=(0,2),keepdim=True) if self.op=="sum" else torch.mean(input=x,dim=(0,2),keepdim=True)
        if self.variant==1:
            return x.sum(-1,False) if self.op=="sum" else x.mean(-1,False)
        return (torch.sum(x) if self.op=="sum" else torch.mean(x)).reshape(1)

class Join(torch.nn.Module):
    def __init__(self,op,variant):
        super().__init__();self.op=op;self.variant=variant
    def forward(self,x,y):
        if self.op=="stack":
            if self.variant==0:return torch.stack(tensors=(x,y),dim=-1)
            if self.variant==1:return torch.stack([x,y,x-y],1)
            return torch.stack((x+y,),0)
        if self.variant==0:return torch.cat(tensors=(x,y),dim=-1)
        if self.variant==1:return torch.concat([x,y,x-y],1)
        return torch.concatenate((x+y,),0)

class TensorCaptureTests(unittest.TestCase):
    def compare(self,module,inputs):
        module=module.double().eval()
        original,lowered,binding=capture(module,inputs,["output"])
        repeated=capture(module,inputs,["output"])
        self.assertEqual((original,lowered,binding),repeated)
        for case in samples(lowered,16):
            expected=module(*(torch.from_numpy(case[item["name"]]) for item in inputs)).detach().numpy()
            for evaluator in (math_reference,torch_reference):
                actual=evaluator(lowered,case)["output"]
                self.assertEqual(actual.shape,expected.shape)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    def test_sum_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):
                self.compare(Reduction("sum",variant),[dict(name="x",shape=[2,3,4])])
    def test_mean_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):
                self.compare(Reduction("mean",variant),[dict(name="x",shape=[2,3,4])])
    def test_concat_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):
                self.compare(Join("concat",variant),[dict(name=n,shape=[2,3]) for n in ("x","y")])
    def test_stack_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):
                self.compare(Join("stack",variant),[dict(name=n,shape=[2,3]) for n in ("x","y")])
    def test_scalar_stack_sum_mean(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack([x.sum(),x.mean()],dim=0)
        self.compare(Model(),[dict(name="x",shape=[2,3])])
    def test_scalar_stack_negative_axis(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack([x.sum(),x.mean(),x.sum()-x.mean()],dim=-1)
        self.compare(Model(),[dict(name="x",shape=[1,2,3,4])])
    def test_scalar_stack_arithmetic_alias(self):
        class Model(torch.nn.Module):
            def forward(self,x,y):
                a=x.mean()+y.sum()
                return torch.stack(tensors=(a,a*a,a-0.125),dim=0)
        self.compare(Model(),[dict(name="x",shape=[2,3]),dict(name="y",shape=[3])])
    def test_scalar_stack_internal_name_does_not_shadow_module(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.stack_scalar_stack=torch.nn.Identity()
            def forward(self,x):
                stacked=torch.stack([x.sum(),x.mean()])
                return self.stack_scalar_stack(stacked)
        self.compare(Model(),[dict(name="x",shape=[2,3])])
    def test_scalar_output_requires_explicit_shape(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.sum()
        with self.assertRaisesRegex(ValueError,"Scalar outputs require reshape"):
            capture(Model(),[dict(name="x",shape=[2,3])],["output"])
    def test_float32_reduction_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.mean(1,dtype=torch.float32)
        with self.assertRaisesRegex(ValueError,"dtype conversion"):
            capture(Model(),[dict(name="x",shape=[2,3])],["output"])
    def test_explicit_float64_reduction(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.sum(dim=1,keepdim=True,dtype=torch.float64)
        self.compare(Model(),[dict(name="x",shape=[2,3])])
    def test_empty_axis_tuple_matches_torch_all_axes(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.mean(dim=(),keepdim=True)
        self.compare(Model(),[dict(name="x",shape=[2,3])])
    def test_duplicate_axis_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.sum(dim=(1,-1),keepdim=True)
        with self.assertRaises(Exception):
            capture(Model(),[dict(name="x",shape=[2,3])],["output"])
    def test_output_capacity_preserved(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack([x,x,x,x],0)
        with self.assertRaisesRegex(ValueError,"budget"):
            capture(Model(),[dict(name="x",shape=[128])],["output"])
if __name__=="__main__":unittest.main()
