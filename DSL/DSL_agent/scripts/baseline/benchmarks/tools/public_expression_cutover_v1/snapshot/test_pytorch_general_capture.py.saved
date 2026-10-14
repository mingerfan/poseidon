"""New input mappings are checked against original normalized modules."""
import unittest
import numpy as np
import torch
from benchmark_graph import samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as independent
from pytorch_model_capture import capture
from pytorch_source_contract import output_tree

class GeneralCaptureTests(unittest.TestCase):
    def compare(self,module,shape,outputs=("output",),policy="convert-float64-v1"):
        inputs=[dict(name="input",shape=shape)]
        original,lowered,binding=capture(module,inputs,list(outputs),dtype_policy=policy,
                                       preserve_output_structure=True)
        tree=binding["pytorch_capture"]["output_structure"]
        for case in samples(lowered,16):
            values,actual_tree=output_tree(module(torch.from_numpy(case["input"])),
                lambda v:isinstance(v,torch.Tensor),list(outputs))
            self.assertEqual(actual_tree,tree)
            expected={name:value.detach().numpy() for name,value in zip(outputs,values)}
            for evaluator in (mathematical,independent):
                actual=evaluator(lowered,case)
                for name in outputs:np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
        return original,lowered,binding
    def test_float32_conversion_is_explicit_and_recorded(self):
        _,_,binding=self.compare(torch.nn.Linear(2,2),[2])
        record=binding["pytorch_capture"]
        self.assertEqual(record["source_state_dtypes"],{"weight":"float32","bias":"float32"})
        self.assertEqual(record["dtype_policy"],"convert-float64-v1")
        self.assertFalse(record["source_precision_execution_compared"])
    def test_strict_float64_rejects_float32_and_accepts_float64(self):
        with self.assertRaisesRegex(ValueError,"Strict float64"):
            self.compare(torch.nn.Linear(2,2),[2],policy="strict-float64-v1")
        self.compare(torch.nn.Linear(2,2).double(),[2],policy="strict-float64-v1")
    def test_other_floating_and_complex_state_rejected(self):
        for dtype in (torch.float16,torch.bfloat16,torch.complex64):
            with self.assertRaises(ValueError):
                self.compare(torch.nn.Linear(2,2).to(dtype),[2])
    def test_static_shape_size_three_contexts(self):
        for variant in range(3):
            class Model(torch.nn.Module):
                def forward(self,x):
                    if variant==0:return x.view(x.size(0),-1)
                    if variant==1:return x.reshape(x.shape[0],-1)
                    return x.reshape(x.size())
            self.compare(Model(),[1,2,2])
    def test_squeeze_contiguous_three_contexts(self):
        for axis,shape in ((0,[2,2]),(-1,[2]),(1,[1,2,2])):
            class Model(torch.nn.Module):
                def forward(self,x):return x.unsqueeze(axis).squeeze(axis).contiguous()
            self.compare(Model(),shape)
    def test_nested_dict_tuple_list_output(self):
        class Model(torch.nn.Module):
            def forward(self,x):return {"prediction":x*x,"features":(x+x,[x-.25])}
        self.compare(Model(),[2],("a","b","c"))
    def test_keyword_arithmetic_module_and_functional_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer=torch.nn.Linear(2,2)
            def forward(self,x):
                a=self.layer(input=x)
                b=torch.nn.functional.linear(input=x,weight=self.layer.weight,bias=self.layer.bias)
                return torch.add(input=a,other=b)
        self.compare(Model(),[2])
    def test_public_integer_indices_are_not_float_converted(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("indices",torch.tensor([2,0],dtype=torch.int64))
            def forward(self,x):return torch.index_select(x,0,self.indices)
        _,_,binding=self.compare(Model(),[3])
        self.assertEqual(binding["pytorch_capture"]["source_state_dtypes"],{"indices":"int64"})
    def test_public_bool_arithmetic_and_inexact_integer_rejected(self):
        for buffer in (torch.tensor([True,False]),torch.tensor([2**53+1,1],dtype=torch.int64)):
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.register_buffer("value",buffer)
                def forward(self,x):return x+self.value
            with self.assertRaises(ValueError):self.compare(Model(),[2])
    def test_public_exact_integer_arithmetic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("value",torch.tensor([1,-2],dtype=torch.int32))
            def forward(self,x):return x+self.value
        self.compare(Model(),[2])
    def test_non_tensor_output_and_encrypted_shape_rejected(self):
        class NonTensor(torch.nn.Module):
            def forward(self,x):return {"tensor":x,"shape":x.size(0)}
        with self.assertRaises(ValueError):self.compare(NonTensor(),[2],("a","b"))
        class Dynamic(torch.nn.Module):
            def forward(self,x):return x.reshape(int(x[0]),-1)
        with self.assertRaises(Exception):self.compare(Dynamic(),[2])

if __name__=="__main__":unittest.main()
