"""Equivalent PyTorch spellings: exact source math, decomposition and refusal."""
import copy, unittest
import torch
import numpy as np
from benchmark_graph import samples
from pytorch_model_capture import capture, functional_input_rules
from operator_reference import evaluate as high
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference
from cases.pytorch_spelling_v1.models import make, fixed

SHAPES={
 "conv1":[[1,1,4],[1,1,3],[1,2,3]],
 "conv2":[[1,1,3,3],[1,1,3,3],[1,2,2,2]],
 "pool1":[[1,1,4],[1,1,3],[1,2,3]],
 "pool2":[[1,1,3,3],[1,1,3,3],[1,2,2,2]],
 "batch":[[2,2],[1,2,3],[1,2,2,2]],
 "layer":[[1,2],[2,2],[1,2,2]],
 "group":[[1,2,3],[1,2,2,2],[2,2,2]],
 "transpose":[[1,3],[2,3],[1,2,3]],
}
APPROX={"rsqrt":dict(method="newton",domain=[1.,5.],iterations=1)}

class SpellingTests(unittest.TestCase):
    def compare(self,module,shape,approximations=None):
        module=module.double().eval()
        specs=[dict(name="x",shape=shape)]
        options=dict(approximations=approximations or {},resource_profile="general-tensors-v1",
                     dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True)
        before={k:v.detach().clone() for k,v in module.state_dict().items()}
        original,lowered,binding=capture(module,specs,["out"],**options)
        self.assertEqual((original,lowered,binding),capture(module,specs,["out"],**options))
        for name,value in before.items():self.assertTrue(torch.equal(value,module.state_dict()[name]))
        for values in samples(lowered,16):
            expected=module(torch.from_numpy(values["x"])).detach().numpy()
            ideal=high(original,values,ideal=True)["out"]
            approximate=high(original,values)["out"]
            self.assertEqual(expected.shape,ideal.shape)
            np.testing.assert_allclose(expected,ideal,atol=1e-12,rtol=1e-12)
            for evaluate in (math_reference,torch_reference):
                np.testing.assert_allclose(evaluate(lowered,values)["out"],approximate,atol=1e-12,rtol=1e-12)
        return original,lowered,binding
    def contexts(self,kind):
        for variant,shape in enumerate(SHAPES[kind]):
            with self.subTest(kind=kind,variant=variant):
                self.compare(make(kind,variant),shape,APPROX if kind in ("layer","group") else None)
    def test_conv1_three_contexts(self):self.contexts("conv1")
    def test_conv2_three_contexts(self):self.contexts("conv2")
    def test_pool1_three_contexts(self):self.contexts("pool1")
    def test_pool2_three_contexts(self):self.contexts("pool2")
    def test_batch_norm_three_contexts(self):self.contexts("batch")
    def test_layer_no_affine_three_contexts(self):self.contexts("layer")
    def test_group_no_affine_three_contexts(self):self.contexts("group")
    def test_tensor_T_three_contexts(self):self.contexts("transpose")
    def test_public_right_matrix_three_contexts(self):
        class Model(torch.nn.Module):
            def __init__(self,cols):
                super().__init__();self.register_buffer("weight",fixed((3,cols),1))
            def forward(self,x):return torch.matmul(x,self.weight)
        for dims,cols in [([2,3],2),([1,3],4),([2,1,3],2)]:
            with self.subTest(dims=dims,cols=cols):self.compare(Model(cols),dims)
    def test_public_left_matrix_three_contexts(self):
        class Model(torch.nn.Module):
            def __init__(self,rows):
                super().__init__();self.register_buffer("weight",fixed((rows,3),1))
            def forward(self,x):return self.weight@x
        for dims,rows in [([3,2],2),([3,1],1),([2,3,2],2)]:
            with self.subTest(dims=dims,rows=rows):self.compare(Model(rows),dims)
    def test_bound_source_weight_is_reused_for_public_T(self):
        class ParameterTranspose(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(fixed((2,3),1))
                self.register_buffer("bias",fixed((2,),2))
            def forward(self,x):return x@self.weight.T+self.bias
        original,_,binding=self.compare(ParameterTranspose(),[1,3])
        self.assertEqual(len(original["constants"]),2)
        self.assertEqual(set(binding["pytorch_capture"]["state_constant_sources"].values()),{"weight","bias"})
        self.assertEqual(original["nodes"][0]["op"],"linear")
    def test_module_and_function_layer_norm_are_equivalent(self):
        class Functional(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.layer_norm(x,(2,),eps=1.)
        a,_,_=self.compare(make("layer",0),[1,2],APPROX)
        b,_,_=self.compare(Functional(),[1,2],APPROX)
        for values in samples(a,16):
            np.testing.assert_allclose(high(a,values)["out"],high(b,values)["out"],atol=1e-12,rtol=1e-12)
    def test_module_and_function_group_norm_are_equivalent(self):
        class Functional(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.group_norm(x,1,eps=1.)
        a,_,_=self.compare(make("group",0),[1,2,3],APPROX)
        b,_,_=self.compare(Functional(),[1,2,3],APPROX)
        for values in samples(a,16):
            np.testing.assert_allclose(high(a,values)["out"],high(b,values)["out"],atol=1e-12,rtol=1e-12)
    def test_pool_ceil_mode_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.avg_pool1d(x,2,ceil_mode=True)
        with self.assertRaisesRegex(ValueError,"Pool ceil/divisor"):
            capture(Model(),[dict(name="x",shape=[1,1,4])],["out"])
    def test_pool_divisor_override_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.avg_pool2d(x,2,divisor_override=3)
        with self.assertRaisesRegex(ValueError,"Pool ceil/divisor"):
            capture(Model(),[dict(name="x",shape=[1,1,3,3])],["out"])
    def test_convolution_string_padding_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("weight",fixed((1,1,1),1))
            def forward(self,x):return torch.nn.functional.conv1d(x,self.weight,padding="same")
        with self.assertRaisesRegex(ValueError,"Explicit zero convolution padding"):
            capture(Model(),[dict(name="x",shape=[1,1,4])],["out"])
    def test_encrypted_convolution_weight_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x,w):return torch.nn.functional.conv1d(x,w)
        with self.assertRaisesRegex(ValueError,"public"):
            capture(Model(),[dict(name="x",shape=[1,1,4]),dict(name="w",shape=[1,1,2])],["out"])
    def test_training_batch_norm_refused(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.batch_norm(x,None,None,training=True)
        with self.assertRaisesRegex(ValueError,"inference BatchNorm"):
            capture(Model(),[dict(name="x",shape=[2,2])],["out"])
    def test_encrypted_batch_norm_statistics_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("variance",torch.ones(2,dtype=torch.float64))
            def forward(self,x,m):return torch.nn.functional.batch_norm(x,m,self.variance)
        with self.assertRaisesRegex(ValueError,"public"):
            capture(Model(),[dict(name="x",shape=[2,2]),dict(name="m",shape=[2])],["out"])
    def test_public_right_vector_matmul_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("weight",fixed((3,2),1))
            def forward(self,x):return x@self.weight
        with self.assertRaisesRegex(ValueError,"encrypted rank-2/3"):
            capture(Model(),[dict(name="x",shape=[3])],["out"])
    def test_public_left_vector_matmul_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("weight",fixed((2,3),1))
            def forward(self,x):return self.weight@x
        with self.assertRaisesRegex(ValueError,"encrypted rank-2/3"):
            capture(Model(),[dict(name="x",shape=[3])],["out"])
    def test_public_batched_matrix_refused(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("weight",fixed((2,3,2),1))
            def forward(self,x):return torch.bmm(x,self.weight)
        with self.assertRaisesRegex(ValueError,"fixed rank-2"):
            capture(Model(),[dict(name="x",shape=[2,2,3])],["out"])
    def test_no_normalization_approximation_refused(self):
        for kind in ("layer","group"):
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,"Explicit approximation"):
                capture(make(kind,0),[dict(name="x",shape=SHAPES[kind][0])],["out"])
    def test_rules_describe_input_mapping_without_execution_claim(self):
        rules=functional_input_rules()
        self.assertEqual(rules["version"],"poseidon-pytorch-functional-input-v2")
        self.assertFalse(rules["batch_norm"]["training"])
        self.assertEqual(rules["matrix_normalization"]["public_rank"],2)
        self.assertNotIn("encrypted_execution",rules)
if __name__=="__main__":unittest.main(verbosity=2)
