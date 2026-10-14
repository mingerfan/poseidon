"""FX common normalization/variance calls, independent references and refusals."""
import copy,unittest
import torch,numpy as np
from pytorch_model_capture import capture
from benchmark_graph import samples
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as tensor_reference
from operator_reference import evaluate as operator_reference
from cases.trusted_pytorch_functional_v1.models import Layer,Group,Variance
APPROX={"rsqrt":dict(method="newton",domain=[1.,5.],iterations=1)}
SHAPES={"layer":[[1,4],[2,2],[1,2,2]],"group":[[1,2,2],[1,2,2,2],[2,2,2]],
        "variance":[[2,3],[2,2,2],[2,2]]}

class FunctionalCaptureTests(unittest.TestCase):
    def compare(self,module,dims,approximations=None):
        module=module.double().eval()
        signature=[dict(name="x",shape=dims)]
        args=dict(approximations=approximations or {},resource_profile="general-tensors-v1",
                  dtype_policy="strict-float64-v1",preserve_output_structure=True)
        original,lowered,binding=capture(module,signature,["out"],**args)
        self.assertEqual(capture(module,signature,["out"],**args),(original,lowered,binding))
        for values in samples(lowered,16):
            actual=module(torch.from_numpy(values["x"])).detach().numpy()
            ideal=operator_reference(original,values,ideal=True)["out"]
            approx=operator_reference(original,values)["out"]
            np.testing.assert_allclose(actual,ideal,atol=1e-12,rtol=1e-12)
            for evaluate in (math_reference,tensor_reference):
                np.testing.assert_allclose(evaluate(lowered,values)["out"],approx,atol=1e-12,rtol=1e-12)
        return original,lowered,binding
    def test_layer_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):self.compare(Layer(variant),SHAPES["layer"][variant],APPROX)
    def test_group_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):self.compare(Group(variant),SHAPES["group"][variant],APPROX)
    def test_variance_three_contexts(self):
        for variant in range(3):
            with self.subTest(variant=variant):self.compare(Variance(variant),SHAPES["variance"][variant])
    def test_norm_missing_approximation_rejected(self):
        for model,shape in [(Layer(0),[1,4]),(Group(0),[1,2,2])]:
            with self.subTest(kind=type(model).__name__),self.assertRaisesRegex(ValueError,"Explicit approximation"):
                capture(model,[dict(name="x",shape=shape)],["out"])
    def test_variance_default_sample_correction_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.var(-1,keepdim=True)
        with self.assertRaisesRegex(ValueError,"population variance"):
            capture(Model(),[dict(name="x",shape=[2,3])],["out"])
    def test_nonzero_correction_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.var(x,dim=-1,correction=1,keepdim=True)
        with self.assertRaisesRegex(ValueError,"population variance"):
            capture(Model(),[dict(name="x",shape=[2,3])],["out"])
    def test_legacy_unbiased_false_supported(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.var(x,-1,False,True)
        self.compare(Model(),[2,3])
    def test_float_correction_rejected_by_pinned_torch(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.var(x,dim=-1,correction=0.,keepdim=True)
        with self.assertRaises(TypeError):
            capture(Model(),[dict(name="x",shape=[2,3])],["out"])
    def test_duplicate_axes_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.var((0,0),False,True)
        with self.assertRaises(Exception):capture(Model(),[dict(name="x",shape=[2,3])],["out"])
    def test_norm_cipher_gain_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x,w):return torch.nn.functional.layer_norm(x,(4,),weight=w,eps=1.)
        with self.assertRaisesRegex(ValueError,"public gain/bias"):
            capture(Model(),[dict(name="x",shape=[2,4]),dict(name="w",shape=[4])],["out"],approximations=APPROX)
    def test_group_rank_two_reports_supported_rank(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.group_norm(x,1,eps=1.)
        with self.assertRaisesRegex(ValueError,"N,C,spatial"):
            capture(Model(),[dict(name="x",shape=[2,2])],["out"],approximations=APPROX)
    def test_epsilon_limit_preserved(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.layer_norm(x,(4,),eps=2.)
        with self.assertRaisesRegex(ValueError,"epsilon"):
            capture(Model(),[dict(name="x",shape=[1,4])],["out"],approximations=APPROX)
    def test_approximation_domain_checked_on_reference(self):
        original,lowered,_=capture(Layer(0),[dict(name="x",shape=[1,4])],["out"],approximations=APPROX)
        with self.assertRaisesRegex(ValueError,"outside explicit"):
            operator_reference(original,{"x":np.array([[0.,10.,0.,10.]],dtype=np.float64)})
    def test_source_operators_are_stable(self):
        expected={"layer":"layer_norm","group":"group_norm","variance":"variance"}
        for name,cls in [("layer",Layer),("group",Group),("variance",Variance)]:
            original,_,_=capture(cls(0),[dict(name="x",shape=SHAPES[name][0])],["out"],approximations=APPROX)
            self.assertIn(expected[name],[n["op"] for n in original["nodes"]])

    def test_functional_rules_are_shared_input_declarations(self):
        from application_capabilities import input_sources
        from component_contract import capabilities
        from pytorch_model_capture import functional_input_rules
        declared=functional_input_rules()
        self.assertEqual(input_sources()["trusted_pytorch"]["functional_input_rules"],declared)
        self.assertEqual(capabilities()["input_sources"]["trusted_pytorch"]["functional_input_rules"],declared)
        self.assertEqual(declared["variance"]["correction"],0)
        self.assertEqual(declared["group_norm"]["ranks"],[3,4])
        self.assertNotIn("encrypted_execution",declared)

if __name__=="__main__":unittest.main()
