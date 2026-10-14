import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from pytorch_model_capture import capture
from benchmark_torch import evaluate
from model_decomposition import decompose
from trusted_pytorch_import import prepare,run

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList([torch.nn.Linear(4,3),torch.nn.Linear(3,2)])
    def forward(self,x):
        x=self.layers[0](x)
        x=x*x
        return self.layers[1](x)

class CaptureTests(unittest.TestCase):
    def compare(self,model,shape):
        model=model.double().eval()
        spec=[dict(name="x",shape=shape)]
        original,lowered,binding=capture(model,spec,["y"])
        x=np.linspace(-.8,.8,np.prod(shape)).reshape(shape)
        np.testing.assert_allclose(evaluate(lowered,{"x":x})["y"],
                                   model(torch.from_numpy(x)).detach().numpy(),atol=1e-12,rtol=1e-12)
        self.assertTrue(binding["pytorch_capture"]["trusted_code"])
    def test_mlp_modulelist(self):self.compare(MLP(),[1,4])
    def test_conv_batchnorm_pool(self):
        m=torch.nn.Sequential(torch.nn.Conv2d(1,2,3,padding=1),
            torch.nn.BatchNorm2d(2),torch.nn.AvgPool2d(2),torch.nn.Flatten(),torch.nn.Linear(8,2))
        self.compare(m,[1,1,4,4])
    def test_residual(self):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(2,2,1)
            def forward(self,x):return self.conv(x)+x
        self.compare(Residual(),[1,2,2,2])
    def test_missing_relu_approximation(self):
        with self.assertRaisesRegex(ValueError,"approximation"):
            capture(torch.nn.Sequential(torch.nn.ReLU()),[dict(name="x",shape=[2])],["y"])
    def test_dynamic_control_rejected(self):
        class Dynamic(torch.nn.Module):
            def forward(self,x):
                if x.sum()>0:return x
                return -x
        with self.assertRaises(torch.fx.proxy.TraceError):
            capture(Dynamic(),[dict(name="x",shape=[2])],["y"])
    def test_input_budget_before_allocation(self):
        with self.assertRaises(ValueError):
            capture(MLP(),[dict(name="x",shape=[10000000])],["y"])
    def test_random_forward_rejected(self):
        class RandomModel(torch.nn.Module):
            def forward(self,x):return x+torch.rand(2)
        with self.assertRaisesRegex(ValueError,"Random forward"):
            capture(RandomModel(),[dict(name="x",shape=[2])],["y"])

    def test_mutating_forward_rejected(self):
        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("count",torch.zeros(2))
            def forward(self,x):
                self.count.add_(1)
                return x+self.count
        with self.assertRaisesRegex(ValueError,"State-mutating"):
            capture(Stateful(),[dict(name="x",shape=[2])],["y"])

    def test_hash_reject(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/"main.py").write_text("pass")
            spec=dict(format="poseidon-trusted-pytorch-v1",files={"main.py":"0"*64},
                entry="main.make",inputs=[],outputs=[],approximations={})
            (root/"manifest.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,"hash"):prepare(root/"manifest.json")
    def test_trust_required(self):
        with self.assertRaisesRegex(ValueError,"trust-local-code"):run("missing","unused")

if __name__=="__main__":unittest.main()
