"""Extrema are approximate math with a fixed tree; exact Torch is scored separately."""
import copy
import unittest
import numpy as np
import torch
from extrema_cases import models,ABS
from model_decomposition import decompose
from benchmark_graph import samples
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from operator_reference import evaluate as high_ref,reduce_extreme
from pytorch_model_capture import capture

class ExtremaTests(unittest.TestCase):
    def test_all_contexts_dual_references(self):
        for item in models():
            g=item["model"]
            with self.subTest(id=item["id"]):
                low,b=decompose(g);again,b2=decompose(copy.deepcopy(g))
                self.assertEqual(low,again);self.assertEqual(b,b2)
                for inputs in samples(low,16):
                    expected=high_ref(g,inputs)["result"]
                    for ref in (math_ref,torch_ref):
                        np.testing.assert_allclose(ref(low,inputs)["result"],expected,atol=1e-12,rtol=1e-12)
                self.assertEqual(set(b["source_map"].values()),{"operation"})

    def test_ideal_matches_native(self):
        for item in models():
            g=item["model"];op=item["operator"];a=g["nodes"][0]["attrs"]
            for inputs in samples(decompose(g)[0],4):
                x=torch.from_numpy(inputs["x"]);actual=high_ref(g,inputs,ideal=True)["result"]
                if op=="padding":
                    # NumPy pad is already independent; test the public pad modes below.
                    continue
                if op in ("maximum","minimum"):
                    expected=(torch.maximum if op=="maximum" else torch.minimum)(x,torch.from_numpy(inputs["other"])).numpy()
                elif op in ("max","min"):
                    expected=(torch.amax if op=="max" else torch.amin)(x,dim=a["axes"],keepdim=a["keepdims"]).numpy().reshape(actual.shape)
                else:
                    fn=torch.nn.functional.max_pool1d if op=="max_pool1d" else torch.nn.functional.max_pool2d
                    expected=fn(x,a["kernel"],a["stride"],a["padding"],a["dilation"]).numpy()
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_interval_checks_intermediate_difference(self):
        g=next(x["model"] for x in models() if x["operator"]=="maximum")
        with self.assertRaisesRegex(ValueError,"outside"):
            high_ref(g,{"x":np.full(3,3.),"other":np.full(3,-3.)})
        # Each original value is in the domain; the comparison difference is not.
        with self.assertRaisesRegex(ValueError,"outside"):
            reduce_extreme(np.array([3.,-3.,0.]),[0],ABS,True,False)

    def test_tree_semantics_and_approximation_error(self):
        g=next(x["model"] for x in models() if x["operator"]=="max")
        inputs={"x":np.zeros(4)}
        actual=high_ref(g,inputs)["result"]
        self.assertGreater(float(np.max(abs(actual-high_ref(g,inputs,ideal=True)["result"]))),0)
        # Freeze the adjacent-pair order: two tied pairs become 0.25, then 0.5.
        np.testing.assert_array_equal(actual,[.5])

    def test_invalid_attributes_and_budgets(self):
        by={x["operator"]:x["model"] for x in models()}
        bad=[]
        def altered(op,**attrs):
            g=copy.deepcopy(by[op]);g["nodes"][0]["attrs"].update(attrs);bad.append(g)
        altered("padding",pads=[[0,0],[0,-1],[0,0]])
        altered("padding",pads=[[0,0],[0,2],[0,0]],mode="reflect")
        altered("padding",mode="circular")
        altered("padding",value=float("inf"))
        altered("max",axes=[1,1]);altered("min",keepdims=1)
        altered("maximum",approximation={})
        altered("max_pool2d",padding=[2,0]);altered("max_pool1d",dilation=[0])
        altered("max_pool2d",stride=[True,1])
        g=copy.deepcopy(by["max"]);g["inputs"][0]["shape"]=[128]
        g["nodes"][0]["attrs"]["axes"]=[0];bad.append(g)
        g=copy.deepcopy(by["max_pool1d"]);g["inputs"][0]["shape"]=[1,1,1]
        g["nodes"][0]["attrs"].update(kernel=[2],dilation=[2],padding=[1],stride=[1]);bad.append(g)
        for g in bad:
            with self.subTest(g=g),self.assertRaises(ValueError):decompose(g)

class ExtremaCaptureTests(unittest.TestCase):
    def compare(self,module,shape,approx=None):
        module=module.double().eval()
        high,low,_=capture(module,[dict(name="x",shape=shape)],["result"],approximations=approx)
        for inputs in samples(low,4):
            np.testing.assert_allclose(torch_ref(low,inputs)["result"],high_ref(high,inputs)["result"],atol=1e-12,rtol=1e-12)
            native=module(torch.from_numpy(inputs["x"])).detach().numpy()
            ideal=high_ref(high,inputs,ideal=True)["result"]
            np.testing.assert_allclose(ideal,native.reshape(ideal.shape),atol=1e-12,rtol=1e-12)

    def test_padding_modules(self):
        for m,s in [(torch.nn.ConstantPad1d((1,2),.125),[1,2,3]),
                    (torch.nn.ReflectionPad2d((1,0,1,0)),[1,1,2,3]),
                    (torch.nn.ReplicationPad1d((1,2)),[1,1,2]),
                    (torch.nn.ZeroPad2d(1),[1,1,2,2])]:
            self.compare(torch.nn.Sequential(m),s)

    def test_functional_padding(self):
        class Pad(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.pad(x,(1,0,0,1),mode="constant",value=.25)
        self.compare(Pad(),[1,1,2,2])

    def test_pool_modules(self):
        for m,s in [(torch.nn.MaxPool1d(2),[1,1,4]),(torch.nn.MaxPool1d(2,1,1),[1,2,3]),
                    (torch.nn.MaxPool2d(2),[1,1,2,2])]:
            self.compare(torch.nn.Sequential(m),s,dict(abs=ABS))

    def test_reductions_and_broadcast(self):
        class Maximum(torch.nn.Module):
            def forward(self,x):return torch.maximum(x,x.amin(dim=-1,keepdim=True))
        # Ideal/native compose agrees; approximate intermediates may differ from
        # exact intermediates, so compare each full semantics independently.
        high,low,_=capture(Maximum(),[dict(name="x",shape=[2,3])],["result"],approximations=dict(abs=ABS))
        inputs={"x":np.linspace(-.8,.8,6).reshape(2,3)}
        np.testing.assert_allclose(torch_ref(low,inputs)["result"],high_ref(high,inputs)["result"],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(high_ref(high,inputs,ideal=True)["result"],Maximum()(torch.from_numpy(inputs["x"])).numpy(),atol=1e-12,rtol=1e-12)

    def test_reject_pool_indices_ceil_missing_approx(self):
        for m in (torch.nn.MaxPool1d(2,return_indices=True),torch.nn.MaxPool1d(2,ceil_mode=True)):
            with self.assertRaisesRegex(ValueError,"floor mode"):
                capture(torch.nn.Sequential(m),[dict(name="x",shape=[1,1,4])],["result"],approximations=dict(abs=ABS))
        with self.assertRaisesRegex(ValueError,"approximation"):
            capture(torch.nn.Sequential(torch.nn.MaxPool1d(2)),[dict(name="x",shape=[1,1,4])],["result"])

if __name__=="__main__":unittest.main()
