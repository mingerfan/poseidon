"""Tensor and activation expansion: independent math, PyTorch, and rejection checks."""
import copy
import unittest
import numpy as np
import torch
from test_operator_extension import graph, POLY
from model_decomposition import decompose
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from operator_reference import evaluate as high_ref
from pytorch_model_capture import capture

class TensorExpansionTests(unittest.TestCase):
    def compare(self,g,x):
        low,b=decompose(g)
        expected=high_ref(g,{"x":x})["result"]
        for runner in (math_ref,torch_ref):
            np.testing.assert_allclose(runner(low,{"x":x})["result"],expected,atol=1e-12,rtol=1e-12)
        again,b2=decompose(copy.deepcopy(g))
        self.assertEqual(low,again);self.assertEqual(b,b2)
        self.assertEqual(set(b["source_map"].values()),{"operation"})
        return expected

    def test_broadcast_three_contexts(self):
        for s,t in [((2,),(3,2)),((1,2,1),(2,2,3)),((1,),(17,))]:
            x=np.arange(np.prod(s),dtype=np.float64).reshape(s)+.1
            actual=self.compare(graph("broadcast",s,dict(shape=list(t))),x)
            np.testing.assert_array_equal(actual,np.broadcast_to(x,t))

    def test_gather_three_contexts(self):
        for s,axis,indices in [((4,),0,[3,0,3]),((2,3),-1,[2,0]),((1,2,3),1,[1,0,1])]:
            x=np.arange(np.prod(s),dtype=np.float64).reshape(s)/10
            actual=self.compare(graph("public_gather",s,dict(axis=axis,indices=indices)),x)
            np.testing.assert_array_equal(actual,np.take(x,indices,axis=axis))

    def test_adaptive_pool_three_contexts(self):
        for s,out in [((1,2,5),[3]),((1,1,5,4),[2,3]),((2,2,3,3),[1,2])]:
            x=np.arange(np.prod(s),dtype=np.float64).reshape(s)/13-.8
            actual=self.compare(graph("adaptive_avg_pool",s,dict(output_size=out)),x)
            fn=torch.nn.functional.adaptive_avg_pool1d if len(s)==3 else torch.nn.functional.adaptive_avg_pool2d
            np.testing.assert_allclose(actual,fn(torch.from_numpy(x),out).numpy(),atol=1e-12,rtol=1e-12)

    def test_leaky_prelu_three_contexts(self):
        for s in ((4,),(2,3),(1,2,2)):
            x=np.linspace(-.9,.7,np.prod(s)).reshape(s)
            gs=[graph("leaky_relu",s,dict(negative_slope=.2,approximation=POLY)),
                graph("prelu",s,dict(approximation=POLY),{"alpha":[.2]},["x","alpha"])]
            if len(s)>1:gs.append(graph("prelu",s,dict(approximation=POLY),
                                      {"alpha":[.1+i/10 for i in range(s[1])]},["x","alpha"]))
            for g in gs:
                with self.subTest(op=g["nodes"][0]["op"],shape=s):
                    self.compare(g,x)
                    if g["nodes"][0]["op"]=="leaky_relu":
                        np.testing.assert_array_equal(high_ref(g,{"x":x},ideal=True)["result"],np.where(x>=0,x,.2*x))

    def test_logsoftmax_three_contexts(self):
        a=dict(axis=-1,exp=dict(method="taylor",domain=[-1.,1.],degree=3),
               log=dict(method="polynomial",basis="power",domain=[.1,8.],coefficients=[-.4,.5,-.03]))
        for s in ((3,),(2,3),(1,2,2)):
            x=np.linspace(-.3,.3,np.prod(s)).reshape(s);g=graph("logsoftmax",s,a)
            self.compare(g,x)
            np.testing.assert_allclose(high_ref(g,{"x":x},ideal=True)["result"],
                                       torch.log_softmax(torch.from_numpy(x),dim=-1).numpy(),atol=1e-12,rtol=1e-12)

    def test_rejections(self):
        cases=[graph("broadcast",(2,),dict(shape=[3])),
               graph("broadcast",(1,),dict(shape=[257])),
               graph("public_gather",(2,),dict(axis=0,indices=[-1])),
               graph("public_gather",(2,),dict(axis=0,indices=[True])),
               graph("public_gather",(2,),dict(axis=0,indices=[2])),
               graph("adaptive_avg_pool",(1,1,3),dict(output_size=[4])),
               graph("adaptive_avg_pool",(1,1,3),dict(output_size=[True])),
               graph("leaky_relu",attrs=dict(negative_slope=True,approximation=POLY)),
               graph("prelu",(2,3),dict(approximation=POLY),{"alpha":[.2,.3]},["x","alpha"])]
        for g in cases:
            with self.subTest(g=g),self.assertRaises(ValueError):decompose(g)

    def test_expansion_budget(self):
        # A legal output tensor must still be rejected if decomposition exceeds 64 nodes.
        with self.assertRaisesRegex(ValueError,"Expanded graph node budget"):
            decompose(graph("public_gather",(2,),dict(axis=0,indices=[0,1]*40)))

    def test_log_domain(self):
        a=dict(axis=0,exp=dict(method="taylor",domain=[-1.,1.],degree=2),
               log=dict(POLY,domain=[.1,1.]))
        with self.assertRaisesRegex(ValueError,"outside"):
            high_ref(graph("logsoftmax",(2,),a),{"x":np.zeros(2)})

class CaptureExpansionTests(unittest.TestCase):
    def test_pool_modules(self):
        for m,s in [(torch.nn.AdaptiveAvgPool1d(3),[1,2,5]),
                    (torch.nn.AdaptiveAvgPool2d((2,3)),[1,1,5,4]),
                    (torch.nn.AdaptiveAvgPool2d((1,2)),[2,2,3,3])]:
            m=torch.nn.Sequential(m).double()
            high,low,_=capture(m,[dict(name="x",shape=s)],["y"])
            x=np.arange(np.prod(s),dtype=np.float64).reshape(s)/7
            np.testing.assert_allclose(torch_ref(low,{"x":x})["y"],m(torch.from_numpy(x)).detach().numpy(),atol=1e-12,rtol=1e-12)

    def test_public_index_and_broadcast(self):
        class Select(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("idx",torch.tensor([2,0,2]))
            def forward(self,x):return torch.index_select(x,1,self.idx).expand(2,3)
        m=Select()
        high,low,_=capture(m,[dict(name="x",shape=[1,3])],["y"])
        x=np.array([[.2,-.8,.5]])
        np.testing.assert_array_equal(torch_ref(low,{"x":x})["y"],m(torch.from_numpy(x)).numpy())

    def test_activation_modules(self):
        approximations=dict(relu=POLY,exp=dict(method="taylor",domain=[-1.,1.],degree=3),
                            log=dict(POLY,domain=[.1,8.]))
        for m in (torch.nn.LeakyReLU(.2),torch.nn.PReLU(3),torch.nn.LogSoftmax(-1)):
            high,low,_=capture(torch.nn.Sequential(m),[dict(name="x",shape=[2,3])],["result"],
                              approximations=approximations)
            x=np.linspace(-.4,.4,6).reshape(2,3)
            np.testing.assert_allclose(torch_ref(low,{"x":x})["result"],high_ref(high,{"x":x})["result"],atol=1e-12,rtol=1e-12)
            expected=m.double()(torch.from_numpy(x)).detach().numpy()
            np.testing.assert_allclose(high_ref(high,{"x":x},ideal=True)["result"],expected,atol=1e-12,rtol=1e-12)


class SingletonLoweringTests(unittest.TestCase):
    def test_singleton_independent_vector_semantics(self):
        from test_packed_prefix_lowering import models,public_execute
        from unified_graph_contract import prepare,validate_candidate
        from compiler_configuration import PROFILE_SHA256
        from unified_graph_lowering import lower
        from benchmark_graph import samples
        for g in models():
            r=prepare(g,PROFILE_SHA256)
            src=lower(r,singleton=True)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
            for inputs in samples(g,4):
                actual=public_execute(src,r,inputs);expected=math_ref(g,inputs)
                for i,o in enumerate(g["outputs"]):
                    want=expected[o["name"]].reshape(-1)
                    np.testing.assert_allclose(actual[i][:want.size],want,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(actual[i][want.size:],0,atol=1e-12,rtol=1e-12)

    def test_original_pool_failure_same_request(self):
        import json
        from pathlib import Path
        from test_packed_prefix_lowering import public_execute
        from unified_graph_contract import prepare,validate_candidate
        from compiler_configuration import PROFILE_SHA256,configuration
        from unified_graph_lowering import lower,candidate_source
        from benchmark_graph import samples
        g=json.loads((Path(__file__).parent/"cases/operator-extension-v3/adaptive_avg_pool_2-model.json").read_text())
        r=prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"));before=copy.deepcopy(r)
        for kwargs in ({},{"packed_prefix":True}):
            with self.assertRaisesRegex(ValueError,"AST node limit"):
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=lower(r,**kwargs)),r)
        src,binding=candidate_source(r)
        self.assertEqual(binding["strategy"],"singleton-v1");self.assertEqual(r,before)
        validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
        for inputs in samples(g,4):
            actual=public_execute(src,r,inputs)[0]
            want=math_ref(g,inputs)["result"].reshape(-1)
            np.testing.assert_allclose(actual[:want.size],want,atol=1e-12,rtol=1e-12)

if __name__=="__main__":unittest.main()
