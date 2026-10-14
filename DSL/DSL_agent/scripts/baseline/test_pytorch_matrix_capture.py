"""FX matrix aliases: original Torch, independent graph references and rejection."""
import unittest
import numpy as np
import torch
from pytorch_model_capture import capture
from benchmark_graph import samples
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference

class Product(torch.nn.Module):
    def __init__(self, kind, spelling):
        super().__init__()
        self.kind=kind
        self.spelling=spelling
    def forward(self,x,y):
        if self.kind=="dot":
            if self.spelling==0:return torch.dot(x,y).reshape(1)
            if self.spelling==1:return x.dot(y).reshape(1)
            return torch.dot(input=x,tensor=y,out=None).reshape(1)
        if self.kind=="mm":
            if self.spelling==0:return torch.mm(x,y)
            if self.spelling==1:return x.mm(y)
            return torch.mm(input=x,mat2=y,out=None)
        if self.kind=="bmm":
            if self.spelling==0:return torch.bmm(x,y)
            if self.spelling==1:return x.bmm(y)
            return torch.bmm(input=x,mat2=y,out=None)
        if self.spelling==0:return x@y
        if self.spelling==1:return x.matmul(y)
        return torch.matmul(input=x,other=y,out=None)

CONTEXTS={
 "dot":[([3],[3]),([5],[5]),([7],[7])],
 "mm":[([2,3],[3,2]),([1,4],[4,3]),([3,2],[2,1])],
 "bmm":[([2,2,3],[2,3,2]),([3,1,2],[3,2,3]),([2,3,1],[2,1,2])],
 "matmul":[([2,3],[3,2]),([2,1,3],[3,2]),([1,2,3],[2,3,2])],
}
class MatrixCaptureTests(unittest.TestCase):
    def compare(self,kind):
        for spelling in range(3):
            for first,second in CONTEXTS[kind]:
                with self.subTest(kind=kind,spelling=spelling,shapes=(first,second)):
                    m=Product(kind,spelling).double().eval()
                    inputs=[dict(name="x",shape=first),dict(name="y",shape=second)]
                    original,lowered,binding=capture(m,inputs,["output"])
                    self.assertEqual((original,lowered,binding),capture(m,inputs,["output"]))
                    self.assertIn("dot" if kind=="dot" else "matmul",{n["op"] for n in original["nodes"]})
                    for values in samples(lowered,16):
                        expected=m(torch.from_numpy(values["x"]),torch.from_numpy(values["y"])).numpy()
                        for evaluator in (math_reference,torch_reference):
                            actual=evaluator(lowered,values)["output"]
                            self.assertEqual(actual.shape,expected.shape)
                            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    def test_dot_three_contexts_and_spellings(self):self.compare("dot")
    def test_mm_three_contexts_and_spellings(self):self.compare("mm")
    def test_bmm_three_contexts_and_spellings(self):self.compare("bmm")
    def test_matmul_three_contexts_and_spellings(self):self.compare("matmul")
    def test_matrix_capacity_baselines_match_two_references(self):
        from component_contract import prepare_task
        from unified_graph_lowering import candidate_source
        from test_unified_capacity_linear import interpreted_output
        for kind in ("dot","mm","bmm"):
            for index,(first,second) in enumerate(CONTEXTS[kind]):
                inputs=[dict(name="x",shape=first),dict(name="y",shape=second)]
                _,graph,_=capture(Product(kind,index),inputs,["output"],resource_profile="medium-v1")
                request=prepare_task(graph,dict(chunk_period=256,generation_guidance="explicit-v10"))
                source,_=candidate_source(request)
                for values in samples(graph,16):
                    a,b=math_reference(graph,values),torch_reference(graph,values)
                    np.testing.assert_allclose(a["output"],b["output"],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(interpreted_output(source,request,values),
                                              a["output"].reshape(-1),atol=1e-12,rtol=1e-12)
    def test_two_cipher_broadcast_three_contexts(self):
        from benchmark_graph import CAPACITY_FORMAT
        from component_contract import prepare_task
        from unified_graph_lowering import candidate_source
        from test_unified_capacity_linear import interpreted_output
        contexts=[([2,1],[1,3],256),([2,1,2,1],[1,3,1,2],256),
                  ([2,1,3],[1,2,1],4)]
        for op in ("add","subtract","multiply"):
            for first,second,period in contexts:
                graph=dict(format=CAPACITY_FORMAT,resource_profile="medium-v1",id="cipher_broadcast",
                    inputs=[dict(name="x",shape=first),dict(name="y",shape=second)],constants={},
                    nodes=[dict(id="combine",op=op,inputs=["x","y"],attrs={},outputs=["z"])],
                    outputs=[dict(name="output",value="z")])
                request=prepare_task(graph,dict(chunk_period=period))
                source,_=candidate_source(request)
                for values in samples(graph,16):
                    a,b=math_reference(graph,values),torch_reference(graph,values)
                    np.testing.assert_allclose(a["output"],b["output"],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(interpreted_output(source,request,values),
                                              a["output"].reshape(-1),atol=1e-12,rtol=1e-12)
    def test_matrix_out_mutation_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x,y):return torch.mm(x,y,out=x)
        with self.assertRaisesRegex(ValueError,"out mutation"):
            capture(Model(),[dict(name=n,shape=[2,2]) for n in ("x","y")],["output"])
    def test_public_matrix_normalizes_to_existing_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("weight",torch.eye(2,dtype=torch.float64))
            def forward(self,x):return x.mm(self.weight)
        model=Model()
        original,lowered,_=capture(model,[dict(name="x",shape=[2,2])],["output"])
        self.assertIn("linear",{n["op"] for n in original["nodes"]})
        for values in samples(lowered,16):
            expected=model(torch.from_numpy(values["x"])).detach().numpy()
            for evaluate in (math_reference,torch_reference):
                np.testing.assert_allclose(evaluate(lowered,values)["output"],expected,atol=1e-12,rtol=1e-12)
    def test_vector_matmul_keeps_existing_rank_contract(self):
        with self.assertRaisesRegex(ValueError,"Rank-2/3 matmul shape"):
            capture(Product("matmul",1),[dict(name=n,shape=[3]) for n in ("x","y")],["output"])
    def test_mm_rank_three_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"ShapeProp error"):
            capture(Product("mm",0),[dict(name=n,shape=[2,2,2]) for n in ("x","y")],["output"])
    def test_bmm_batch_broadcast_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"ShapeProp error"):
            capture(Product("bmm",0),[dict(name="x",shape=[1,2,2]),dict(name="y",shape=[2,2,2])],["output"])
if __name__=="__main__":unittest.main()
