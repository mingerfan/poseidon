"""Safety and numerical tests for the explicit common-FX development prototype."""
import importlib.util,math,operator,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from benchmark_graph import canonical
from hecate_python_env import ROOT
P=ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-capture-prototype-v2/pytorch_arithmetic_normalization.py"
spec=importlib.util.spec_from_file_location("staged_common_adapter",P)
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

def graph(target,args=None,kwargs=None,*,names=("x","y"),method=False):
    g=torch.fx.Graph();ports=[g.placeholder(n) for n in names]
    chosen=tuple(ports) if args is None else args(ports)
    node=g.call_method(target,chosen,kwargs or {}) if method else g.call_function(target,chosen,kwargs or {})
    g.output(node)
    return torch.fx.GraphModule(torch.nn.Module(),g)

class CommonPrototypeTests(unittest.TestCase):
    def test_alpha_add_sub_method_preserves_values_and_source(self):
        for fn,method in ((torch.add,False),(torch.sub,False),("add",True),("sub",True)):
            for alpha in (-2.,0.,2.,0.125):
                with self.subTest(fn=fn,alpha=alpha):
                    model=graph(fn,kwargs={"alpha":alpha},method=method)
                    source=str(model.graph)
                    transformed,record=adapter.normalize_common(model,[dict(name="x",shape=[2,3]),dict(name="y",shape=[3])])
                    x=torch.arange(6,dtype=torch.float64).reshape(2,3)/3;y=torch.tensor([0.25,-2.,3.],dtype=torch.float64)
                    torch.testing.assert_close(transformed(x,y),model(x,y),atol=1e-12,rtol=1e-12)
                    self.assertEqual(str(model.graph),source)
                    self.assertEqual(record["entries"][0]["alpha"],alpha)

    def test_alpha_dynamic_bool_nonfinite_and_out_rejected(self):
        for alpha in (True,float("nan"),float("inf")):
            with self.subTest(alpha=alpha):
                model=graph(torch.add,kwargs={"alpha":alpha})
                with self.assertRaisesRegex(ValueError,"Finite public scalar"):
                    adapter.normalize_common(model,[dict(name="x",shape=[2]),dict(name="y",shape=[2])])
        g=torch.fx.Graph();x=g.placeholder("x");y=g.placeholder("y");a=g.placeholder("alpha")
        n=g.call_function(torch.add,(x,y),{"alpha":a});g.output(n);m=torch.fx.GraphModule(torch.nn.Module(),g)
        with self.assertRaisesRegex(ValueError,"Finite public scalar"):
            adapter.normalize_common(m,[dict(name=n,shape=[2]) for n in ("x","y","alpha")])
        # FX cannot render an arbitrary object; use a data node as the forbidden out target.
        g=torch.fx.Graph();x=g.placeholder("x");y=g.placeholder("y")
        n=g.call_function(torch.add,(x,y),{"out":x});g.output(n)
        with self.assertRaisesRegex(ValueError,"out mutation"):
            adapter.normalize_common(torch.fx.GraphModule(torch.nn.Module(),g),[dict(name=n,shape=[2]) for n in ("x","y")])

    def test_eval_functional_dropout_identity(self):
        for fn in (F.dropout,F.dropout1d,F.dropout2d,F.dropout3d,F.alpha_dropout,F.feature_alpha_dropout):
            with self.subTest(fn=fn):
                model=graph(fn,names=("x",),kwargs={"p":0.25,"training":False,"inplace":False})
                dims=(1,2,4) if fn is F.dropout1d else (1,2,1,2,2) if fn is F.dropout3d else (1,2,2,2)
                x=torch.arange(8,dtype=torch.float64).reshape(dims)
                out,record=adapter.normalize_common(model,[dict(name="x",shape=list(x.shape))])
                torch.testing.assert_close(out(x),model(x),atol=0,rtol=0)
                self.assertEqual(len(record["entries"]),1)

    def test_training_inplace_invalid_probability_rejected(self):
        for kwargs,reason in (({"training":True,"p":0},"Only inference"),
                              ({"training":False,"inplace":True},"In-place"),
                              ({"training":False,"p":-0.1},"probability"),
                              ({"training":False,"p":True},"Finite public scalar")):
            with self.subTest(kwargs=kwargs):
                model=graph(F.dropout,names=("x",),kwargs=kwargs)
                with self.assertRaisesRegex(ValueError,reason):
                    adapter.normalize_common(model,[dict(name="x",shape=[2])])

    def test_eval_dropout_module_identity_and_training_rejection(self):
        root=torch.nn.Module();root.drop=torch.nn.Dropout2d(0.2).eval()
        g=torch.fx.Graph();x=g.placeholder("x");y=g.call_module("drop",(x,));g.output(y)
        model=torch.fx.GraphModule(root,g)
        out,record=adapter.normalize_common(model,[dict(name="x",shape=[1,2,2,2])])
        x=torch.arange(8,dtype=torch.float64).reshape(1,2,2,2)
        torch.testing.assert_close(out(x),model(x),atol=0,rtol=0)
        root.drop.train();model=torch.fx.GraphModule(root,g)
        with self.assertRaisesRegex(ValueError,"Only inference"):
            adapter.normalize_common(model,[dict(name="x",shape=[1,2,2,2])])

    def test_dense_sdpa_three_shapes_and_determinism(self):
        shapes=(([2,3],[4,3],[4,2]),([1,2,2,3],[1,2,4,3],[1,2,4,2]),
                ([2,1,2,3],[1,2,4,3],[2,2,4,2]))
        for dims in shapes:
            with self.subTest(dims=dims):
                model=graph(F.scaled_dot_product_attention,names=("q","k","v"),kwargs={"dropout_p":0.,"is_causal":False})
                inputs=[dict(name=n,shape=d) for n,d in zip(("q","k","v"),dims)]
                first,r1=adapter.normalize_common(model,inputs);second,r2=adapter.normalize_common(model,inputs)
                generator=torch.Generator().manual_seed(183)
                tensors=[torch.randn(d,dtype=torch.float64,generator=generator)*0.25 for d in dims]
                torch.testing.assert_close(first(*tensors),model(*tensors),atol=1e-12,rtol=1e-12)
                self.assertEqual(canonical(r1),canonical(r2));self.assertEqual(str(first.graph),str(second.graph))

    def test_sdpa_mask_causal_dropout_and_shapes_rejected(self):
        for kwargs in ({"is_causal":True},{"attn_mask":True},{"dropout_p":0.25}):
            with self.subTest(kwargs=kwargs):
                model=graph(F.scaled_dot_product_attention,names=("q","k","v"),kwargs=kwargs)
                with self.assertRaises(ValueError):
                    adapter.normalize_common(model,[dict(name=n,shape=[2,2]) for n in ("q","k","v")])
        model=graph(F.scaled_dot_product_attention,names=("q","k","v"))
        for dims in (([2,3],[4,2],[4,2]),([2,3],[4,3],[3,2]),([256,2],[512,2],[512,2])):
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    adapter.normalize_common(model,[dict(name=n,shape=d) for n,d in zip(("q","k","v"),dims)])

    def test_spatial_dropout_invalid_rank_rejected(self):
        model=graph(F.dropout1d,names=("x",),kwargs={"training":False})
        with self.assertRaisesRegex(ValueError,"Spatial Dropout input rank"):
            adapter.normalize_common(model,[dict(name="x",shape=[1,2,2,2])])

    def test_unknown_calls_unchanged(self):
        model=graph(torch.flip,args=lambda p:(p[0],(0,)),names=("x",))
        result,record=adapter.normalize_common(model,[dict(name="x",shape=[2])])
        self.assertEqual(str(model.graph),str(result.graph));self.assertEqual(record["entries"],[])
