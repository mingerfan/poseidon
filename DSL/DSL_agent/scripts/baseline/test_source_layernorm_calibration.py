"""LayerNorm's actual variance+epsilon, frozen profile and original output controls."""
import copy,unittest
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from source_calibration_contract import checked,CalibrationError,FX_MODULE_FORMAT,FX_FORMAT
from source_fx_observer import FXObserver,graph_description
from source_fx_metadata import layer_norm_parameters
import test_source_approximation_preparation as fixture
def describe():return dict(format=FX_MODULE_FORMAT,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])
class Norm(torch.nn.Module):
    def __init__(self,shape,eps,dtype):super().__init__();self.norm=torch.nn.LayerNorm(shape,eps=eps,elementwise_affine=False,dtype=dtype)
    def forward(self,x):return self.norm(x)
class Functional(torch.nn.Module):
    def forward(self,x):return F.layer_norm(x,(2,),eps=1e-4)
class ProjectionTests(unittest.TestCase):
    def observer(self,m,selection=None):return FXObserver(m,selection or describe(),tensor_elements=64,max_compute_nodes=64)
    def selected(self,m,target="norm"):
        o=self.observer(m);n=next(n for n in o.description["nodes"]if n["target"]==target)
        return dict(describe(),mode="observe",graph_sha256=o.description["graph_sha256"],selectors=[
          dict(id="variance",node=n["name"],op=n["op"],target=n["target"],point="input",argument=0,
              projection={"kind":"layer_norm_variance"},domain=None)])
    def test_three_normalized_shapes_and_native_precision(self):
        for shape,norm,dtype in(((2,),(2,),torch.float32),((2,2),(2,),torch.float64),((1,2,2),(2,2),torch.float64)):
            with self.subTest(shape=shape,dtype=dtype):
                m=Norm(norm,1e-4,dtype).eval();sel=self.selected(m);o=self.observer(m,sel)
                x=torch.arange(4 if len(shape)>1 else 2,dtype=dtype).reshape(shape)/10
                with o:o.begin();out=o.execute(x);events=o.observations()
                self.assertTrue(torch.equal(out,m(x)))
                expected=x.var(dim=tuple(range(x.ndim-len(norm),x.ndim)),unbiased=False,keepdim=True)+1e-4
                row=events["variance"][0]
                self.assertEqual(row["dtype"],str(dtype).removeprefix("torch."))
                self.assertEqual(row["minimum"],float(expected.min()));self.assertEqual(row["maximum"],float(expected.max()))
    def test_functional_parameters_and_native_output(self):
        m=Functional();sel=self.selected(m,"torch.nn.functional.layer_norm");o=self.observer(m,sel)
        x=torch.tensor([[.1,.4],[-.2,.3]],dtype=torch.float64)
        with o:o.begin();got=o.execute(x);events=o.observations()
        self.assertTrue(torch.equal(got,m(x)));self.assertEqual(events["variance"][0]["minimum"],float((x.var(-1,unbiased=False,keepdim=True)+1e-4).min()))
    def test_epsilon_and_shape_changes_are_bound_to_graph(self):
        m=Norm((2,),1e-4,torch.float64);o=self.observer(m);before=o.description
        m.norm.eps=.1
        self.assertNotEqual(before,graph_description(o.traced,source_locations=True))
        with self.assertRaises(CalibrationError):
            with o:pass
    def test_projection_for_wrong_operation_or_operand_is_rejected(self):
        m=torch.nn.Sequential(torch.nn.Sigmoid());sel=self.selected(m,"0")
        with self.assertRaises(CalibrationError):self.observer(m,sel)
        sel=self.selected(Norm((2,),1e-4,torch.float64));sel["selectors"][0]["argument"]=1
        with self.assertRaises(CalibrationError):checked(sel)
    def test_v1_does_not_silently_accept_compound_projection(self):
        sel=self.selected(Norm((2,),1e-4,torch.float64));sel["format"]=FX_FORMAT
        with self.assertRaises(CalibrationError):checked(sel)
class FreezeTests(fixture.FreezeTests):
    def setUp(self):
        super().setUp()
        self.graph.update(format="poseidon-source-calibration-fx-graph-v2")
        self.graph["nodes"]=[dict(name="norm",op="call_module",target="norm",source=dict(module_paths=["norm"],
           module=dict(type="torch.nn.LayerNorm",attributes=dict(normalized_shape=[2],eps=1e-4))))]
        self.request["selection"].update(format=FX_MODULE_FORMAT)
        self.request["selection"]["selectors"][0].update(node="norm",op="call_module",target="norm",projection={"kind":"layer_norm_variance"})
        self.report["observations"]["value"].update(op="call_module",target="norm",minimum=.01,maximum=.04)
        self.config=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[
           dict(path="norm",function="rsqrt",profile=dict(method="newton",domain=[1e-4,4.],iterations=2))])
    def test_observed_variance_freezes_layer_norm_rsqrt_at_module_path(self):
        result=self.call();row=result["approximation_selection"]["selections"][0]
        self.assertEqual(row["operator"],"layer_norm");self.assertEqual(row["function"],"rsqrt")
        self.assertEqual(row["module_paths"],["norm"]);self.assertEqual(row["source"],dict(kind="modules",selector=0))
    def test_raw_input_range_cannot_substitute_for_variance(self):
        self.request["selection"]["selectors"][0]["projection"]={"kind":"tensor"}
        with self.assertRaises(ValueError):self.call()
    def test_uncovered_variance_domain_never_expands(self):
        self.report["observations"]["value"]["maximum"]=8.
        with self.assertRaises(ValueError):self.call()
for name in fixture.FreezeTests.__dict__:
    if name.startswith("test_")and name not in FreezeTests.__dict__:setattr(FreezeTests,name,None)
if __name__=="__main__":unittest.main(verbosity=2)
