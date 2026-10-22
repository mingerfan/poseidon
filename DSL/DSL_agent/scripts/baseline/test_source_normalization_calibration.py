"""Normalization domain projections; original forward and dtype are retained."""
import copy,unittest
import torch
from torch.nn import functional as F
from source_calibration_contract import checked,FX_COMPOUND_FORMAT,FX_MODULE_FORMAT,CalibrationError
from source_fx_observer import FXObserver,graph_description
from source_normalization_calibration import parameters,variance
import test_source_approximation_preparation as fixture
def describe():return dict(format=FX_COMPOUND_FORMAT,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])
class Root(torch.nn.Module):
    def __init__(self,module):super().__init__();self.norm=module
    def forward(self,x):return self.norm(x)
class GroupFunction(torch.nn.Module):
    def forward(self,x):return F.group_norm(x,2,eps=1e-4)
class InstanceFunction(torch.nn.Module):
    def forward(self,x):return F.instance_norm(x,eps=1e-4)
class ProjectionTests(unittest.TestCase):
    def observer(self,m,selection=None):return FXObserver(m,selection or describe(),tensor_elements=128,max_compute_nodes=64)
    def selected(self,m,kind):
        o=self.observer(m);n=next(n for n in o.description["nodes"]if n["op"] in("call_module","call_function"))
        return dict(describe(),mode="observe",graph_sha256=o.description["graph_sha256"],selectors=[
         dict(id="variance",node=n["name"],op=n["op"],target=n["target"],point="input",argument=0,
          projection={"kind":kind+"_variance"},domain=[1e-4,4.])])
    def test_group_module_and_function_three_shapes_native_dtype(self):
        for shape,groups in(((1,4,2),2),((2,4,2),2),((1,4,2,2),1)):
            for dtype in(torch.float32,torch.float64):
                m=Root(torch.nn.GroupNorm(groups,4,eps=1e-4,affine=False)).eval()
                self.check(m,shape,dtype,"group_norm",lambda x:x.reshape(x.shape[0],groups,-1).var(-1,unbiased=False,keepdim=True)+1e-4)
        self.check(GroupFunction(),(1,4,2),torch.float64,"group_norm",lambda x:x.reshape(1,2,-1).var(-1,unbiased=False,keepdim=True)+1e-4)
    def check(self,m,shape,dtype,kind,expected):
        x=(torch.arange(torch.tensor(shape).prod().item(),dtype=dtype)/20).reshape(shape)
        before=x.clone();sel=self.selected(m,kind);o=self.observer(m,sel)
        with o:o.begin();actual=o.execute(x);events=o.observations()
        self.assertTrue(torch.equal(actual,m(x)));self.assertTrue(torch.equal(x,before))
        result=expected(x);row=events["variance"][0]
        self.assertEqual(row["shape"],list(result.shape))
        self.assertEqual(row["dtype"],str(dtype).removeprefix("torch."))
        self.assertAlmostEqual(row["minimum"],float(result.min()),places=12)
        self.assertAlmostEqual(row["maximum"],float(result.max()),places=12)
    def test_instance_module_batched_and_unbatched_input_stats(self):
        for rank,constructor in((1,torch.nn.InstanceNorm1d),(2,torch.nn.InstanceNorm2d)):
            for batched in(False,True):
                shape=(2,3)if rank==1 else(2,2,3)
                if batched:shape=(1,)+shape
                m=Root(constructor(2,eps=1e-4,affine=False)).eval()
                def expected(x):
                    if not batched:x=x.unsqueeze(0)
                    return x.var(tuple(range(2,x.ndim)),unbiased=False,keepdim=True)+1e-4
                self.check(m,shape,torch.float64,"instance_norm",expected)
        self.check(InstanceFunction(),(1,2,3),torch.float32,"instance_norm",
          lambda x:x.var((2,),unbiased=False,keepdim=True)+1e-4)
    def test_running_stats_not_mistaken_for_input_variance(self):
        m=Root(torch.nn.InstanceNorm1d(2,eps=1e-4,track_running_stats=True)).eval()
        node=next(n for n in self.observer(m).description["nodes"]if n["op"]=="call_module")
        self.assertIs(parameters(node)["input_statistics"],False)
        with self.assertRaises(CalibrationError):self.observer(m,self.selected(m,"instance_norm"))
    def test_groups_epsilon_and_stats_mode_change_graph_binding(self):
        m=Root(torch.nn.GroupNorm(2,4,eps=1e-4));old=self.observer(m).description
        m.norm.num_groups=1;new=graph_description(torch.fx.symbolic_trace(m),source_locations=True,compound=True)
        self.assertNotEqual(old,new)
        m.norm.eps=.1;self.assertNotEqual(new,graph_description(torch.fx.symbolic_trace(m),source_locations=True,compound=True))
        m=Root(torch.nn.InstanceNorm1d(2));old=self.observer(m).description
        m.norm.track_running_stats=True
        self.assertNotEqual(old,graph_description(torch.fx.symbolic_trace(m),source_locations=True,compound=True))
    def test_v2_remains_unknown_for_the_new_module_metadata(self):
        m=Root(torch.nn.GroupNorm(2,4));g=graph_description(torch.fx.symbolic_trace(m),source_locations=True)
        n=next(n for n in g["nodes"]if n["op"]=="call_module")
        self.assertIsNone(n["source"]["module"]["type"])
        sel=self.selected(m,"group_norm");sel["format"]=FX_MODULE_FORMAT
        with self.assertRaises(CalibrationError):checked(sel)
    def test_dynamic_groups_and_wrong_projection_rejected(self):
        node=next(n for n in self.observer(GroupFunction()).description["nodes"]if n["op"]=="call_function")
        for value in(None,True,{"node":"dynamic"},0):
            bad=copy.deepcopy(node);bad["args"]["tuple"][1]=value
            with self.assertRaises(CalibrationError):parameters(bad)
        with self.assertRaises(CalibrationError):variance(torch.ones(1,4,2),node,{"kind":"instance_norm_variance"})
    def test_channel_group_shape_mismatch_rejected(self):
        node=next(n for n in self.observer(Root(torch.nn.GroupNorm(2,4))).description["nodes"]if n["op"]=="call_module")
        with self.assertRaises(CalibrationError):variance(torch.ones(1,2,2),node,{"kind":"group_norm_variance"})
    def test_wrong_operand_and_raw_tensor_do_not_become_variance(self):
        sel=self.selected(Root(torch.nn.GroupNorm(2,4)),"group_norm")
        for arg in(1,False,"num_groups"):
            bad=copy.deepcopy(sel);bad["selectors"][0]["argument"]=arg
            with self.assertRaises(CalibrationError):checked(bad)

class FreezeTests(fixture.FreezeTests):
    def setUp(self):
        super().setUp()
        self.graph.update(format="poseidon-source-calibration-fx-graph-v3")
        self.graph["nodes"]=[dict(name="norm",op="call_module",target="norm",source=dict(module_paths=["norm"],
          module=dict(type="torch.nn.GroupNorm",attributes=dict(num_groups=2,num_channels=4,eps=1e-4,affine=False))))]
        self.request["selection"].update(format=FX_COMPOUND_FORMAT)
        self.request["selection"]["selectors"][0].update(node="norm",op="call_module",target="norm",projection={"kind":"group_norm_variance"})
        self.report["observations"]["value"].update(op="call_module",target="norm",minimum=.01,maximum=.04)
        self.config=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[
          dict(path="norm",function="rsqrt",profile=dict(method="newton",domain=[1e-4,4.],iterations=2))])
    def call(self,**kw):return super().call(**dict(compound_components=True,**kw))
    def test_group_variance_resolves_rsqrt_at_bound_module(self):
        r=self.call();row=r["source_locations"][0]
        self.assertEqual(row["operator"],"group_norm");self.assertEqual(row["function"],"rsqrt")
        self.assertEqual(row["source_parameters"]["groups"],2)
    def test_instance_stats_parameters_and_shape_are_bound(self):
        m=self.graph["nodes"][0]["source"]["module"]
        m.update(type="torch.nn.InstanceNorm1d",attributes=dict(num_features=4,eps=1e-4,momentum=.1,affine=False,
          track_running_stats=False,training=False))
        self.request["selection"]["selectors"][0]["projection"]={"kind":"instance_norm_variance"}
        row=self.call()["source_locations"][0]
        self.assertEqual(row["operator"],"instance_norm")
        self.assertTrue(row["source_parameters"]["input_statistics"])
    def test_raw_tensor_and_wrong_domain_rejected(self):
        self.request["selection"]["selectors"][0]["projection"]={"kind":"tensor"}
        with self.assertRaises(ValueError):self.call()
    def test_domain_is_not_automatically_expanded(self):
        self.report["observations"]["value"]["maximum"]=5.
        with self.assertRaises(ValueError):self.call()
for name in fixture.FreezeTests.__dict__:
    if name.startswith("test_")and name not in FreezeTests.__dict__:setattr(FreezeTests,name,None)
if __name__=="__main__":unittest.main(verbosity=2)
