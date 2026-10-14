"""Declared compound statistics and legacy controls; no API or ciphertext."""
import copy,math,unittest
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from source_fx_observer import FXObserver,graph_description
from source_calibration_contract import FX_COMPOUND_FORMAT,FX_MODULE_FORMAT,FX_FORMAT,checked,CalibrationError
from source_compound_calibration import ARITHMETIC,soft_parameters,checked_projection,exp_sum
import test_source_approximation_preparation as fixture

EXP=dict(method="taylor",domain=[-1.,1.],degree=2)
def describe(version=FX_COMPOUND_FORMAT):return dict(format=version,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])
def projection(exp=None):return dict(kind="softmax_exp_sum",exp=copy.deepcopy(EXP if exp is None else exp),arithmetic=ARITHMETIC)
class Model(torch.nn.Module):
    def __init__(self,dim,log=False):super().__init__();self.norm=(torch.nn.LogSoftmax if log else torch.nn.Softmax)(dim=dim)
    def forward(self,x):return self.norm(x)
class Functional(torch.nn.Module):
    def forward(self,x):return F.softmax(x,dim=-1)
class LogFunctional(torch.nn.Module):
    def forward(self,x):return F.log_softmax(x,dim=0,dtype=torch.float64)
class Method(torch.nn.Module):
    def forward(self,x):return x.softmax(1)
class LogMethod(torch.nn.Module):
    def forward(self,x):return x.log_softmax(-1)
class TorchFunction(torch.nn.Module):
    def forward(self,x):return torch.softmax(x,1,dtype=torch.float64)
class CompoundProjectionTests(unittest.TestCase):
    def observer(self,m,selection=None):return FXObserver(m,selection or describe(),tensor_elements=64,max_compute_nodes=64)
    def selected(self,m):
        o=self.observer(m);n=next(n for n in o.description["nodes"]if n["op"] in("call_module","call_function","call_method"))
        return dict(describe(),mode="observe",graph_sha256=o.description["graph_sha256"],selectors=[
          dict(id="input",node=n["name"],op=n["op"],target=n["target"],point="input",argument=0,projection={"kind":"tensor"},domain=[-1.,1.]),
          dict(id="sum",node=n["name"],op=n["op"],target=n["target"],point="input",argument=0,projection=projection(),domain=[.1,8.])])
    def test_declared_denominator_in_six_original_source_forms(self):
        for m in(Model(-1),Model(0,True),Functional(),LogFunctional(),Method(),LogMethod(),TorchFunction()):
            for dtype in(torch.float32,torch.float64):
                with self.subTest(model=type(m).__name__,dtype=dtype):
                    sel=self.selected(m);o=self.observer(m,sel);x=torch.tensor([[.1,.2,.3],[-.2,.4,.5]],dtype=dtype)
                    with o:o.begin();got=o.execute(x);events=o.observations()
                    self.assertTrue(torch.equal(got,m(x)))
                    node=next(n for n in o.description["nodes"]if n["name"]==sel["selectors"][0]["node"])
                    params=soft_parameters(node);a=x.numpy().astype(np.float64)
                    want=(1+a+a*a/2).sum(axis=params["dim"],keepdims=True)
                    row=events["sum"][0]
                    self.assertEqual(row["dtype"],"float64");self.assertEqual(row["shape"],list(want.shape))
                    self.assertAlmostEqual(row["minimum"],float(want.min()),places=12)
                    self.assertAlmostEqual(row["maximum"],float(want.max()),places=12)
                    self.assertEqual(events["input"][0]["dtype"],str(dtype).removeprefix("torch."))
    def test_axis_changes_bind_v3_but_legacy_v2_stays_identical(self):
        for log in(False,True):
            a=torch.fx.symbolic_trace(Model(0,log));b=torch.fx.symbolic_trace(Model(1,log))
            self.assertEqual(graph_description(a,source_locations=True),graph_description(b,source_locations=True))
            self.assertNotEqual(graph_description(a,source_locations=True,compound=True),graph_description(b,source_locations=True,compound=True))
    def test_axis_mutation_after_observer_creation_rejected(self):
        m=Model(-1);o=self.observer(m);m.norm.dim=0
        with self.assertRaises(CalibrationError):
            with o:pass
    def test_float64_derived_arithmetic_does_not_change_native_forward(self):
        m=Model(1);sel=self.selected(m);o=self.observer(m,sel)
        x=torch.tensor([[.3,.4],[-.2,.1]],dtype=torch.float32);before=x.clone()
        with o:o.begin();actual=o.execute(x)
        self.assertEqual(actual.dtype,torch.float32);self.assertTrue(torch.equal(x,before))
    def test_native_exp_is_not_used_for_declared_denominator(self):
        x=torch.tensor([[.5,.6]],dtype=torch.float64);m=Model(-1)
        n=next(n for n in self.observer(m).description["nodes"]if n["op"]=="call_module")
        actual=exp_sum(x,n,projection())
        self.assertFalse(torch.equal(actual,x.exp().sum(-1,keepdim=True)))
    def test_wrong_operation_operand_and_old_contract_rejected(self):
        sel=self.selected(Model(-1))
        for version in(FX_FORMAT,FX_MODULE_FORMAT):
            bad=copy.deepcopy(sel);bad["format"]=version
            with self.assertRaises(CalibrationError):checked(bad)
        for argument in(1,False,"dim"):
            bad=copy.deepcopy(sel);bad["selectors"][1]["argument"]=argument
            with self.assertRaises(CalibrationError):checked(bad)
        bad=copy.deepcopy(sel);bad["selectors"][1]["target"]="torch.relu";bad["selectors"][1]["op"]="call_function"
        with self.assertRaises(CalibrationError):self.observer(torch.nn.Sequential(torch.nn.ReLU()),bad)
    def test_implicit_dynamic_axis_and_dtype_rejected(self):
        node=next(n for n in self.observer(Functional()).description["nodes"]if n["op"]=="call_function")
        for value in(None,True,{"node":"dynamic"},33):
            bad=copy.deepcopy(node);bad["kwargs"]["dict"]["dim"]=value
            with self.assertRaises(CalibrationError):soft_parameters(bad)
        bad=copy.deepcopy(node);bad["kwargs"]["dict"]["dtype"]={"dtype":"torch.int64"}
        with self.assertRaises(CalibrationError):soft_parameters(bad)
    def test_invalid_profile_and_arithmetic_rejected(self):
        for change in(dict(arithmetic="native"),dict(exp=dict(method="taylor",degree=True,domain=[-1.,1.])),dict(extra=True)):
            bad=projection();bad.update(change)
            with self.assertRaises(CalibrationError):checked_projection(bad)
    def test_resource_limit_and_nonfinite_projection_rejected(self):
        m=Model(-1);sel=self.selected(m)
        with self.assertRaises(CalibrationError):FXObserver(m,sel,tensor_elements=1,max_compute_nodes=64).observe(sel["selectors"][1],torch.ones(2))
    def test_softmax_and_logsoftmax_component_names_are_bound(self):
        self.assertEqual(soft_parameters(next(n for n in self.observer(Model(1)).description["nodes"]if n["op"]=="call_module"))["operator"],"softmax")
        self.assertEqual(soft_parameters(next(n for n in self.observer(Model(1,True)).description["nodes"]if n["op"]=="call_module"))["operator"],"logsoftmax")

class CompoundFreezeTests(fixture.FreezeTests):
    def setUp(self):
        super().setUp()
        self.graph.update(format="poseidon-source-calibration-fx-graph-v3")
        self.graph["nodes"]=[dict(name="norm",op="call_module",target="norm",
         source=dict(module_paths=["norm"],module=dict(type="torch.nn.Softmax",attributes=dict(dim=-1))))]
        self.request["selection"].update(format=FX_COMPOUND_FORMAT,selectors=[
         dict(id="raw",node="norm",op="call_module",target="norm",point="input",argument=0,projection={"kind":"tensor"}),
         dict(id="total",node="norm",op="call_module",target="norm",point="input",argument=0,projection=projection())])
        self.report["observations"]={
         "raw":dict(op="call_module",target="norm",minimum=-.4,maximum=.4,samples=3,values=6),
         "total":dict(op="call_module",target="norm",minimum=1.8,maximum=2.2,samples=3,values=3)}
        self.config=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[
         dict(path="norm",function="exp",profile=copy.deepcopy(EXP)),
         dict(path="norm",function="reciprocal",profile=dict(method="goldschmidt",iterations=2,domain=[.5,4.]))])
    def call(self,**kw):return super().call(**dict(compound_components=True,**kw))
    def test_two_component_profiles_and_domains_freeze(self):
        result=self.call();rows=result["source_locations"]
        self.assertEqual(result["format"],"poseidon-source-approximation-freeze-v3")
        self.assertEqual([r["function"]for r in rows],["exp","reciprocal"])
        self.assertEqual([r["source_parameters"]["dim"]for r in rows],[-1,-1])
        self.assertFalse(result["profile_fitted"]);self.assertFalse(result["ckks_validation_performed"])
    def test_logsoftmax_selects_exp_and_log(self):
        self.graph["nodes"][0]["source"]["module"]["type"]="torch.nn.LogSoftmax"
        self.config["modules"][1].update(function="log",profile=dict(method="polynomial",basis="power",domain=[.5,4.],coefficients=[0.,.5]))
        result=self.call();self.assertEqual([r["function"]for r in result["source_locations"]],["exp","log"])
    def test_missing_component_and_wrong_exp_derivative_rejected(self):
        for variant in("missing","wrong"):
            self.setUp()
            if variant=="missing":
                self.request["selection"]["selectors"].pop()
                self.report["observations"].pop("total")
            else:self.request["selection"]["selectors"][1]["projection"]["exp"]["degree"]=3
            with self.assertRaises(ValueError):self.call()
    def test_separate_domains_do_not_expand(self):
        self.report["observations"]["total"]["maximum"]=5.
        with self.assertRaises(ValueError):self.call()
    def test_v2_does_not_accept_v3_evidence(self):
        with self.assertRaises(ValueError):super().call()
for name in fixture.FreezeTests.__dict__:
    if name.startswith("test_")and name not in CompoundFreezeTests.__dict__:setattr(CompoundFreezeTests,name,None)
if __name__=="__main__":unittest.main(verbosity=2)
