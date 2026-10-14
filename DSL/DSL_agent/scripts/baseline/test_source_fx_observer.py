"""Functional-node calibration controls; native unit tensors, no DSL/FHE/API."""
import copy, math, operator, unittest
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from source_calibration_contract import FX_FORMAT, checked, CalibrationError
from source_fx_observer import FXObserver, graph_description, summarize_fx

def describe():
    return dict(format=FX_FORMAT,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])

def selection(observer,target,*,point="input",argument=0,domain=None,op="call_function"):
    node=next(n for n in observer.description["nodes"]if n["op"]==op and n["target"]==target)
    return dict(format=FX_FORMAT,purpose="calibration",mode="observe",
        graph_sha256=observer.description["graph_sha256"],selectors=[
         dict(id="activation",node=node["name"],op=op,target=target,point=point,
              argument=argument,projection=dict(kind="tensor"),domain=domain)])

class Gelu(torch.nn.Module):
    def forward(self,x):return F.gelu(x*.5+.25)
class RMS(torch.nn.Module):
    def forward(self,x):
        square=x*x
        denom=square.mean(dim=-1,keepdim=True)+1e-4
        return x*torch.rsqrt(denom)
class MethodNorm(torch.nn.Module):
    def forward(self,x):
        denom=(x*x).mean(dim=-1,keepdim=True)+1e-4
        return x*denom.rsqrt()

class FXObservationTests(unittest.TestCase):
    def observer(self,module,sel=None,limit=64):
        return FXObserver(module,sel or describe(),tensor_elements=limit,max_compute_nodes=64)
    def evaluated(self,module,value,sel):
        o=self.observer(module,sel)
        with o:
            o.begin();one=o.execute(value);events=o.observations()
            o.begin();two=o.execute(value)
            self.assertEqual(events,o.observations())
        self.assertTrue(torch.equal(one,module(value)));self.assertTrue(torch.equal(one,two))
        return summarize_fx(sel,[events])["activation"]

    def test_graph_inventory_is_deterministic_and_binds_scalar_attributes(self):
        a=self.observer(Gelu());b=self.observer(Gelu())
        self.assertEqual(a.description,b.description)
        self.assertEqual(a.description["graph_sha256"],digest({k:v for k,v in a.description.items()if k!="graph_sha256"}))
        self.assertTrue(any(n["target"]=="torch.nn.functional.gelu"for n in a.description["nodes"]))
        class Changed(torch.nn.Module):
            def forward(self,x):return F.gelu(x*.25+.25)
        self.assertNotEqual(a.description["graph_sha256"],self.observer(Changed()).description["graph_sha256"])

    def test_gelu_inputs_in_three_contexts_and_float_dtypes(self):
        for shape,dtype in (((2,),torch.float32),((2,2),torch.float64),((1,4),torch.float64)):
            with self.subTest(shape=shape,dtype=dtype):
                x=torch.arange(math.prod(shape),dtype=dtype).reshape(shape)/4
                module=Gelu();o=self.observer(module)
                sel=selection(o,"torch.nn.functional.gelu",domain=[-.1,.1])
                r=self.evaluated(module,x,sel)
                expected=(x*.5+.25).to(torch.float64).numpy()
                self.assertEqual(r["minimum"],float(expected.min()))
                self.assertEqual(r["maximum"],float(expected.max()))
                self.assertEqual(r["dtypes"],[str(dtype).removeprefix("torch.")])
                self.assertEqual(r["domain_status"],"outside")

    def test_rsqrt_actual_operands_in_three_contexts(self):
        for shape in ((2,),(2,2),(1,4)):
            with self.subTest(shape=shape):
                x=torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)/4
                module=RMS();sel=selection(self.observer(module),"torch.rsqrt")
                r=self.evaluated(module,x,sel);expected=((x*x).mean(-1,keepdim=True)+1e-4).numpy()
                self.assertEqual(r["mean"],float(expected.mean()))
                self.assertEqual(r["values"],expected.size)

    def test_method_rsqrt_output_in_three_contexts(self):
        for shape in ((2,),(2,2),(1,4)):
            with self.subTest(shape=shape):
                x=torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)/4
                module=MethodNorm()
                sel=selection(self.observer(module),"rsqrt",point="output",argument=None,op="call_method")
                r=self.evaluated(module,x,sel)
                expected=((x*x).mean(-1,keepdim=True)+1e-4).rsqrt().numpy()
                self.assertEqual(r["minimum"],float(expected.min()))
                self.assertEqual(r["maximum"],float(expected.max()))

    def test_keyword_arguments_and_keyword_only_forward_are_bound(self):
        class Keyword(torch.nn.Module):
            def forward(self,*,x):return F.gelu(input=x)
        module=Keyword();o=self.observer(module);sel=selection(o,"torch.nn.functional.gelu",argument="input")
        current=self.observer(module,sel);x=torch.tensor([.1,.2],dtype=torch.float64)
        with current:current.begin();actual=current.execute(x=x);events=current.observations()
        self.assertTrue(torch.equal(actual,module(x=x)))
        self.assertEqual(events["activation"][0]["minimum"],.1)

    def test_structured_argument_is_kept_for_original_interpreter(self):
        class Structured(torch.nn.Module):
            def forward(self,payload):return F.gelu(payload["left"]+payload["right"])
        m=Structured();sel=selection(self.observer(m),"torch.nn.functional.gelu")
        o=self.observer(m,sel);p={"left":torch.tensor([.1]),"right":torch.tensor([.2])}
        with o:o.begin();v=o.execute(p);events=o.observations()
        self.assertTrue(torch.equal(v,m(p)));self.assertEqual(len(events["activation"]),1)

    def test_shared_function_calls_have_distinct_nodes(self):
        class Shared(torch.nn.Module):
            def forward(self,x):return F.gelu(x)+F.gelu(x*2)
        o=self.observer(Shared())
        names=[n["name"]for n in o.description["nodes"]if n["target"]=="torch.nn.functional.gelu"]
        self.assertEqual(len(names),2);self.assertNotEqual(names[0],names[1])

    def test_changed_graph_or_wrong_operator_target_is_rejected(self):
        module=Gelu();sel=selection(self.observer(module),"torch.nn.functional.gelu")
        for key,value in (("graph_sha256","0"*64),):
            bad=copy.deepcopy(sel);bad[key]=value
            with self.assertRaises(CalibrationError):self.observer(module,bad)
        for key,value in (("target","torch.tanh"),("op","call_method"),("node","missing")):
            bad=copy.deepcopy(sel);bad["selectors"][0][key]=value
            with self.subTest(key=key),self.assertRaises(CalibrationError):self.observer(module,bad)

    def test_dynamic_data_control_is_rejected(self):
        class Dynamic(torch.nn.Module):
            def forward(self,x):
                if x.sum()>0:return F.gelu(x)
                return x
        with self.assertRaises(CalibrationError)as error:self.observer(Dynamic())
        self.assertEqual(error.exception.code,"calibration_fx_trace_rejected")

    def test_non_tensor_argument_and_missing_keyword_are_rejected(self):
        module=Gelu();o=self.observer(module);sel=selection(o,"operator.mul",argument=1)
        obs=self.observer(module,sel)
        with self.assertRaises(CalibrationError):
            with obs:obs.begin();obs.execute(torch.tensor([.1]))
        sel=selection(o,"torch.nn.functional.gelu",argument="unknown")
        with self.assertRaises(CalibrationError):
            obs=self.observer(module,sel)
            with obs:obs.begin();obs.execute(torch.tensor([.1]))

    def test_tensor_capacity_and_nonfinite_observations_reject(self):
        module=Gelu();sel=selection(self.observer(module),"torch.nn.functional.gelu")
        for x in (torch.tensor([.1,.2]),torch.tensor([float("nan")])):
            with self.assertRaises(CalibrationError):
                obs=self.observer(module,sel,limit=1)
                with obs:obs.begin();obs.execute(x)

    def test_graph_mutation_at_exit_is_rejected(self):
        obs=self.observer(Gelu())
        with self.assertRaises(CalibrationError):
            with obs:
                node=next(n for n in obs.traced.graph.nodes if n.op=="call_function")
                node.target=operator.neg

    def test_state_mutation_at_exit_is_rejected(self):
        m=torch.nn.Linear(2,2,dtype=torch.float64);obs=self.observer(m)
        with self.assertRaises(CalibrationError):
            with obs:
                with torch.no_grad():next(obs.traced.parameters()).add_(1)

    def test_selector_shape_roles_and_duplicate_targets_reject(self):
        o=self.observer(Gelu());sel=selection(o,"torch.nn.functional.gelu")
        bads=[dict(sel,purpose="test"),dict(sel,purpose="validation"),dict(sel,extra="callback"),
              dict(sel,mode="describe"),dict(sel,graph_sha256=None)]
        for item in bads:
            with self.subTest(item=list(item)),self.assertRaises(CalibrationError):checked(item)
        for field,value in (("argument",True),("point","output"),("projection",dict(kind="rms_mean_square",epsilon=.1)),
                            ("domain",[1.,-1.])):
            item=copy.deepcopy(sel);item["selectors"][0][field]=value
            with self.subTest(field=field),self.assertRaises(CalibrationError):checked(item)
        item=copy.deepcopy(sel);item["selectors"].append(dict(item["selectors"][0],id="duplicate"))
        with self.assertRaises(CalibrationError):checked(item)

    def test_inventory_has_no_implicit_observations(self):
        m=Gelu();o=self.observer(m);x=torch.tensor([.1,.2])
        with o:o.begin();actual=o.execute(x);events=o.observations()
        self.assertEqual(events,{});self.assertTrue(torch.equal(actual,m(x)))
        self.assertEqual(summarize_fx(describe(),[events]),{})

if __name__=="__main__":unittest.main()
