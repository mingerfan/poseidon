"""Original module source metadata and explicit profile selection controls."""
import copy,json,unittest
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from source_calibration_contract import FX_FORMAT,FX_MODULE_FORMAT,checked,CalibrationError
from source_fx_observer import FXObserver,graph_description
from source_fx_metadata import checked_source
from source_preparation_service import checked as checked_request,MODULE_REQUEST,REQUEST,public_summary
import test_source_approximation_preparation as freeze_tests
def selection(fmt=FX_MODULE_FORMAT):
    return dict(format=fmt,purpose="calibration",mode="describe",graph_sha256=None,selectors=[])
class Activation(torch.nn.Module):
    def __init__(self,kind=torch.nn.GELU):super().__init__();self.activation=kind()
    def forward(self,x):return self.activation(x*.25+.1)
class Block(torch.nn.Module):
    def forward(self,x):return F.gelu(x*.25+.1)
class Nested(torch.nn.Module):
    def __init__(self):super().__init__();self.block=Block()
    def forward(self,x):return self.block(x)
class ModuleObservationTests(unittest.TestCase):
    def observer(self,module,sel=None):
        return FXObserver(module,sel or selection(),tensor_elements=64,max_compute_nodes=64)
    def test_three_shapes_and_native_dtypes_keep_original_module_values(self):
        for shape,dtype in(((2,),torch.float32),((2,2),torch.float64),((1,4),torch.float64)):
            m=Activation().eval();d=self.observer(m);n=next(n for n in d.description["nodes"]if n["op"]=="call_module")
            self.assertEqual(n["source"]["module"],dict(type="torch.nn.GELU",attributes=dict(approximate="none")))
            self.assertEqual(n["source"]["module_paths"],["activation"])
            sel=dict(selection(),mode="observe",graph_sha256=d.description["graph_sha256"],selectors=[
                dict(id="observed",node=n["name"],op=n["op"],target=n["target"],point="input",argument=0,
                     projection=dict(kind="tensor"),domain=[-1.,1.])])
            o=self.observer(m,sel);x=torch.arange(4 if shape!=(2,)else 2,dtype=dtype).reshape(shape)/10
            with o:o.begin();got=o.execute(x);events=o.observations()
            self.assertTrue(torch.equal(got,m(x)))
            self.assertEqual(events["observed"][0]["dtype"],str(dtype).removeprefix("torch."))
            self.assertEqual(events["observed"][0]["minimum"],float((x*.25+.1).min()))
    def test_function_inside_custom_module_keeps_real_source_path(self):
        o=self.observer(Nested());n=next(n for n in o.description["nodes"]if n["target"]=="torch.nn.functional.gelu")
        self.assertEqual(n["source"]["module_paths"],["block"]);self.assertIsNone(n["source"]["module"])
    def test_shared_module_calls_have_distinct_nodes_and_same_source_path(self):
        class Shared(Activation):
            def forward(self,x):return self.activation(x)+self.activation(x*.5)
        d=self.observer(Shared()).description
        nodes=[n for n in d["nodes"]if n["op"]=="call_module"]
        self.assertEqual(len(nodes),2);self.assertNotEqual(nodes[0]["name"],nodes[1]["name"])
        self.assertEqual(nodes[0]["source"],nodes[1]["source"])
    def test_module_attribute_drift_changes_graph_identity(self):
        m=Activation();o=self.observer(m);before=o.description
        m.activation.approximate="tanh"
        self.assertNotEqual(before,graph_description(o.traced,source_locations=True))
        with self.assertRaises(CalibrationError):
            with o:pass
    def test_five_activation_types_have_exact_public_class_and_attributes(self):
        for kind in(torch.nn.ReLU,torch.nn.GELU,torch.nn.SiLU,torch.nn.Sigmoid,torch.nn.Tanh):
            d=self.observer(Activation(kind)).description;n=next(n for n in d["nodes"]if n["op"]=="call_module")
            self.assertEqual(n["source"]["module"]["type"],"torch.nn."+kind.__name__)
    def test_v1_graph_fields_and_module_rejection_stay_explicit(self):
        o=self.observer(Activation(),selection(FX_FORMAT))
        self.assertTrue(all(set(n)=={"name","op","target","args","kwargs"}for n in o.description["nodes"]))
        value=dict(selection(FX_FORMAT),mode="observe",graph_sha256=o.description["graph_sha256"],selectors=[
          dict(id="a",node="activation",op="call_module",target="activation",point="input",argument=0,
               projection={"kind":"tensor"},domain=None)])
        with self.assertRaises(CalibrationError):checked(value)
    def test_numeric_module_path_and_unknown_target_are_checked(self):
        value=dict(selection(),mode="observe",graph_sha256="a"*64,selectors=[
          dict(id="a",node="_0",op="call_module",target="layers.0",point="input",argument=0,
               projection={"kind":"tensor"},domain=None)])
        self.assertEqual(checked(value),value)
        value["selectors"][0]["target"]="../host"
        with self.assertRaises(CalibrationError):checked(value)
    def test_agent_graphmodule_retrace_keeps_nested_scopes_and_computation(self):
        from pytorch_static_trace import trace,ALIAS_POLICY
        from pytorch_approximation import module_paths
        inputs=[dict(name="x",shape=[2])]
        first=trace(Nested().eval(),inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
        second=trace(first,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
        self.assertEqual(str(first.graph),str(second.graph))
        old=next(n for n in first.graph.nodes if n.op=="call_function"and n.target is F.gelu)
        new=next(n for n in second.graph.nodes if n.op=="call_function"and n.target is F.gelu)
        self.assertEqual(module_paths(old),["block"]);self.assertEqual(module_paths(new),["block"])
        value=torch.tensor([-.3,.2],dtype=torch.float64)
        torch.testing.assert_close(first(value),second(value),atol=0,rtol=0)
    def test_input_normalization_renames_do_not_drop_bare_module_source(self):
        from pytorch_static_trace import trace,ALIAS_POLICY
        from pytorch_input_binding import plan,signature_record,normalize
        from pytorch_approximation import module_paths
        class BareBlock(torch.nn.Module):
            def forward(self,x):return F.gelu(x)
        class Bare(torch.nn.Module):
            def __init__(self):super().__init__();self.block=BareBlock()
            def forward(self,x):return self.block(x)
        model=Bare().eval();inputs=[dict(name="x",shape=[2])]
        spec=dict(inputs=inputs,call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={}))
        binding=plan(spec,signature_record(model))
        first=trace(model,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY,input_plan=binding)
        normalized,_=normalize(first,binding)
        second=trace(normalized,inputs,resource_profile="general-tensors-v1",policy=ALIAS_POLICY)
        original=next(n for n in normalized.graph.nodes if n.op=="call_function"and n.target is F.gelu)
        current=next(n for n in second.graph.nodes if n.op=="call_function"and n.target is F.gelu)
        self.assertEqual(module_paths(original),["block"]);self.assertEqual(module_paths(current),["block"])
        self.assertEqual(original.args[0].target,current.args[0].target)
        value=torch.tensor([-.3,.2],dtype=torch.float64)
        torch.testing.assert_close(normalized(value),second(value),atol=0,rtol=0)
    def test_wrong_source_module_schema_is_rejected(self):
        n=next(n for n in self.observer(Activation()).description["nodes"]if n["op"]=="call_module")
        for change in({"module_paths":[]},{"module":dict(type="torch.nn.GELU",attributes={"approximate":"unknown"})},
                       {"extra":"environment"}):
            v=copy.deepcopy(n);v["source"].update(change)
            with self.assertRaises(CalibrationError):checked_source(v)
class ModuleFreezeTests(freeze_tests.FreezeTests):
    def setUp(self):
        super().setUp()
        self.graph.update(format="poseidon-source-calibration-fx-graph-v2")
        self.graph["nodes"]=[dict(name="activation",op="call_module",target="activation",source=dict(
            module_paths=["activation"],module=dict(type="torch.nn.GELU",attributes=dict(approximate="none"))))]
        self.request["selection"].update(format=FX_MODULE_FORMAT)
        self.request["selection"]["selectors"][0].update(node="activation",op="call_module",target="activation")
        self.report["observations"]["value"].update(op="call_module",target="activation")
        self.report["fx_graph"]=self.graph
        self.config=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[
          dict(path="activation",function="gelu",profile=dict(method="polynomial",basis="power",domain=[-1.,1.],coefficients=[0.,.5,.25]))])
    # Only focused controls; inherited functions test v1 through its own class.
    def test_module_profile_uses_exact_observed_path(self):
        result=self.call()
        self.assertEqual(result["format"],"poseidon-source-approximation-freeze-v2")
        row=result["approximation_selection"]["selections"][0]
        self.assertEqual(row["module_paths"],["activation"])
        self.assertEqual(row["source"],dict(kind="modules",selector=0))
    def test_function_path_selection_is_preserved(self):
        self.graph["nodes"][0].update(op="call_function",target="torch.nn.functional.gelu",source=dict(module_paths=["activation"],module=None))
        self.request["selection"]["selectors"][0].update(op="call_function",target="torch.nn.functional.gelu")
        self.report["observations"]["value"].update(op="call_function",target="torch.nn.functional.gelu")
        result=self.call()
        self.assertEqual(result["approximation_selection"]["selections"][0]["module_paths"],["activation"])
    def test_node_selector_overrides_module_and_default_without_losing_paths(self):
        profile=copy.deepcopy(self.config["modules"][0]["profile"])
        self.config["defaults"]={"gelu":copy.deepcopy(profile)}
        profile["domain"]=[-.5,.5]
        self.config["nodes"]=[dict(name="activation",function="gelu",profile=profile)]
        result=self.call();row=result["approximation_selection"]["selections"][0]
        self.assertEqual(row["source"],dict(kind="nodes",selector=0))
        self.assertEqual(row["module_paths"],["activation"])
        self.assertEqual(row["profile_sha256"],digest(profile))
    def test_overlapping_module_selectors_at_same_priority_are_rejected(self):
        self.graph["nodes"][0]["source"]["module_paths"]=["block","activation"]
        self.config["modules"].append(dict(path="block",function="gelu",profile=copy.deepcopy(self.config["modules"][0]["profile"])))
        with self.assertRaises(ValueError):self.call()
    def test_unmapped_compound_module_remains_unsupported(self):
        self.graph["nodes"][0]["source"]["module"]=dict(type="torch.nn.LayerNorm",attributes={})
        with self.assertRaises(ValueError):self.call()
    def test_v1_graph_cannot_freeze_unobserved_modules(self):
        self.graph.update(format="poseidon-source-calibration-fx-graph-v1")
        self.graph["nodes"][0].pop("source")
        with self.assertRaises(ValueError):self.call()
# Prevent unrelated v1 inherited cases from accidentally using modified fixtures.
for name in list(freeze_tests.FreezeTests.__dict__):
    if name.startswith("test_") and name not in ModuleFreezeTests.__dict__:
        setattr(ModuleFreezeTests,name,None)
if __name__=="__main__":unittest.main(verbosity=2)
