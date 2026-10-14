"""Overlay request/CLI/application boundaries. All stdlib; no capture or backend."""
import ast,contextlib,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import public_math_context as m
from test_controls import fixture
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
@contextlib.contextmanager
def overlay_contract():
 weight=load("entry_weight",HERE/"overlay/public_weight_context.py")
 with patch.dict(sys.modules,{"public_weight_context":weight}):
  unified=load("entry_unified",HERE/"overlay/unified_graph_contract.py")
  with patch.dict(sys.modules,{"unified_graph_contract":unified}):
   yield load("entry_component",BASE/"component_contract.py"),weight
def read_model_function():
 tree=ast.parse((HERE/"overlay/run_candidate.py").read_text())
 fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="read_model")
 module=ast.Module(body=[fun],type_ignores=[])
 env=dict(require=m.require,read_json=lambda p:json.loads(p.read_text()))
 exec(compile(ast.fix_missing_locations(module),"overlay-read-model","exec"),env)
 return env["read_model"]
def budget_expression(name,target):
 tree=ast.parse((HERE/"overlay"/name).read_text())
 selected=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets)
  and isinstance(n.value,ast.IfExp) and "PUBLIC_WEIGHT_POLICIES" in ast.unparse(n.value)]
 assert len(selected)==1
 return compile(ast.Expression(selected[0]),"overlay-provenance-budget","eval")
class EntryControls(unittest.TestCase):
 def request(self,name):
  q=fixture(name)
  with overlay_contract() as (component,weights):
   new=component.prepare_task(q["model"],dict(generation_guidance="explicit-v11",
    public_weight_policy=m.POLICY,capacity_packing_version="capacity-public-tensors-v9"))
   self.assertEqual(component.reconstruct_request(new),new)
   self.assertEqual(new,m.upgrade(q));self.assertEqual(weights.verify_context(weights.project(new)),weights.project(new))
   self.assertIn("--public-weights",component.runner_options(new))
   self.assertIn(m.POLICY,component.runner_options(new))
 def test_component_module_roundtrip(self):self.request("tanh_module")
 def test_component_functional_roundtrip(self):self.request("tanh_functional")
 def test_component_keyword_roundtrip(self):self.request("tanh_input_keyword")
 def test_real_chunked_vector_request_preserved(self):
  model=dict(format="poseidon-model-graph-v2",id="nonrepeating_vector",
   resource_profile="general-tensors-v1",inputs=[dict(name="x",shape=[8])],
   constants=dict(gain=[.13,.27,-.41,.59,.61,-.73,.83,.97]),
   nodes=[dict(id="product",op="multiply",inputs=["x","gain"],attrs={},outputs=["r"])],
   outputs=[dict(name="result",value="r")])
  with overlay_contract() as (component,weights):
   opts=dict(chunk_period=4,generation_guidance="explicit-v11",capacity_packing_version="capacity-public-tensors-v9")
   legacy=component.prepare_task(model,dict(opts,public_weight_policy="descriptors-v1"))
   new=component.prepare_task(model,dict(opts,public_weight_policy=m.POLICY))
   self.assertEqual(new["layout"],legacy["layout"]);self.assertEqual(new["constant_origins"],legacy["constant_origins"])
   self.assertEqual(new["public_constants"],legacy["public_constants"]);self.assertEqual(component.reconstruct_request(new),new)
   role=new["public_weight_binding"]["math_roles"]
   self.assertEqual(role["constants"],[]);self.assertEqual(role["opaque_constants"][0]["descriptor"]["elements"],8)
   self.assertTrue(new["constant_origins"]["capacity_tensor_packing"]["bindings"])
   weights.verify_context(weights.project(new))
 def test_cli_reads_v1_and_v2(self):
  with overlay_contract() as (_,weights):
   read_model=read_model_function();q=fixture()["model"]
   with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    path=Path(tmp)/"model.json";path.write_bytes(m.canonical(q))
    self.assertEqual(read_model(path,"descriptors-v1"),q);self.assertEqual(read_model(path,m.POLICY),q)
 def test_cli_rejects_unknown_policy(self):
  with overlay_contract():
   with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    path=Path(tmp)/"model.json";path.write_bytes(m.canonical(fixture()["model"]))
    with self.assertRaises(ValueError):read_model_function()(path,"unknown")
 def test_cli_rejects_symlink(self):
  with overlay_contract():
   with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    path=Path(tmp)/"model.json";path.write_bytes(m.canonical(fixture()["model"]));alias=Path(tmp)/"alias.json";alias.symlink_to(path)
    with self.assertRaises(ValueError):read_model_function()(alias,m.POLICY)
 def test_cli_four_mib_ceiling_retained(self):
  class TooLarge:
   def is_file(self):return True
   def is_symlink(self):return False
   def stat(self):return type("Size",(),dict(st_size=4*1024**2+1))()
   def read_text(self):raise AssertionError("oversized file must not read")
  with overlay_contract():
   for policy in ("descriptors-v1",m.POLICY):
    with self.assertRaises(ValueError):read_model_function()(TooLarge(),policy)
 def budget(self,name,target,key):
  expression=budget_expression(name,target)
  env=dict(TRUSTED_BYTES=4*1024**2,PUBLIC_WEIGHT_POLICIES=("descriptors-v1",m.POLICY))
  for policy in (None,"descriptors-v1",m.POLICY,"unknown"):
   options={"public_weight_policy":policy}
   env.update(options=options,spec={"requirements":{"options":options}})
   actual=eval(expression,env)
   self.assertEqual(actual,4*1024**2 if policy in env["PUBLIC_WEIGHT_POLICIES"] else 131072)
 def test_application_import_provenance_budget(self):self.budget("application_component.py","max_bytes","options")
 def test_application_source_provenance_budget(self):self.budget("application_model_source.py","limit","spec")
 def test_cli_choices_explicit_only(self):
  tree=ast.parse((HERE/"overlay/run_candidate.py").read_text())
  call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(isinstance(a,ast.Constant) and a.value=="--public-weights" for a in n.args))
  choices=next(k.value for k in call.keywords if k.arg=="choices")
  self.assertEqual(ast.literal_eval(choices),("descriptors-v1",m.POLICY))
 def test_feedback_policy_name_not_shadowed(self):
  for name in ("application_component.py","application_model_source.py"):
   tree=ast.parse((HERE/"overlay"/name).read_text())
   imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module=="public_weight_context"]
   for n in imports:
    for alias in n.names:
     if alias.name=="POLICIES":self.assertEqual(alias.asname,"PUBLIC_WEIGHT_POLICIES")
if __name__=="__main__":unittest.main()
