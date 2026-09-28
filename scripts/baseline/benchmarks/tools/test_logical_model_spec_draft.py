"""Public math-spec draft checks; no Agent, compiler or encrypted execution."""
import copy,json,unittest
from logical_model_spec_draft import specification,OPERATIONS
from benchmark_graph import OPS
from workspace_paths import RESULTS
class LogicalSpec(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=RESULTS/"stage2-agent-campaign-prepare-r106"
  cls.specs=[]
  for ref in json.loads((root/"index.json").read_text())["shards"]:
   cls.specs.extend(json.loads((root/ref["file"]).read_text())["cases"])
 def test_all_registered_operators_defined(self):self.assertEqual(set(OPERATIONS),set(OPS))
 def test_all_2030_frozen_models_have_exact_used_operation_specs(self):
  self.assertEqual(len(self.specs),2030)
  for task in self.specs:
   actual=specification(task["model"])
   self.assertEqual(set(actual["operations"]),{n["op"] for n in task["model"]["nodes"]})
 def test_description_does_not_contain_model_values_or_input_arrays(self):
  allowed={"version","scope","ordering","attributes","broadcasting","axes","layout_boundary","operations"}
  for task in self.specs:
   actual=specification(task["model"]);self.assertEqual(set(actual),allowed)
   self.assertNotIn("hecate_source",actual);self.assertNotIn("reference",actual)
   self.assertTrue(all(isinstance(v,str) for v in actual["operations"].values()))
 def test_renaming_model_does_not_change_spec(self):
  model=copy.deepcopy(self.specs[0]["model"]);a=specification(model)
  model["id"]="new_public_name";self.assertEqual(specification(model),a)
 def test_logical_rotation_differs_from_period_rotation_in_three_contexts(self):
  for n,p,step in [(63,64,-2),(3,4,1),(7,8,-3)]:
   values=list(range(1,n+1));logical=[values[(j+step)%n] for j in range(n)]
   padded=values+[0]*(p-n);physical=[padded[(j+step)%p] for j in range(n)]
   self.assertNotEqual(logical,physical)
  self.assertIn("modulo N",OPERATIONS["rotate"]);self.assertIn("not the padded",OPERATIONS["rotate"])
 def test_unknown_model_operation_rejected(self):
  model=copy.deepcopy(self.specs[0]["model"]);model["nodes"][0]["op"]="unknown"
  with self.assertRaises(ValueError):specification(model)
if __name__=="__main__":unittest.main()
