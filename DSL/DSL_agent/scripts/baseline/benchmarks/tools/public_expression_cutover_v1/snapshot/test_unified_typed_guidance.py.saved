"""v6 metadata cannot widen syntax or replace a required semantic operation."""
import copy, json, unittest
from benchmark_suite import Builder
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare, validate_request, validate_candidate, static_repair_hint
from unified_graph_lowering import candidate_source
from unified_typed_guidance import FEATURE_RULES
class TypedGuidanceTests(unittest.TestCase):
 def request(self,version="explicit-v6",exercise=None):
  b=Builder([(2,)])
  return prepare(b.finish(b.node("square",["input0"])),PROFILE_SHA256,
   construction_profile="hecate-unified-public-v1",construction=exercise,generation_guidance=version)
 def test_only_opt_in_metadata_changes(self):
  old=self.request("explicit-v5");new=self.request()
  for k in old:
   if k not in ("request_id","generation_guidance"):self.assertEqual(old[k],new[k])
  g=copy.deepcopy(new["generation_guidance"]);g.pop("typed_request_bindings");g["version"]="explicit-v5"
  self.assertEqual(g,old["generation_guidance"]);validate_request(new)
 def test_sorted_json_roundtrip(self):
  r=self.request();validate_request(json.loads(json.dumps(r,sort_keys=True)))
 def test_rehashed_forged_binding_is_rejected(self):
  for key,value in [("output_ciphertexts",2),("rotation_steps",[3]),("constant_bindings",[])]:
   r=self.request();r["generation_guidance"]["typed_request_bindings"][key]=value
   r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
   with self.assertRaises(ValueError):validate_request(r)
 def test_shapes_and_names_are_public_registry_only(self):
  r=self.request();g=r["generation_guidance"]["typed_request_bindings"]
  self.assertEqual([v["name"] for v in g["constant_bindings"]],sorted(r["public_constants"]))
  for v in g["constant_bindings"]:
   value=r["public_constants"][v["name"]]
   self.assertEqual(v["shape"],[len(value)] if type(value) is list else [])
  self.assertEqual(g["rotation_steps"],[1,2])
  self.assertIn("not a bound runtime dictionary",g["type_rules"]["names"])
 def test_scoped_feedback_never_echoes_untrusted_diagnostics(self):
  r=self.request();s=candidate_source(r)[0]
  for message,expected in [("Undefined or unbound local construction value: float SECRET","Bare dtype=float"),
    ("Ragged public array SECRET","rectangular"),("Golden ciphertext outputs mismatch SECRET","exact number"),
    ("Rotation requires SECRET","provisioned")]:
   hint=static_repair_hint(s,r,message)
   self.assertIn(expected,hint);self.assertNotIn("SECRET",hint)
 def test_unknown_diagnostic_no_new_feedback(self):
  r=self.request();self.assertEqual(static_repair_hint(candidate_source(r)[0],r,"/private/SECRET"),"")
 def test_old_feedback_is_unchanged(self):
  r=self.request("explicit-v5")
  self.assertEqual(static_repair_hint(candidate_source(r)[0],r,"Undefined or unbound local construction value: float"),"")
 def test_scoped_directed_requirements_are_actual_registered_features(self):
  import unified_public_exercises as public
  for feature in FEATURE_RULES:
   with self.subTest(feature=feature):
    found=[v for v in [public.spec(name) for name in public.SPECS] if v["required_features"]==[feature]]
    self.assertTrue(found,feature)
    r=self.request(exercise=found[0]["id"])
    self.assertEqual(r["generation_guidance"]["typed_request_bindings"]["directed_acceptance"],
      [{"feature":feature,"acceptance":FEATURE_RULES[feature]}])
 def test_no_permission_widening(self):
  r=self.request()
  for body in ('return [float(x)]','return np.array([x], dtype="float64")','return [+x]'):
   with self.subTest(body=body), self.assertRaises(ValueError):
    validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source='@hc.func("c,c")\ndef golden(x, zero_ct):\n    '+body+'\n'),r)
 def test_cli_and_sandbox(self):
  from run_candidate import parse_args
  from candidate_sandbox import MODULES
  self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v6"]).unified_guidance,"explicit-v6")
  self.assertIn("unified_typed_guidance.py",MODULES)
if __name__=="__main__":unittest.main()
