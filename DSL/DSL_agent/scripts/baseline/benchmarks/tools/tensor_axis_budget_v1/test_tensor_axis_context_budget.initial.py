"""Whole-envelope v12 budget checks with public regression models."""
import copy,json,unittest
from pathlib import Path
from benchmark_graph import canonical,digest
from compiler_configuration import configuration
from component_contract import reconstruct_request,request_options
from unified_capacity_constants import request_version
from unified_graph_contract import prepare,validate_request
from public_weight_context import project,CONTEXT_BYTES
from unified_tensor_axis_guidance import fit_public_context,specification
FIXTURES=Path(__file__).with_name("cases")/"tensor-axis-context-v1"
def current(label,version="explicit-v12"):
 old=reconstruct_request(json.loads((FIXTURES/(label+"-v11.json")).read_text()))
 opts=request_options(old);name=opts.pop("compiler_configuration")
 opts.update(generation_guidance=version,capacity_packing_version=request_version(old),
             capacity_abi=old["layout"]["execution_abi"])
 value=prepare(old["model"],old["compiler_profile_sha256"],configuration(name),
  constant_policy=old["constant_origins"].get("policy"),**opts)
 return old,value
class ContextBudgetTests(unittest.TestCase):
 def test_cross_fits_by_trimming_examples_and_preserves_mathematical_binding(self):
  old,new=current("cross");validate_request(new)
  self.assertLessEqual(len(canonical(project(new))),CONTEXT_BYTES)
  self.assertEqual(reconstruct_request(new),new)
  left,right=copy.deepcopy(old),copy.deepcopy(new)
  for r in (left,right):r.pop("generation_guidance");r.pop("request_id")
  self.assertEqual(canonical(left),canonical(right))
  full=specification(new);fitted=new["generation_guidance"]["poseidon_expression_rules"]
  self.assertLess(len(canonical(fitted)),len(canonical(full)))
  self.assertEqual(fitted["rules"],full["rules"])
  for field in ("reduction_count","transform_count","linear_count","broadcast_count","input_chunk_count"):
   self.assertEqual(fitted["tensor_axis_annotations"][field],full["tensor_axis_annotations"][field])

 def test_small_envelope_keeps_all_examples(self):
  old,new=current("self")
  self.assertEqual(new["generation_guidance"]["poseidon_expression_rules"],specification(new))
  self.assertLessEqual(len(canonical(project(new))),CONTEXT_BYTES)

 def test_public_projection_and_guidance_deterministic_and_not_mutated(self):
  _,new=current("cross");before=copy.deepcopy(new)
  guide=copy.deepcopy(new["generation_guidance"])
  self.assertEqual(fit_public_context(new,guide),guide)
  self.assertEqual(new,before)
  self.assertEqual(guide,new["generation_guidance"])
  self.assertEqual(current("cross")[1],new)

 def test_tampering_a_trimmed_annotation_is_rejected_even_if_rehashed(self):
  _,r=current("cross")
  r["generation_guidance"]["poseidon_expression_rules"]["tensor_axis_annotations"]["reduction_count"]+=1
  from public_weight_context import request_digest
  r["request_id"]=request_digest(r)
  with self.assertRaises(ValueError):validate_request(r)

 def test_required_graph_or_rules_are_not_removed_to_accept_oversize(self):
  _,r=current("self");before=copy.deepcopy(r)
  r["rules"]+="PUBLIC_FIXED_RULE"*10000
  immutable=copy.deepcopy(r)
  with self.assertRaisesRegex(ValueError,"required public context"):
   fit_public_context(r,r["generation_guidance"])
  self.assertEqual(r,immutable)

 def test_historical_v11_fixture_bytes_reconstruct(self):
  for label in ("self","cross"):
   path=FIXTURES/(label+"-v11.json");before=path.read_bytes()
   stored=json.loads(before)
   self.assertEqual(current(label,"explicit-v11")[1],stored)
   self.assertEqual(path.read_bytes(),before)

 def test_prior_small_v12_qualification_requests_unchanged(self):
  # Public request snapshots, not encrypted outputs or private probes.
  fixture=FIXTURES/"qualified-small-v12.json"
  requests=json.loads(fixture.read_text())
  for r in requests:self.assertEqual(reconstruct_request(r),r)

if __name__=="__main__":unittest.main()
