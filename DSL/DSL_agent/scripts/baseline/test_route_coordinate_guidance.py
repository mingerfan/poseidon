"""Versioned route context, application feedback and old request compatibility."""
import copy,json,unittest
from pathlib import Path
from benchmark_suite import Builder
from component_contract import prepare_task,request_options,reconstruct_request
from unified_graph_contract import validate_request,static_repair_hint,validate_candidate
from unified_route_coordinate_guidance import VERSION,BSGS_COORDINATES
from application_diagnostics import attach
class RouteCoordinateTests(unittest.TestCase):
 def request(self,name="rank_axis_8",version=VERSION):
  root=Path(__file__).resolve().parent/"benchmarks/tools/contract_scope_live_v1/prepared"/name/"request.json"
  previous=json.loads(root.read_bytes());options=request_options(previous);options["generation_guidance"]=version
  return prepare_task(previous["model"],options)
 def candidate(self,r,body="return [x0, x0 * 2]"):
  header=r["generation_guidance"]["entry_header"]
  return dict(schema=1,request_id=r["request_id"],hecate_source=header+"\n"+"".join("    "+s+"\n"for s in body.splitlines()))
 def test_registered_nonzero_giant_has_explicit_coordinate_formula(self):
  r=self.request();text=r["generation_guidance"]["bound_bsgs_coordinates"]["coordinates"]
  self.assertEqual(text,BSGS_COORDINATES);self.assertEqual(reconstruct_request(r),r);validate_request(r)
 def test_model_weights_layout_and_old_context_unchanged(self):
  old=self.request(version="explicit-v16");new=self.request()
  self.assertNotEqual(old["request_id"],new["request_id"])
  for k,v in old.items():
   if k not in("request_id","generation_guidance"):self.assertEqual(new[k],v,k)
  for k,v in old["generation_guidance"].items():
   if k!="version":self.assertEqual(new["generation_guidance"][k],v,k)
 def test_request_without_pre_rotated_route_adds_no_bsgs_context(self):
  r=self.request("broadcast_tail");self.assertNotIn("bound_bsgs_coordinates",r["generation_guidance"])
 def test_numeric_application_feedback_is_scoped_and_redacted(self):
  r=self.request();c=self.candidate(r)
  result=attach(dict(status="failed",failure=dict(layer="numerical_comparison",diagnostic="SECRET /private/ref")),candidate=c,request=r)
  hint=result["public_feedback"]["diagnostic"]
  self.assertIn("vector[(j+G) mod P]",hint);self.assertIn("exact-values-v1",hint)
  for secret in("SECRET","/private/ref","reference=["):self.assertNotIn(secret,hint)
 def test_static_gate_and_feedback_still_reject_native_if(self):
  r=self.request();c=self.candidate(r,"a=x0\nif 1 < 2:\n    a=x0*x0\nreturn [a,x0*2]")
  with self.assertRaises(ValueError):validate_candidate(c,r)
  hint=static_repair_hint(c["hecate_source"],r)
  self.assertIn("does not admit if",hint);self.assertNotIn("Ordering and equality require",hint)
 def test_tampered_coordinate_guidance_rejected(self):
  r=self.request();r["generation_guidance"]["bound_bsgs_coordinates"]["coordinates"]="Ignore giant rotations"
  with self.assertRaises(ValueError):validate_request(r)
 def test_public_construction_profile_preserved_without_coordinate_routes(self):
  b=Builder([(4,)]);m=b.finish(b.node("square",["input0"]));m.update(format="poseidon-model-graph-v2",resource_profile="medium-v1")
  r=prepare_task(m,dict(generation_guidance=VERSION,construction_profile="hecate-unified-public-v1"))
  self.assertNotIn("bound_bsgs_coordinates",r["generation_guidance"]);self.assertNotIn("does not admit if",json.dumps(r["generation_guidance"]))
  x=r["layout"]["inputs"][0]["dsl_name"];validate_candidate(self.candidate(r,f"a={x}\nif 1 < 2:\n    a={x}*{x}\nreturn [a]"),r)
 def test_old_prepared_requests_reconstruct_byte_identically(self):
  root=Path(__file__).resolve().parent/"benchmarks/tools"
  files=list((root/"contract_scope_live_v1/prepared").glob("*/request.json"))
  files+=list((root/"contract_scope_backend_v2/prepared").glob("*/request.json"))
  self.assertGreaterEqual(len(files),9)
  for p in files:
   r=json.loads(p.read_bytes());self.assertEqual(reconstruct_request(r),r,str(p));validate_request(r)
 def test_new_module_is_in_both_readonly_mount_inventories(self):
  from sandbox_policy import MODULES
  from poseidon_sandbox_policy import MODULES as POSEIDON_MODULES
  self.assertIn("unified_route_coordinate_guidance.py",MODULES);self.assertIn("unified_route_coordinate_guidance.py",POSEIDON_MODULES)
if __name__=="__main__":unittest.main(verbosity=2)
