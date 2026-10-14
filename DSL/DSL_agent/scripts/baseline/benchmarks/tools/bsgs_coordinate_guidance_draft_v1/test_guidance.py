"""Draft selection, old-context preservation and fixed public feedback."""
import copy,json,pathlib,unittest
import unified_route_coordinate_guidance as g
class CoordinateGuidance(unittest.TestCase):
 def request(self,role="tensor_route",giant=6):
  return dict(layout=dict(input_slot_period=8),constant_origins=dict(capacity_tensor_packing=dict(bindings=[
   dict(role=role,baby_step=1,giant_step=giant,step=(1+giant)%8)])))
 def inherited(self):return dict(version="explicit-v16",typed_request_bindings={"plain":["p0"]},limits={"ast":4096})
 def test_registered_nonzero_giant_selects_fixed_math(self):
  v=g.specification(self.request(),self.inherited())
  self.assertIn("vector[(j+G) mod P]",v["bound_bsgs_coordinates"]["coordinates"])
  self.assertEqual(v["bound_bsgs_coordinates"]["roles"],["tensor_route"])
 def test_zero_giant_and_unrelated_roles_keep_existing_context(self):
  for request in(self.request(giant=0),self.request(role="unrelated")):
   value=g.specification(request,self.inherited());self.assertNotIn("bound_bsgs_coordinates",value)
   self.assertEqual(g.coordinate_hint(request),"")
 def test_existing_native_public_context_and_limits_unchanged(self):
  for kind in("native","public"):
   old=self.inherited();old["contract"]=kind;before=copy.deepcopy(old)
   v=g.specification(self.request(),old)
   self.assertEqual(old,before)
   self.assertEqual({k:x for k,x in v.items()if k not in("version","bound_bsgs_coordinates")},
       {k:x for k,x in old.items()if k!="version"})
 def test_all_pre_rotated_gather_roles_are_registered(self):
  for role in g.ROLES:self.assertEqual(g.selected_roles(self.request(role)),[role])
 def test_invalid_step_bool_or_out_of_range_rejected(self):
  for field,value in(("step",2),("baby_step",True),("giant_step",8)):
   r=self.request();r["constant_origins"]["capacity_tensor_packing"]["bindings"][0][field]=value
   with self.assertRaises(ValueError):g.specification(r,self.inherited())
 def test_untrusted_diagnostic_is_not_an_argument_or_content(self):
  text=g.coordinate_hint(self.request())
  for secret in("SECRET","reference=[","/private/","case_id"):self.assertNotIn(secret,text)
 def test_existing_frozen_request_is_not_rebound_or_modified(self):
  root=pathlib.Path(__file__).resolve().parents[1]/"contract_scope_live_v1/prepared/rank_axis_8/request.json"
  request=json.loads(root.read_bytes());before=copy.deepcopy(request)
  v=g.specification(request,request["generation_guidance"])
  self.assertEqual(request,before);self.assertEqual(v["version"],"explicit-v17")
  self.assertEqual(v["bound_bsgs_coordinates"]["roles"],["tensor_route"])
 def test_prior_version_inheritance_is_required(self):
  with self.assertRaises(ValueError):g.specification(self.request(),dict(version="explicit-v15"))
if __name__=="__main__":unittest.main(verbosity=2)
