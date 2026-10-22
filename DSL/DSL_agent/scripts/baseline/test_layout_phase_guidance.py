"""Phase frames against three direct tensor-coordinate contexts."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from unified_layout_phase_guidance import VERSION,PHASE_RULE,phase_frames
from unified_capacity_tiling import row_stages
from component_contract import prepare_task,request_options,reconstruct_request
from unified_graph_contract import validate_request
from application_diagnostics import attach
class PhaseFramesTests(unittest.TestCase):
 def geometry(self,width,stride,count,period,role):
  stages=row_stages(width,stride,count,period,compress=role=="linear_tile_compress")
  constants={};bindings=[]
  for phase,x in enumerate(stages):
   for side in("fixed","moving"):
    key=f"v_{phase}_{side}";constants[key]=x[side]
    bindings.append(dict(node_id="n",tile_batches=count,role=role,phase=phase,side=side,key=key))
  return dict(public_constants=constants,layout=dict(input_slot_period=period),constant_origins=dict(capacity_tensor_packing=
    dict(bindings=bindings,linear_tiles=dict(n=dict(stride=stride))))),stages
 def test_three_contexts_match_direct_compact_and_expanded_coordinates(self):
  for width,stride,count,period in((5,8,2,16),(3,8,3,32),(17,32,4,128)):
   compact=np.arange(1,count*width+1,dtype=np.float64)
   expected=np.zeros(period)
   for b in range(count):expected[b*stride:b*stride+width]=compact[b*width:(b+1)*width]
   r,_=self.geometry(width,stride,count,period,"linear_tile_expand");x=np.zeros(period);x[:len(compact)]=compact
   for frame in phase_frames(r):
    fixed=np.asarray(r["public_constants"][frame["fixed"]]);moving=np.asarray(r["public_constants"][frame["moving"]])
    x=x*fixed+np.roll(x*moving,-frame["shift"])
   np.testing.assert_array_equal(x,expected)
   r,_=self.geometry(width,stride,count,period,"linear_tile_compress")
   for frame in phase_frames(r):
    x=x*np.asarray(r["public_constants"][frame["fixed"]])+np.roll(x*np.asarray(r["public_constants"][frame["moving"]]),-frame["shift"])
   expected=np.zeros(period);expected[:len(compact)]=compact
   np.testing.assert_array_equal(x,expected)
 def test_rotating_before_selecting_changes_the_mapping(self):
  r,_=self.geometry(5,8,2,16,"linear_tile_expand");frame=phase_frames(r)[0]
  x=np.zeros(16);x[:10]=np.arange(1,11)
  fixed=np.asarray(r["public_constants"][frame["fixed"]]);moving=np.asarray(r["public_constants"][frame["moving"]])
  good=x*fixed+np.roll(x*moving,-frame["shift"])
  bad=x*fixed+np.roll(x,-frame["shift"])*moving
  self.assertFalse(np.array_equal(good,bad))
 def request(self):
  p=Path(__file__).resolve().parent/"benchmarks/tools/provenance_transport_live_v2/prepared/wide_tail_linear/request.json"
  old=json.loads(p.read_bytes());opts=request_options(old);opts["generation_guidance"]=VERSION
  return old,prepare_task(old["model"],opts)
 def test_real_unchanged_model_and_layout_receive_new_phase_identity(self):
  old,new=self.request();validate_request(new);self.assertEqual(reconstruct_request(new),new)
  self.assertNotEqual(old["request_id"],new["request_id"])
  for key,value in old.items():
   if key not in("generation_guidance","request_id"):self.assertEqual(new[key],value,key)
  phases=new["generation_guidance"]["bound_tile_phase_frames"]["phases"]
  self.assertEqual([(x["role"],x["signed_shift"],x["mask_frame"])for x in phases],
   [("linear_tile_compress",2045,"phase_input"),("linear_tile_expand",-1023,"phase_input")])
 def test_feedback_is_public_fixed_math_and_preserves_alias_rules(self):
  _,r=self.request();c=dict(schema=1,request_id=r["request_id"],hecate_source=r["generation_guidance"]["entry_header"]+"\n    return x0, x1\n")
  v=attach(dict(status="failed",failure=dict(layer="numerical_comparison",diagnostic="PRIVATE /tmp/ref")),candidate=c,request=r)
  text=v["public_feedback"]["diagnostic"];self.assertIn("moving selection happens before",text);self.assertIn("exact-values-v1",text)
  self.assertNotIn("PRIVATE",text);self.assertNotIn("/tmp/ref",text)
 def test_request_options_preserves_selected_capacity_packing(self):
  old,new=self.request()
  self.assertEqual(request_options(old)["capacity_packing_version"],"capacity-public-tensors-v10")
  self.assertEqual(new["constant_origins"],old["constant_origins"])
 def test_phase_pair_and_mask_integrity_rejected(self):
  r,_=self.geometry(5,8,2,16,"linear_tile_expand")
  r["public_constants"]["v_0_fixed"][0]=0.25
  with self.assertRaises(ValueError):phase_frames(r)
 def test_ordinary_request_adds_no_tile_context(self):
  p=Path(__file__).resolve().parent/"benchmarks/tools/provenance_transport_live_v2/prepared/rank_permute_32/request.json"
  old=json.loads(p.read_bytes());o=request_options(old);o["generation_guidance"]=VERSION
  r=prepare_task(old["model"],o);self.assertNotIn("bound_tile_phase_frames",r["generation_guidance"])
if __name__=="__main__":unittest.main(verbosity=2)
