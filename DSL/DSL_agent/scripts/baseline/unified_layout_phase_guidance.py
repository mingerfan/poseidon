"""Versioned mathematical frames for existing public Linear tile bindings."""
import copy
VERSION="explicit-v18"
PREVIOUS_VERSION="explicit-v17"
ROLES=("linear_tile_expand","linear_tile_compress")
PHASE_RULE=(
 "A tile fixed/moving vector selects slots in the input frame of its phase. "
 "For that phase, destination j equals fixed[j]*input[j] plus "
 "moving[(j+shift) mod P]*input[(j+shift) mod P]. The moving selection happens "
 "before its rotation. Multiplying the unchanged moving vector after rotating "
 "the input selects a different set of slots. Both expansion and compression "
 "follow this same coordinate rule. Apply phases in their registered order; "
 "later masks refer to the preceding phase's result, not the initial compact "
 "tensor. Rotation notation denotes compositions of the admitted literal "
 "rotations; it grants no new key or native integer parameter."
)
REPRESENTATION_RULE=(
 "Compact rows use the declared logical width; tile rows use the registered "
 "power-of-two stride. Row weights and row-start masks act in the expanded "
 "tile frame. Compression returns the declared compact output rows before "
 "the compact bias and any later reduction are applied. A vector reused in "
 "another operation keeps its actual slot support; its name does not move "
 "that support or replicate a scalar. Track each intermediate frame explicitly."
)
def require(ok,code):
 if not ok:raise ValueError(code)
def phase_frames(request):
 packed=request.get("constant_origins",{}).get("capacity_tensor_packing",{})
 vectors=request["public_constants"];period=request["layout"]["input_slot_period"]
 groups={}
 for b in packed.get("bindings",[]):
  if b.get("role")not in ROLES:continue
  require(b.get("side")in("fixed","moving"),"Tile phase side")
  key=(b["node_id"],b["tile_batches"],b["role"],b["phase"])
  row=groups.setdefault(key,{})
  require(b["side"]not in row,"Duplicate tile phase side");row[b["side"]]=b["key"]
 frames=[]
 for (node,count,role,phase),keys in sorted(groups.items()):
  require(set(keys)=={"fixed","moving"}and type(count)is int and count>1 and type(phase)is int,"Tile phase pair")
  stride=packed["linear_tiles"][node]["stride"];bits=(count-1).bit_length()
  require(type(stride)is int and 0<=phase<bits and stride*count<=period,"Tile phase geometry")
  fixed,moving=(vectors[keys[s]]for s in("fixed","moving"))
  require(type(fixed)is list and type(moving)is list and len(fixed)==len(moving)==period,"Tile phase vectors")
  require(all(type(a)in(int,float)and type(b)in(int,float)and a in(0.,1.)and b in(0.,1.)and not(a and b)
   for a,b in zip(fixed,moving)),"Tile masks")
  cells=int(sum(fixed)+sum(moving));require(cells%count==0,"Tile row cardinality")
  width=cells//count;require(0<width<=stride,"Tile row width")
  bit=bits-1-phase if role=="linear_tile_expand"else phase
  signed=(stride-width)*(1<<bit)*(-1 if role=="linear_tile_expand"else 1)
  frames.append(dict(node_id=node,tile_batches=count,role=role,phase=phase,
   fixed=keys["fixed"],moving=keys["moving"],row_width=width,stride=stride,
   shift=signed%period,signed_shift=signed,mask_frame="phase_input"))
 return frames
def specification(request,inherited):
 require(type(inherited)is dict and inherited.get("version")==PREVIOUS_VERSION,"Tile context requires explicit-v17")
 result=copy.deepcopy(inherited);result["version"]=VERSION
 frames=phase_frames(request)
 if frames:
  result["bound_tile_phase_frames"]=dict(version="linear-tile-phase-frames-v1",phases=frames,
   coordinates=PHASE_RULE,representations=REPRESENTATION_RULE)
 return result
def coordinate_hint(request):
 frames=phase_frames(request)
 return PHASE_RULE+"\n"+REPRESENTATION_RULE if frames else""
