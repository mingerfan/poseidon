"""Draft fixed coordinate guidance. No candidate rewriting or backend changes."""
import copy
VERSION="explicit-v17"
PREVIOUS_VERSION="explicit-v16"
ROLES=("tensor_route","pool_gather","pool_stream_gather","suffix_gather")
BSGS_COORDINATES=(
 "Packed rotations follow R_k(c)[j] = c[(j+k) mod P]. For a registered route "
 "with baby_step=B and giant_step=G, step=(B+G) mod P. The bound vector is in "
 "the baby coordinate frame: vector[j] equals the destination coefficient at "
 "(j-G) mod P. Multiply this vector by R_B(source), then apply R_G to that "
 "whole product. Equivalently the destination j receives vector[(j+G) mod P] "
 "times source[(j+B+G) mod P]. A direct multiplication of the same pre-rotated "
 "vector by R_(B+G)(source) changes the selected positions when G is nonzero. "
 "Routes with the same source chunk, output chunk and G may sum their masked "
 "baby contributions before that final rotation. Keep different source and "
 "output chunks separate. G=0 reduces to the direct route formula. This "
 "coordinate rule applies to the registered vectors, not to arbitrary "
 "intermediate layouts or logical tensor dimensions."
)
def selected_roles(request):
 descriptor=request.get("constant_origins",{}).get("capacity_tensor_packing",{})
 period=request["layout"]["input_slot_period"]
 if type(period)is not int or period<1:raise ValueError("Invalid slot period")
 selected=set()
 for b in descriptor.get("bindings",[]):
  if b.get("role")not in ROLES or not("baby_step"in b and "giant_step"in b):continue
  baby,giant,step=(b[x]for x in("baby_step","giant_step","step"))
  if not all(type(x)is int and 0<=x<period for x in(baby,giant,step))or step!=(baby+giant)%period:
   raise ValueError("Invalid registered BSGS coordinates")
  if giant:selected.add(b["role"])
 return sorted(selected)
def specification(request,inherited):
 if type(inherited)is not dict or inherited.get("version")!=PREVIOUS_VERSION:
  raise ValueError("BSGS guidance requires explicit-v16")
 result=copy.deepcopy(inherited);result["version"]=VERSION
 roles=selected_roles(request)
 if roles:
  result["bound_bsgs_coordinates"]=dict(version="bound-bsgs-coordinate-frame-v1",roles=roles,coordinates=BSGS_COORDINATES)
 return result
def coordinate_hint(request):
 return BSGS_COORDINATES if selected_roles(request)else""
