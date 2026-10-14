"""Public tensor-axis annotations; independent of ciphertext lowering and answers."""
import ast
import copy
import json
import math
from benchmark_graph import axis,require,validate

VERSION="explicit-v12"
MAX_EXAMPLES=12
MAX_FIBERS=4
MAX_MEMBERS=8
MAX_BYTES=32768
REDUCTION=(
 "Reduce exactly the declared logical axes. Every non-reduced coordinate identifies a separate "
 "group, including batch, head, row and column coordinates. A rotate-and-add sums the source "
 "slots of that rotation, not automatically one logical group. For each output, retain only "
 "its own reduction members; exclude other groups, padded tails and repeated periodic copies. "
 "The sampled fibers below are logical C-order indices, not intermediate ciphertext slot plans. "
 "Choose and track your own intermediate slot mapping. A value valid at one lane must be moved "
 "or replicated before it is used at another lane. Mean divides by the full logical member count.")
TRANSFORM=(
 "Reshape and flatten preserve logical C-order element indices. Permute and transpose change "
 "which logical coordinate occupies each flat index. Reshaping an object array rearranges whole "
 "Expr cells; it does not move encrypted slots inside those cells. Preserve the coordinate "
 "mapping through each transform and every later reduction or broadcast.")
LINEAR=(
 "Linear contracts the input's last axis with the weight's input-feature axis. All leading input "
 "coordinates remain separate. Compute every output-feature row with its declared coefficients "
 "and bias; neither head/batch groups nor uniform-looking matrix rows imply scalar multiplication.")
BROADCAST=(
 "Broadcasting right-aligns logical axes and expands only singleton dimensions. For a product "
 "followed by a reduction, match both operand coordinates before contraction. Keep all non-reduced "
 "batch/head/row/column coordinates separate. Matching object-array shapes or selecting the same "
 "physical lane in different expressions does not establish matching logical coordinates.")
INPUT=(
 "Input annotations describe only the frozen input binding. Valid logical elements occupy their "
 "declared chunk prefix; padding is not another logical feature. A chunk can contain multiple "
 "logical rows or groups. Preserve their identities after projections and intermediate transforms. "
 "Intermediate ciphertext representations are your choice; named outputs still use the frozen selectors.")

def strides(shape):
 return [math.prod(shape[i+1:]) for i in range(len(shape))]
def coordinates(index,shape):
 require(type(index) is int and 0<=index<math.prod(shape),"Logical coordinate range")
 result=[]
 for width,stride in zip(shape,strides(shape)):
  result.append((index//stride)%width)
 return result
def samples(shape,limit=MAX_FIBERS):
 total=math.prod(shape);result=[]
 for value in (0,total-1,*strides(shape),total//2):
  if 0<=value<total and value not in result:result.append(value)
  if len(result)==limit:break
 return result
def reduction_example(node,shapes):
 shape=list(shapes[node["inputs"][0]]);rank=len(shape)
 axes=sorted(axis(i,rank) for i in node["attrs"]["axes"])
 kept=[i for i in range(rank) if i not in axes]
 out=list(shapes[node["outputs"][0]]);reduced=[shape[i] for i in axes]
 members=math.prod(reduced);out_strides=strides(shape);fibers=[]
 selected=list(range(min(members,MAX_MEMBERS)))
 if members>MAX_MEMBERS:selected[-1]=members-1
 for flat in samples(out):
  out_coord=coordinates(flat,out);base=[0]*rank
  if node["attrs"]["keepdims"]:
   for i in kept:base[i]=out_coord[i]
  else:
   for j,i in enumerate(kept):base[i]=out_coord[j]
  indices=[]
  for member in selected:
   coord=base[:]
   for j,i in enumerate(axes):coord[i]=coordinates(member,reduced)[j]
   indices.append(sum(x*s for x,s in zip(coord,out_strides)))
  fibers.append(dict(output_index=flat,input_indices=indices,
   listed_members=len(indices),total_members=members,complete=len(indices)==members))
 return dict(node=node["id"],operator=node["op"],input=node["inputs"][0],
  input_shape=shape,output=node["outputs"][0],output_shape=out,
  reduced_axes=axes,preserved_axes=kept,keepdims=node["attrs"]["keepdims"],
  input_c_strides=out_strides,member_count=members,logical_fibers=fibers)
def transform_example(node,shapes):
 source=list(shapes[node["inputs"][0]]);out=list(shapes[node["outputs"][0]])
 rank=len(source);op=node["op"]
 if op in ("flatten","reshape"):order=None
 elif op=="permute":order=[axis(i,rank) for i in node["attrs"]["dims"]]
 else:
  order=list(range(rank));i=axis(node["attrs"]["dim0"],rank);j=axis(node["attrs"]["dim1"],rank)
  order[i],order[j]=order[j],order[i]
 mapping=[]
 for flat in samples(out):
  if order is None:input_index=flat
  else:
   old=[0]*rank
   for i,value in enumerate(coordinates(flat,out)):old[order[i]]=value
   input_index=sum(x*s for x,s in zip(old,strides(source)))
  mapping.append(dict(output_index=flat,input_index=input_index))
 return dict(node=node["id"],operator=op,input=node["inputs"][0],input_shape=source,
  output=node["outputs"][0],output_shape=out,sampled_index_map=mapping)

def specification(request):
 from unified_binding_guidance import specification as previous
 checked=validate(request["model"]);shapes=checked["shapes"]
 context=copy.deepcopy(previous(request));groups=context["rules"]
 reductions=[];transforms=[];linears=[];broadcasts=[]
 for node in request["model"]["nodes"]:
  op=node["op"]
  if op in ("sum","mean"):reductions.append(reduction_example(node,shapes))
  if op in ("flatten","reshape","permute","transpose"):transforms.append(transform_example(node,shapes))
  if op=="linear":
   shape=list(shapes[node["inputs"][0]]);weight=list(shapes[node["inputs"][1]])
   linears.append(dict(node=node["id"],input_shape=shape,weight_shape=weight,
    output_shape=list(shapes[node["outputs"][0]]),input_feature_axis=len(shape)-1,
    weight_feature_axis=1,preserved_input_axes=list(range(len(shape)-1))))
  if op=="multiply":
   out=list(shapes[node["outputs"][0]])
   if any(list(shapes[n])!=out for n in node["inputs"]):
    broadcasts.append(dict(node=node["id"],operand_shapes=[list(shapes[n]) for n in node["inputs"]],
     output_shape=out))
 if reductions:groups["segmented_logical_reduction"]=REDUCTION
 if transforms:groups["logical_axis_transforms"]=TRANSFORM
 if linears:groups["linear_contraction_groups"]=LINEAR
 if broadcasts:groups["logical_broadcast_coordinates"]=BROADCAST
 inputs=[]
 for row in request["layout"]["inputs"][:MAX_EXAMPLES]:
  shape=list(row["shape"]);offset=row.get("offset",0);size=row.get("elements",math.prod(shape))
  require(type(offset) is int and type(size) is int and size>0 and offset+size<=math.prod(shape),
          "Public input chunk range")
  points=sorted(set((0,size-1)))
  inputs.append(dict(name=row["name"],dsl_name=row["dsl_name"],shape=shape,
   logical_offset=offset,valid_elements=size,padded_slots=request["layout"]["input_slot_period"]-size,
   sampled_input_coordinates=[dict(slot=i,logical_index=offset+i,
    coordinate=coordinates(offset+i,shape)) for i in points]))
 groups["ordered_input_coordinates"]=INPUT
 context.update(version=VERSION,selection=sorted(groups),rules={k:groups[k] for k in sorted(groups)},
  tensor_axis_annotations=dict(reductions=reductions[:MAX_EXAMPLES],reduction_count=len(reductions),
   transforms=transforms[:MAX_EXAMPLES],transform_count=len(transforms),
   linears=linears[:MAX_EXAMPLES],linear_count=len(linears),
   broadcasts=broadcasts[:MAX_EXAMPLES],broadcast_count=len(broadcasts),
   input_chunks=inputs,input_chunk_count=len(request["layout"]["inputs"]),
   scope="Sampled public logical index relations. No encrypted intermediate plan, test values or DSL implementation."))
 # Example lists are optional context; retain counts and every applicable rule.
 lists=[context["tensor_axis_annotations"][k] for k in ("transforms","linears","broadcasts","input_chunks","reductions")]
 lists.extend([context["shape_examples"][k] for k in ("reductions","broadcasts")])
 lists.extend([context["logical_constant_scalar_bindings"]["bindings"],
               context["constant_role_examples"]["bindings"]])
 while len(json.dumps(context,sort_keys=True).encode())>MAX_BYTES:
  candidates=[row for row in lists if row]
  require(candidates,"Tensor-axis guidance byte bound")
  max(candidates,key=lambda row:len(json.dumps(row,sort_keys=True))).pop()
 return context

def fit_public_context(request,guidance):
 """Trim optional examples only; keep the frozen mathematical/DSL envelope."""
 from benchmark_graph import canonical
 from public_weight_context import CONTEXT_BYTES,project,selected
 result=copy.deepcopy(guidance)
 base={k:v for k,v in request.items() if k not in ("request_id","generation_guidance")}
 envelope=project(base) if selected(base) else copy.deepcopy(base)
 envelope.update(generation_guidance=result,request_id="0"*64)
 context=result["poseidon_expression_rules"]
 lists=[context["tensor_axis_annotations"][k] for k in
        ("transforms","linears","broadcasts","input_chunks","reductions")]
 lists.extend(context["shape_examples"][k] for k in ("reductions","broadcasts"))
 lists.extend([context["logical_constant_scalar_bindings"]["bindings"],
               context["constant_role_examples"]["bindings"]])
 while len(canonical(envelope))>CONTEXT_BYTES:
  candidates=[row for row in lists if row]
  require(candidates,"Tensor-axis required public context exceeds generation budget")
  max(candidates,key=lambda row:len(canonical(row))).pop()
 return result


def repair_hint(request,source,diagnostic="",stage="static_check"):
 if request.get("generation_guidance",{}).get("version")!=VERSION:return ""
 if type(source) is not str or len(source.encode())>65536:return ""
 try:tree=ast.parse(source)
 except (SyntaxError,ValueError,RecursionError):return ""
 if sum(1 for _ in ast.walk(tree))>4096:return ""
 context=specification(request)
 names=("segmented_logical_reduction","logical_axis_transforms","linear_contraction_groups",
        "logical_broadcast_coordinates","ordered_input_coordinates")
 hints=[context["rules"][n] for n in names if n in context["rules"]] if stage in (
  "static_check","compiler","artifact_gate","numerical_comparison") else []
 from unified_binding_guidance import repair_hint as previous
 old=dict(request,generation_guidance=dict(request["generation_guidance"],version="explicit-v11"))
 prior=previous(old,source,diagnostic,stage)
 if prior:hints.append(prior)
 return "\n".join(dict.fromkeys(hints))[:5000]
