"""Draft logical view normalization. No compiler, runtime, provider or tracing."""
import copy,hashlib,itertools,json,math
from dataclasses import dataclass
from benchmark_graph import validate
FORMAT="poseidon-static-view-normalization-draft-v1"
MAX_ELEMENTS=4096
MAX_INDEX_CELLS=262144
def canonical(v):
 return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def strides(shape):
 result=[1]*len(shape)
 for i in range(len(shape)-2,-1,-1):result[i]=result[i+1]*shape[i+1]
 return result
def permutation(shape,target,indices):
 """Infer axes from strides, then verify every coordinate; never guess by size."""
 if len(shape)!=len(target)or sorted(shape)!=sorted(target)or len(indices)!=math.prod(shape):
  return None
 n=math.prod(shape)
 if n>MAX_ELEMENTS or sorted(indices)!=list(range(n))or indices[0]!=0:return None
 source_strides=strides(shape);target_strides=strides(target);axes=[];unused=set(range(len(shape)))
 for dim,stride in zip(target,target_strides):
  if dim==1:
   axes.append(None);continue
  delta=indices[stride]-indices[0]
  choices=[i for i in unused if shape[i]==dim and source_strides[i]==delta]
  if len(choices)!=1:return None
  found=choices[0];axes.append(found);unused.remove(found)
 singleton=iter(sorted(unused))
 for j in range(len(axes)):
  if axes[j]is None:
   i=next(singleton)
   if shape[i]!=1:return None
   axes[j]=i
 for flat in range(n):
  expected=sum(((flat//target_strides[j])%target[j])*source_strides[i]for j,i in enumerate(axes))
  if indices[flat]!=expected:return None
 return axes
@dataclass(frozen=True)
class Trace:
 root:str
 indices:tuple
 nodes:frozenset
def normalize(model,*,max_index_cells=MAX_INDEX_CELLS):
 if type(max_index_cells)is not int or not 1<=max_index_cells<=MAX_INDEX_CELLS:
  raise ValueError("Closed normalization work budget")
 before=validate(model);shapes=before["shapes"];cache={};spent=0;exhausted=False;rewrites=[];nodes=[]
 def admit(root,indices,contributors):
  nonlocal spent,exhausted
  cost=len(indices)+len(contributors)
  if len(indices)>MAX_ELEMENTS or spent+cost>max_index_cells:
   exhausted=True;return None
  spent+=cost;return Trace(root,tuple(indices),frozenset(contributors))
 def trace(value):
  if value not in cache:
   shape=shapes[value];count=math.prod(shape)
   cache[value]=admit(value,range(count),())if count<=MAX_ELEMENTS else None
  return cache[value]
 for node in model["nodes"]:
  n=copy.deepcopy(node);op=n["op"];refs=n["inputs"];a=n["attrs"];out=n["outputs"]
  result=None
  if len(out)==1 and op=="reshape":
   t=trace(refs[0])
   if t is not None:
    result=admit(t.root,t.indices,t.nodes|{n["id"]})
    axes=permutation(shapes[t.root],shapes[out[0]],t.indices)
    if axes is not None and axes!=list(range(len(axes)))and len(t.nodes)>=2:
     n=dict(id=n["id"],op="permute",inputs=[t.root],attrs=dict(dims=axes),outputs=out)
     rewrites.append(dict(final_node=n["id"],root_value=t.root,output_value=out[0],
       source_shape=shapes[t.root],target_shape=shapes[out[0]],axes=axes,
       index_map_sha256=digest(list(t.indices)),contributing_nodes=sorted(t.nodes),
       rule="flat_slice_concat_to_full_axis_permutation"))
  elif len(out)==1 and op=="slice" and len(shapes[refs[0]])==1 and a["axis"]in(0,-1):
   t=trace(refs[0])
   if t is not None:
    selected=t.indices[slice(a["start"],a["stop"],a["step"])]
    result=admit(t.root,selected,t.nodes|{n["id"]})
  elif len(out)==1 and op=="concat" and a["axis"]in(0,-1)and all(len(shapes[r])==1 for r in refs):
   ts=[trace(r)for r in refs]
   if all(t is not None for t in ts)and len({t.root for t in ts})==1:
    indices=tuple(itertools.chain.from_iterable(t.indices for t in ts))
    contributors=frozenset().union(*(t.nodes for t in ts))|{n["id"]}
    result=admit(ts[0].root,indices,contributors)
  nodes.append(n)
  for value in out:cache[value]=result
  # An unsupported mathematical producer becomes a fresh root on later use.
  # Failed work allocation remains unavailable; it must not be bypassed.
  if op not in("reshape","slice","concat")or len(out)!=1:
   for value in out:cache.pop(value,None)
 result_model=copy.deepcopy(model)
 if rewrites:
  active={o["value"]for o in model["outputs"]};kept=[]
  for n in reversed(nodes):
   if any(v in active for v in n["outputs"]):
    kept.append(n);active.update(n["inputs"])
  result_model["nodes"]=list(reversed(kept))
 after=validate(result_model)
 if after["output_shapes"]!=before["output_shapes"]:
  raise ValueError("Output shape changed")
 if result_model["inputs"]!=model["inputs"]or result_model["constants"]!=model["constants"]or result_model["outputs"]!=model["outputs"]:
  raise ValueError("Immutable declarations changed")
 record=dict(format=FORMAT,original_sha256=digest(model),normalized_sha256=digest(result_model),
  original_nodes=len(model["nodes"]),normalized_nodes=len(result_model["nodes"]),
  rules=rewrites,exact_coordinate_verification=True,proof_max_elements=MAX_ELEMENTS,
  proof_index_cells=spent,proof_index_cell_limit=max_index_cells,work_budget_exhausted=exhausted,
  production_active=False,backend_calls=0,provider_calls=0)
 record["binding"]=digest(record)
 return result_model,record
