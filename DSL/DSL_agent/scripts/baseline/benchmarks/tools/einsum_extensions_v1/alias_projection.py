"""Pure alias projection for future source-policy integration, not activation."""
import hashlib,itertools,math
from pathlib import Path
from benchmark_graph import digest
from planner import check
FORMAT="poseidon-einsum-alias-projection-draft-v1"
def projection(record):
 check(record)
 shapes=record["source_shapes"];core=record["core_plan"]
 view=len(shapes)==1 and(core is None or set(core["operand_labels"][0])==set(core["output_labels"]))
 indices=None
 if view:
  if core is None:indices=[0]
  else:
   shape=shapes[0];diag=record["diagonals"]
   values=diag[0]["indices"]if diag else list(range(math.prod(shape)))
   labels=core["operand_labels"][0];dims=core["source_shapes"][0];output=core["output_labels"]
   stride=1;strides=[]
   for n in reversed(dims):strides.insert(0,stride);stride*=n
   sizes=dict(zip(labels,dims));positions={label:i for i,label in enumerate(output)}
   indices=[values[sum(c[positions[label]]*s for label,s in zip(labels,strides))]
    for c in itertools.product(*(range(sizes[label])for label in output))]
 result=dict(format=FORMAT,plan_binding=record["binding"],role="view"if view else"computed",
  source_operand=0 if view else None,logical_shape=record["result_shape"],source_flat_indices=indices,
  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),production_source_policy=False)
 result["binding"]=digest(result);return result
