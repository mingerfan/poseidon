"""Draft public Tensor.gather normalization into existing high-level operations.

The draft never imports or executes a model, emits DSL, or modifies a backend.
Caller must bind indices as public int64 data before invoking this planner.
"""
import itertools
import math
from benchmark_graph import canonical, digest, require
from model_resources import resolve

VERSION = "poseidon-public-tensor-gather-v1"

def index_tensor(value, *, max_rank=32, max_elements=65536):
    """Bounded rectangular DFS; never recurse through user-provided nesting."""
    stack=[(value,0)];shape=[];leaves=[];leaf_depth=None;visited=0
    while stack:
        current,depth=stack.pop();visited+=1
        require(visited<=max_elements*(max_rank+1)+1, "Gather index work budget")
        if type(current) is int:
            require(leaf_depth is None or leaf_depth==depth, "Gather index tensor must be rectangular")
            leaf_depth=depth;leaves.append(current)
            require(len(leaves)<=max_elements, "Gather index element budget")
            continue
        require(type(current) is list and 0<len(current)<=max_elements,
                "Gather index tensor must be nonempty int64 data")
        require(depth<max_rank, "Gather index rank budget")
        if depth==len(shape):shape.append(len(current))
        require(shape[depth]==len(current) and (leaf_depth is None or leaf_depth>depth),
                "Gather index tensor must be rectangular")
        require(math.prod(shape)<=max_elements, "Gather index element budget")
        stack.extend((child,depth+1)for child in reversed(current))
    require(leaf_depth==len(shape), "Gather index tensor must be rectangular")
    return tuple(shape),leaves

def plan(source_shape, index, axis, *, resource_profile="legacy-v1", index_dtype="int64"):
    limits = resolve(resource_profile)
    require(type(source_shape) is list and 1 <= len(source_shape) <= limits.rank
            and all(type(v) is int and 0 < v <= limits.axis_size for v in source_shape)
            and math.prod(source_shape) <= limits.tensor_elements, "Gather input shape budget")
    require(index_dtype == "int64", "Gather requires public int64 indices")
    require(type(axis) is int and -len(source_shape) <= axis < len(source_shape), "Gather static axis")
    normalized_axis = axis % len(source_shape)
    target_shape, values = index_tensor(index,max_rank=limits.rank,max_elements=limits.tensor_elements)
    require(len(canonical(index)) <= limits.json_bytes, "Gather index byte budget")
    require(len(target_shape) == len(source_shape)
            and all(0 < v <= limits.axis_size for v in target_shape)
            and len(values) <= limits.tensor_elements, "Gather index rank or element budget")
    require(all(i == normalized_axis or target_shape[i] <= source_shape[i]
                for i in range(len(source_shape))), "Gather non-index axis exceeds input")
    require(all(0 <= v < source_shape[normalized_axis] for v in values), "Gather index out of bounds")
    require(len(values)+2<=limits.expanded_nodes, "Gather minimum decomposition node budget")
    strides = [math.prod(source_shape[i+1:]) for i in range(len(source_shape))]
    offsets = []
    for position, value in zip(itertools.product(*(range(v) for v in target_shape)), values):
        coordinate = list(position)
        coordinate[normalized_axis] = value
        offsets.append(sum(c * stride for c, stride in zip(coordinate, strides)))
    result = dict(version=VERSION, source_shape=list(source_shape), index_shape=list(target_shape),
        axis=normalized_axis, index_dtype=index_dtype, index_sha256=digest(index),
        result_shape=list(target_shape), steps=[
            dict(op="reshape", attrs=dict(shape=[math.prod(source_shape)])),
            dict(op="public_gather", attrs=dict(axis=0, indices=offsets)),
            dict(op="reshape", attrs=dict(shape=list(target_shape))),
        ], resource_profile=resource_profile,
        decomposition_estimate=dict(minimum_core_nodes=len(values)+2,certainty="lower_bound",
                                    exact_core_validation_required=True),
        backend_or_layout_claimed=False, normalization_only=True)
    require(len(canonical(result)) <= limits.json_bytes, "Gather plan byte budget")
    result["binding"] = digest(result)
    return result

def check_plan(declared, source_shape, index, axis, **options):
    require(type(declared) is dict and declared == plan(source_shape,index,axis,**options),
            "Gather plan binding")
    return declared
