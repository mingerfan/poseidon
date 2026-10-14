"""Draft public Tensor.gather normalization into existing high-level operations.

The draft never imports or executes a model, emits DSL, or modifies a backend.
Caller must bind indices as public int64 data before invoking this planner.
"""
import itertools
import math
from benchmark_graph import canonical, digest, require
from model_resources import resolve

VERSION = "poseidon-public-tensor-gather-v1"

def index_tensor(value):
    """Validate rectangular Python integer data without coercing floats or bool."""
    if type(value) is int:
        return (), [value]
    require(type(value) is list and len(value) > 0, "Gather index tensor must be nonempty int64 data")
    children = [index_tensor(v) for v in value]
    require(all(shape == children[0][0] for shape, _ in children), "Gather index tensor must be rectangular")
    return (len(value),) + children[0][0], [x for _, values in children for x in values]

def plan(source_shape, index, axis, *, resource_profile="legacy-v1", index_dtype="int64"):
    limits = resolve(resource_profile)
    require(type(source_shape) is list and 1 <= len(source_shape) <= limits.rank
            and all(type(v) is int and 0 < v <= limits.axis_size for v in source_shape)
            and math.prod(source_shape) <= limits.tensor_elements, "Gather input shape budget")
    require(index_dtype == "int64", "Gather requires public int64 indices")
    require(type(axis) is int and -len(source_shape) <= axis < len(source_shape), "Gather static axis")
    normalized_axis = axis % len(source_shape)
    require(len(canonical(index)) <= limits.json_bytes, "Gather index byte budget")
    target_shape, values = index_tensor(index)
    require(len(target_shape) == len(source_shape)
            and all(0 < v <= limits.axis_size for v in target_shape)
            and len(values) <= limits.tensor_elements, "Gather index rank or element budget")
    require(all(i == normalized_axis or target_shape[i] <= source_shape[i]
                for i in range(len(source_shape))), "Gather non-index axis exceeds input")
    require(all(0 <= v < source_shape[normalized_axis] for v in values), "Gather index out of bounds")
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
        backend_or_layout_claimed=False, normalization_only=True)
    require(len(canonical(result)) <= limits.json_bytes, "Gather plan byte budget")
    result["binding"] = digest(result)
    return result

def check_plan(declared, source_shape, index, axis, **options):
    require(type(declared) is dict and declared == plan(source_shape,index,axis,**options),
            "Gather plan binding")
    return declared
