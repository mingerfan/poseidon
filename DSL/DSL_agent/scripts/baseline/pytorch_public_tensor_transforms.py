"""Closed public-state repeat/permutation math with allocation checks.

Only declared public arrays enter this layer. It calls fixed NumPy operations,
never a model callable, a DSL lowering, or a backend helper.
"""
import copy
import math
import numpy as np
from benchmark_graph import require, digest
from model_resources import resolve, identity as resource_identity
from pytorch_public_value_ops import checked_array, signature, materialize
from pytorch_static_tensor_transforms import arguments

POLICY = "poseidon-public-value-expressions-v4"
FORMAT = "poseidon-public-tensor-transform-plan-v1"
METHODS = frozenset(("repeat", "tile", "repeat_interleave", "flip", "roll"))
FUNCTIONS = frozenset("torch." + name for name in METHODS if name != "repeat")


def registered(kind, target):
    return (kind == "call_method" and target in METHODS or
            kind == "call_function" and target in FUNCTIONS)


def describe(kind, target, args, kwargs, shape, *, resource_profile):
    limits = resolve(resource_profile)
    require(registered(kind, target), "public_transform_operation")
    op = target if kind == "call_method" else target.removeprefix("torch.")
    _, attrs = arguments(op, args, kwargs, kind)
    rank = len(shape)
    require(rank <= limits.rank and all(type(d) is int and 0 < d <= limits.axis_size for d in shape),
            "public_transform_input_shape")

    def sequence(value, label, *, empty=False):
        require(type(value) is list and (empty or value) and len(value) <= limits.rank and
                all(type(n) is int for n in value), label)
        return list(value)

    def axis(value):
        require(type(value) is int and -rank <= value < rank, "public_transform_axis")
        return value % rank

    if op in ("repeat", "tile"):
        repeats = sequence(attrs["repeats"], "public_transform_repeats", empty=op == "tile")
        require(all(n > 0 for n in repeats), "public_transform_positive_repeats")
        if op == "repeat":
            require(len(repeats) >= max(rank, 1), "public_transform_repeat_rank")
        width = max(rank, len(repeats))
        repeats = [1] * (width - len(repeats)) + repeats
        padded = [1] * (width - rank) + list(shape)
        result = [a * n for a, n in zip(padded, repeats)]
        normalized = dict(repeats=repeats)
    elif op == "repeat_interleave":
        count = attrs["repeats"]
        require(type(count) is int and count > 0, "public_transform_repeat_count")
        ax = None if attrs["axis"] is None else axis(attrs["axis"])
        result = [math.prod(shape)] if ax is None else list(shape)
        result[0 if ax is None else ax] *= count
        size = attrs["output_size"]
        require(size is None or type(size) is int and size == result[0 if ax is None else ax],
                "public_transform_output_size")
        normalized = dict(repeats=count, axis=ax, output_size=size)
    elif op == "flip":
        raw_axes = sequence(attrs["axes"], "public_transform_axes", empty=True)
        axes = [axis(a) for a in raw_axes]
        require(len(axes) == len(set(axes)), "public_transform_duplicate_axis")
        result = list(shape)
        normalized = dict(axes=axes)
    else:
        shifts = sequence(attrs["shifts"], "public_transform_shifts")
        if attrs["axes"] is None:
            require(len(shifts) == 1, "public_transform_flat_roll")
            shifts = [shifts[0] % math.prod(shape)]
            axes = None
        else:
            raw_axes = sequence(attrs["axes"], "public_transform_axes")
            require(len(raw_axes) == len(shifts), "public_transform_roll_arity")
            axes = [axis(a) for a in raw_axes]
            shifts = [s % shape[a] for s, a in zip(shifts, axes)]
        result = list(shape)
        normalized = dict(shifts=shifts, axes=axes)
    # Public state uses the constant budget, not the encrypted tensor budget.
    require(len(result) <= limits.rank and all(0 < d <= limits.axis_size for d in result) and
            math.prod(result) <= limits.constant_elements, "public_transform_output_capacity")
    return op, normalized, result


def plan(value, kind, target, args, kwargs, *, resource_profile="general-tensors-v1"):
    value = checked_array(value, resource_profile=resource_profile)
    op, attrs, shape = describe(kind, target, args, kwargs, list(value.shape),
                               resource_profile=resource_profile)
    if op in ("repeat", "tile"):
        logical = np.tile(value, attrs["repeats"])
    elif op == "repeat_interleave":
        logical = np.repeat(value, attrs["repeats"], axis=attrs["axis"])
    elif op == "flip":
        logical = np.flip(value, axis=tuple(attrs["axes"]))
    elif attrs["axes"] is None:
        logical = np.roll(value.reshape(-1), attrs["shifts"][0]).reshape(value.shape)
    else:
        logical = np.roll(value, tuple(attrs["shifts"]), axis=tuple(attrs["axes"]))
    require(list(logical.shape) == shape and logical.dtype == value.dtype, "public_transform_math_shape")
    logical = checked_array(np.asarray(logical), resource_profile=resource_profile)
    physical = materialize(logical, resource_profile=resource_profile)
    record = dict(format=FORMAT, policy=POLICY, source_kind=kind, source_target=target,
                  operation=op, attributes=copy.deepcopy(attrs),
                  input=signature(value, resource_profile=resource_profile),
                  logical_output=signature(logical, resource_profile=resource_profile),
                  materialized_output=signature(physical, resource_profile=resource_profile),
                  resource_profile=resource_identity(resource_profile),
                  storage="independent_copy", allocation_bound_elements=math.prod(shape),
                  exact_float64_materialization=logical.dtype.str == "<f4")
    record["binding"] = digest(record)
    return logical, physical, record
