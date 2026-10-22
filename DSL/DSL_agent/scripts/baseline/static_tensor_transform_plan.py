"""Checked static tensor repetition and permutation; graph semantics only."""
import math
from benchmark_graph import require, dimensions, axis
from model_resources import resolve

POLICY = "poseidon-static-tensor-transforms-v1"
OPERATORS = frozenset(("repeat", "tile", "repeat_interleave", "flip", "roll"))
ATTRIBUTES = {
    "repeat": frozenset(("repeats",)), "tile": frozenset(("repeats",)),
    "repeat_interleave": frozenset(("repeats", "axis", "output_size")),
    "flip": frozenset(("axes",)), "roll": frozenset(("shifts", "axes")),
}

def integer(value, message, *, positive=False):
    require(type(value) is int and (not positive or value > 0), message)
    return value

def sequence(value, message):
    require(type(value) is list and all(type(x) is int for x in value), message)
    return list(value)

def plan(op, shape, attrs, *, resource_profile):
    limits = resolve(resource_profile)
    require(op in OPERATORS and type(attrs) is dict and set(attrs) == ATTRIBUTES[op],
            "static_transform_attributes")
    dimensions(list(shape), resource_profile=resource_profile)
    rank = len(shape)
    steps = []; current = "input"; current_shape = list(shape)
    def append(operation, refs, parameters, result_shape):
        nonlocal current, current_shape
        require(len(steps) < limits.expanded_nodes, "static_transform_expanded_capacity")
        dimensions(list(result_shape), resource_profile=resource_profile)
        name = "step_" + str(len(steps))
        steps.append(dict(name=name, op=operation, inputs=list(refs), attrs=parameters,
                          shape=list(result_shape)))
        current, current_shape = name, list(result_shape)
        return name
    def reshape(target):
        if list(target) != current_shape:
            append("reshape", [current], dict(shape=list(target)), target)
    def join(refs, ax, shapes):
        require(len(refs) == len(shapes) and refs, "static_transform_join")
        while len(refs) > 1:
            next_refs = []; next_shapes = []
            for start in range(0, len(refs), 8):
                part = refs[start:start+8]; ss = shapes[start:start+8]
                out = list(ss[0]); out[ax] = sum(s[ax] for s in ss)
                value = append("concat", part, dict(axis=ax), out) if len(part)>1 else part[0]
                next_refs.append(value); next_shapes.append(out)
            refs, shapes = next_refs, next_shapes
        return refs[0], shapes[0]
    def repeated(value, source_shape, count, ax):
        # Binary joins bound fan-in and avoid a repeat-count-sized allocation.
        blocks = []; shapes = []; block = value; block_shape = list(source_shape)
        while count:
            if count & 1: blocks.append(block); shapes.append(list(block_shape))
            count >>= 1
            if count:
                out = list(block_shape); out[ax] *= 2
                block = append("concat", [block, block], dict(axis=ax), out); block_shape = out
        return join(blocks, ax, shapes)
    if op in ("repeat", "tile"):
        factors = sequence(attrs["repeats"], "static_transform_repeats")
        require(len(factors) <= limits.rank and all(n > 0 for n in factors),
                "static_transform_positive_repeats")
        require(op == "tile" or len(factors) >= rank, "static_transform_repeat_rank")
        width = max(rank, len(factors)); source = [1]*(width-rank) + list(shape)
        factors = [1]*(width-len(factors)) + factors
        target = [n*f for n,f in zip(source, factors)]
        dimensions(target, resource_profile=resource_profile)  # before any expansion
        reshape(source)
        for ax, count in enumerate(factors):
            current, current_shape = repeated(current, current_shape, count, ax)
    elif op == "repeat_interleave":
        count = integer(attrs["repeats"], "static_transform_repeat_count", positive=True)
        ax = attrs["axis"]
        source = list(shape)
        if ax is None: source = [math.prod(shape)]; ax = 0
        else: ax = axis(ax, rank)
        target = list(source); target[ax] *= count
        dimensions(target, resource_profile=resource_profile)
        size = attrs["output_size"]
        require(size is None or type(size) is int and size == target[ax],
                "static_transform_output_size")
        reshape(source)
        original, source = current, list(current_shape)
        if count > 1:
            require(source[ax]*(1+count.bit_length()) <= limits.expanded_nodes,
                    "static_transform_expanded_capacity")
            values = []; shapes = []
            for i in range(source[ax]):
                ss = list(source); ss[ax] = 1
                part = append("slice", [original], dict(axis=ax,start=i,stop=i+1,step=1), ss)
                part, ss = repeated(part, ss, count, ax)
                values.append(part); shapes.append(ss)
            current, current_shape = join(values, ax, shapes)
    elif op == "flip":
        axes = sequence(attrs["axes"], "static_transform_axes")
        normalized = [axis(ax, rank) for ax in axes]
        require(len(set(normalized)) == len(normalized), "static_transform_duplicate_axis")
        for ax in normalized:
            original, source = current, list(current_shape); n = source[ax]
            if n == 1: continue
            require(n + (n+6)//7 <= limits.expanded_nodes-len(steps),
                    "static_transform_expanded_capacity")
            values = []; shapes = []
            for i in range(n-1, -1, -1):
                ss = list(source); ss[ax] = 1
                values.append(append("slice", [original], dict(axis=ax,start=i,stop=i+1,step=1), ss))
                shapes.append(ss)
            current, current_shape = join(values, ax, shapes)
    else:
        shifts = sequence(attrs["shifts"], "static_transform_shifts"); axes = attrs["axes"]
        require(shifts, "static_transform_empty_shifts")
        if axes is None:
            require(len(shifts) == 1, "static_transform_flat_roll")
            reshape([math.prod(shape)]); normalized = [0]
        else:
            raw = sequence(axes, "static_transform_axes")
            require(len(raw) == len(shifts), "static_transform_roll_arity")
            normalized = [axis(ax, rank) for ax in raw]
        for ax, shift in zip(normalized, shifts):
            source, original = list(current_shape), current; n = source[ax]; amount = shift % n
            if amount:
                left = list(source); left[ax] = amount
                right = list(source); right[ax] = n-amount
                a = append("slice", [original], dict(axis=ax,start=n-amount,stop=n,step=1), left)
                c = append("slice", [original], dict(axis=ax,start=0,stop=n-amount,step=1), right)
                current = append("concat", [a,c], dict(axis=ax), source)
        if axes is None: reshape(shape)
    if not steps:
        append("reshape", [current], dict(shape=list(current_shape)), current_shape)
    return dict(policy=POLICY, operator=op, input_shape=list(shape), output_shape=list(current_shape),
                steps=steps, result=current, output_storage="independent_contiguous_copy",
                expanded_nodes=len(steps), runtime_validated=False)
