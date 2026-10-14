"""Resolve known Torch FX calls inside a trusted capture worker.

This draft consumes existing FX nodes and validated operand metadata. It never
imports or runs user code and is not connected to production input preparation.
"""
import hashlib
from pathlib import Path

from source_ops_draft import source_call, fail


def from_fx_call(node, operands, *, approximation=None, output_name="result",
                 resource_profile="general-tensors-v1"):
    import operator
    import torch
    from torch.nn import functional as F
    from torch.fx.immutable_collections import immutable_list

    if type(node) is not torch.fx.Node or type(operands) is not dict:
        fail("source_call_unregistered", "plan", "call")
    functions = (
        (operator.truediv, "operator.truediv"),
        (torch.div, "torch.div"),
        (torch.divide, "torch.divide"),
        (torch.true_divide, "torch.true_divide"),
        (F.max_pool1d, "F.max_pool1d"),
        (F.max_pool2d, "F.max_pool2d"),
    )
    methods = {"div": "Tensor.div", "divide": "Tensor.divide",
               "true_divide": "Tensor.true_divide"}
    if node.op == "call_function":
        # boolean_dispatch wrapper names are insufficient. Match the exact
        # pinned functional objects, rather than arbitrary __module__/__name__.
        call = next((name for target, name in functions if node.target is target), None)
    elif node.op == "call_method" and type(node.target) is str:
        call = methods.get(node.target)
    else:
        call = None
    if call is None:
        fail("source_call_unregistered", "plan", "call")

    from model_resources import resolve as resource_limits
    try:
        limits = resource_limits(resource_profile)
    except (ValueError, TypeError):
        fail("source_shape_rejected", "plan", "resource_profile")
    copied = 0
    budget = min(limits.constant_elements, limits.total_constant_elements) + 4*limits.rank + 32

    def static_copy(value, depth=0):
        nonlocal copied
        copied += 1
        if depth > limits.rank + 6 or copied > budget:
            fail("source_call_arguments_rejected", "plan", "call")
        if type(value) is dict:
            if any(type(k) is not str for k in value):
                fail("source_call_arguments_rejected", "plan", "call")
            return {k: static_copy(v, depth+1) for k, v in value.items()}
        if type(value) in (list, tuple):
            return [static_copy(v, depth+1) for v in value]
        if value is None or type(value) in (str, bool, int, float):
            return value
        # Avoid arbitrary __deepcopy__, iteration and conversion hooks.
        fail("source_call_arguments_rejected", "plan", "call")

    def resolve(value):
        if type(value) is torch.fx.Node:
            if value not in operands:
                fail("source_fx_operand_unbound", "plan", "input")
            return static_copy(operands[value])
        if type(value) in (list, tuple, immutable_list):
            # FX owns this exact immutable container class. Subclasses and
            # arbitrary sequence conversion hooks remain rejected.
            return [resolve(v) for v in value]
        if value is None or type(value) in (str, bool, int, float):
            return value
        fail("source_call_arguments_rejected", "plan", "call")

    args = [resolve(v) for v in node.args]
    kwargs = {key: resolve(value) for key, value in node.kwargs.items()}
    if call.startswith(("torch.", "Tensor.")) or call == "operator.truediv":
        if len(args) >= 2 and type(args[1]) is not dict:
            args[1] = {"kind": "public", "value": args[1]}
        if "other" in kwargs and type(kwargs["other"]) is not dict:
            kwargs["other"] = {"kind": "public", "value": kwargs["other"]}
    return source_call(call, args, kwargs, approximation=approximation,
                       output_name=output_name, resource_profile=resource_profile)


def implementation():
    return {"fx_source_ops_draft.py":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
