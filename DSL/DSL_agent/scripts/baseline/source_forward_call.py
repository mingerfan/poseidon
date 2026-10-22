"""Data-only reconstruction of a declared original forward call.

Tensor objects are supplied by the trusted worker. This helper does not load
weights, inspect or execute source, coerce dtype, or change mathematical work.
"""
from pytorch_input_binding import validate_call, materialize
from benchmark_graph import require

def bind_call(manifest, tensors):
    require(type(manifest) is dict and type(manifest.get("inputs")) is list and
            1 <= len(manifest["inputs"]) <= 4, "source_call_input_inventory")
    require(type(tensors) in (list, tuple) and len(tensors) == len(manifest["inputs"]),
            "source_call_tensor_inventory")
    names = [entry.get("name") if type(entry) is dict else None for entry in manifest["inputs"]]
    require(all(type(name) is str for name in names) and len(set(names)) == len(names),
            "source_call_input_names")
    if "call" not in manifest:
        return list(tensors), {}
    call = validate_call(manifest["call"], manifest["inputs"])
    by_name = dict(zip(names, tensors))
    return ([materialize(tree, by_name) for tree in call["args"]],
            {name: materialize(tree, by_name) for name, tree in call["kwargs"].items()})
