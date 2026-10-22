"""Exact graph-value aliases, versioned independently of the transport ABI.

Logical limits include every named output. Physical limits count each unique
value's complete chunks. Equal-valued expressions with different refs stay separate.
"""
import copy
import math
from benchmark_graph import NAME, canonical, dimensions, digest, require, validate
from model_resources import resolve

POLICY = "exact-values-v1"
FORMAT = "poseidon-exact-output-binding-v1"

def policy(value):
    if type(value) is not dict or "output_binding" not in value:
        return None
    binding = value["output_binding"]
    require(type(binding) is dict and binding.get("format") == FORMAT and
            binding.get("policy") == POLICY, "Unknown output binding policy")
    return POLICY

def declarations(inputs, outputs, period, profile, execution_abi, model_sha256):
    require(type(model_sha256) is str and len(model_sha256) == 64 and
            all(c in "0123456789abcdef" for c in model_sha256), "Output binding model identity")
    limits = resolve(profile)
    require(type(outputs) is list and 1 <= len(outputs) <= limits.outputs,
            "Capacity logical I/O count")
    require(all(type(o) is dict and set(o) == {"name", "value", "shape"} and
                all(type(o[k]) is str and NAME.fullmatch(o[k]) for k in ("name", "value"))
                for o in outputs), "Exact output declarations")
    require(len({o["name"] for o in outputs}) == len(outputs), "Duplicate capacity tensor name")
    sizes = [math.prod(dimensions(o["shape"], resource_profile=profile)) for o in outputs]
    require(sum(sizes) <= limits.output_elements, "Capacity logical element budget")
    owners, unique = {}, []
    for o in outputs:
        owner = owners.get(o["value"])
        if owner is None:
            owners[o["value"]] = o
            unique.append(dict(name=o["name"], shape=o["shape"]))
        else:
            require(owner["shape"] == o["shape"], "Exact output alias shape mismatch")
    from unified_capacity_layout import declarations as physical_declarations
    plan = physical_declarations(inputs, unique, period, profile, execution_abi)
    physical = {o["name"]: o for o in plan["outputs"]}
    logical, selectors = [], []
    for o in outputs:
        owner = owners[o["value"]]["name"]
        row = copy.deepcopy(physical[owner])
        row["name"] = o["name"]
        if row["name"] != owner:
            row["alias_of"] = owner
        logical.append(row)
        for part in row["chunks"]:
            selectors.extend([[part["ciphertext"], i] for i in range(part["elements"])])
    plan.update(outputs=logical, output_selectors=selectors, output_shape=[len(selectors)],
                output_binding=dict(format=FORMAT, policy=POLICY,
                    model_sha256=model_sha256, logical_outputs=copy.deepcopy(outputs)))
    return plan

def layout(model, period=None, *, execution_abi=None):
    checked = validate(model)
    require(all(checked["kinds"][o["value"]] == "cipher" for o in model["outputs"]),
            "Exact output requires ciphertext values")
    from benchmark_graph import graph_resource_profile
    return declarations(model["inputs"],
        [dict(name=o["name"], value=o["value"], shape=checked["shapes"][o["value"]])
         for o in model["outputs"]], period, graph_resource_profile(model), execution_abi, digest(model))

def validate_layout(value):
    require(policy(value) == POLICY, "Exact output binding required")
    binding = value["output_binding"]
    require(set(binding) == {"format", "policy", "model_sha256", "logical_outputs"},
            "Output binding fields")
    profile = value.get("resource_profile", {})
    require(type(profile) is dict and type(profile.get("name")) is str,
            "Output binding resource profile")
    expected = declarations(value.get("logical_inputs"), binding["logical_outputs"],
        value.get("input_slot_period"), profile["name"], value.get("execution_abi"),
        binding["model_sha256"])
    require(canonical(expected) == canonical(value), "Changed exact output layout")
    return expected["input_slot_period"]

def physical_output_items(model, value):
    """First owners in physical order; unselected requests keep the old path."""
    if policy(value) is None:
        return list(zip(model["outputs"], value["outputs"]))
    validate_layout(value)
    require(value["output_binding"]["model_sha256"] == digest(model),
            "Output binding model mismatch")
    checked = validate(model)
    expected = [dict(name=o["name"], value=o["value"], shape=checked["shapes"][o["value"]])
                for o in model["outputs"]]
    require(expected == value["output_binding"]["logical_outputs"], "Output value binding mismatch")
    return [(o, b) for o, b in zip(model["outputs"], value["outputs"]) if "alias_of" not in b]

def rules(value):
    validate_layout(value)
    return """\nExact output binding exact-values-v1: names with the same model value share the
first owner's physical chunks. Return each physical ciphertext exactly once in
ciphertext-index order, not one copy per logical name. Output selectors reconstruct
all named logical outputs in their original order, including aliases and tails.
Equal shapes or numerically equal expressions with different value references are
not aliases. Logical output element limits still count every named output.
Every unique physical output retains the distinct compiled Value ID requirement.
"""
