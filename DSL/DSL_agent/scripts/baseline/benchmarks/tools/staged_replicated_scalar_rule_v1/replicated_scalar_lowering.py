"""Staged generic replicated-cell rule baseline, never a Provider rewrite.

The original request, graph, public registry, compiler profile and reference stay
fixed. Each encrypted logical cell is extracted once and replicated throughout P.
Tensor views retain cell references; only final immutable physical output chunks
are packed. Missing scalar Plain bindings can use finite exact binary64 literals
only with independently reconstructed original public-data provenance. This is a
trusted manual baseline strategy, not a public accessor or Provider capability.
"""
import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import unified_graph_lowering as graph_lowering
import unified_capacity_lowering as capacity_lowering
from benchmark_graph import require, validate
from unified_graph_contract import validate_request, validate_candidate
from unified_graph_lowering import Emitter, Scalar, emit_spatial, power_polynomial

VERSION = "poseidon-replicated-scalar-rule-v1"
FORMAT = "poseidon-replicated-scalar-rule-record-v1"
SOURCE_KIND = "deterministic_rule_baseline"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact(value):
    require(type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1024,
            "Finite exact64 public literal boundary")
    return float(value).hex()


def literal_sources(model, checked):
    """Rebuild authority from original public tensors/attributes, never pv values."""
    result = {}
    def add(value, origin):
        key = exact(value)
        result.setdefault(key, []).append(origin)
    def cells(value, coord=()):
        if type(value) is list:
            for i, child in enumerate(value):
                yield from cells(child, (*coord, i))
        else:
            yield coord, value
    for name, value in sorted(model["constants"].items()):
        constant_sha = digest(value)
        for coord, number in cells(value):
            add(number, dict(kind="model_constant", logical_name=name, coordinate=list(coord),
                             constant_sha256=constant_sha, scalar_sha256=digest(number)))
    for node in model["nodes"]:
        op, attrs, refs = node["op"], node["attrs"], node["inputs"]
        base = dict(node_id=node["id"], operator=op, attributes_sha256=digest(attrs),
                    input_shape=checked["shapes"][refs[0]])
        if op == "mean":
            shape = checked["shapes"][refs[0]]
            count = math.prod(shape[i % len(shape)] for i in attrs["axes"])
            add(1.0/count, dict(base, kind="mean_divisor", count=count))
        elif op in ("avg_pool1d", "avg_pool2d"):
            rank = 1 if op.endswith("1d") else 2
            spatial = checked["shapes"][refs[0]][-rank:]
            output = checked["shapes"][node["outputs"][0]][-rank:]
            counts = {math.prod(attrs["kernel"])} if attrs["count_include_pad"] else set()
            if not attrs["count_include_pad"]:
                for pos in itertools.product(*(range(n) for n in output)):
                    counts.add(math.prod(max(0, min(n, p*st-pad+k)-max(0, p*st-pad))
                        for n, p, st, pad, k in zip(spatial, pos, attrs["stride"], attrs["padding"], attrs["kernel"])))
            for count in sorted(counts):
                require(count > 0, "Empty public pool window")
                add(1.0/count, dict(base, kind="pool_divisor", count=count))
        elif op == "batch_norm":
            mean, var, gamma, beta = [model["constants"][ref] for ref in refs[1:]]
            identity = {ref:digest(model["constants"][ref]) for ref in refs[1:]}
            for channel, (m, v, g, b) in enumerate(zip(mean, var, gamma, beta)):
                gain = float(g)/math.sqrt(float(v)+attrs["eps"])
                add(gain, dict(base, kind="batch_norm_gain", channel=channel, constants_sha256=identity))
                add(float(b)-float(m)*gain, dict(base, kind="batch_norm_offset", channel=channel, constants_sha256=identity))
    return result


def budget_floor(request, checked):
    """A primitive-work lower bound, never depth/capacity or a zero guess.

    Every logical encrypted input is extracted, even if unused. Only pure views
    of input cells prove final output expressions have the nonzero flag used by
    Scalar.pack; all other outputs receive the conservative zero lower bound.
    """
    layout = request["layout"]
    period = layout["input_slot_period"]
    bits = period.bit_length()-1
    input_ops = 0
    for binding in layout["inputs"]:
        count = binding.get("elements", math.prod(binding["shape"]))
        input_ops += count*(1+2*bits)+sum(i.bit_count() for i in range(count))
    known = {item["name"]:True for item in request["model"]["inputs"]}
    views = {"reshape", "flatten", "transpose", "permute", "rotate", "slice", "split", "concat", "stack"}
    for node in request["model"]["nodes"]:
        nonzero = node["op"] in views and all(known.get(ref, False) for ref in node["inputs"])
        known.update({name:nonzero for name in node["outputs"]})
    output_ops = cells = 0
    for output, binding in zip(request["model"]["outputs"], layout["outputs"]):
        if not known.get(output["value"], False):
            continue
        size = math.prod(checked["shapes"][output["value"]])
        chunks = binding.get("chunks", [dict(elements=size)])
        for chunk in chunks:
            count = chunk["elements"]
            cells += count
            output_ops += count+max(0, count-1)  # masks and additions, rotations conservatively omitted.
    return dict(input_extraction=input_ops, final_pack=output_ops,
                proven_nonzero_output_cells=cells, total=input_ops+output_ops,
                scope="primitive-emission lower bound only; no capacity claim")


class ReplicatedEmitter(Emitter):
    """Explicit instance-local extension; no shared emitter or global mutation."""
    def __init__(self, request, allowed):
        super().__init__(request)
        self.scalar_names = {exact(value):name for name, value in request["public_constants"].items()
                             if type(value) in (int, float)}
        self.allowed = allowed
        self.fallbacks = {}

    def literal(self, value):
        key = exact(float(value))
        if key in self.scalar_names:
            return self.scalar_names[key]
        require(key in self.allowed, "Missing literal original public provenance")
        self.fallbacks.setdefault(key, dict(value_hex=key, source_literal=repr(float(value)),
            magnitude=abs(float(value)), origins=copy.deepcopy(self.allowed[key]),
            origins_sha256=digest(self.allowed[key])))
        return repr(float(value))


def implementation():
    paths = [Path(__file__).resolve(), Path(graph_lowering.__file__).resolve(),
             Path(capacity_lowering.__file__).resolve()]
    return {p.name:dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}


def source_statistics(source):
    tree = ast.parse(source)
    return dict(ast_nodes=sum(1 for _ in ast.walk(tree)),
        assignments=sum(isinstance(n, ast.Assign) for n in ast.walk(tree)),
        additions=sum(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) for n in ast.walk(tree)),
        subtractions=sum(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) for n in ast.walk(tree)),
        multiplications=sum(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) for n in ast.walk(tree)),
        rotations=sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "rotate" for n in ast.walk(tree)),
        source_bytes=len(source.encode()))


def literal_map(source, fallbacks):
    from decorated_functions import literal
    locations = {key:[] for key in fallbacks}
    class Visit(ast.NodeVisitor):
        def visit(self, node):
            number = literal(node)
            if number is not None:
                key = float(number).hex()
                if key in locations:
                    locations[key].append([node.lineno, node.col_offset, node.end_lineno, node.end_col_offset])
                return None
            return super().visit(node)
    Visit().visit(ast.parse(source))
    return [dict(fallbacks[key], source_locations=locations[key],
                 source_locations_sha256=digest(locations[key])) for key in sorted(fallbacks)]


def lower(request, *, source_kind=SOURCE_KIND):
    require(type(source_kind) is str and source_kind == SOURCE_KIND, "Rule-only source authority required")
    original_sha = digest(request)
    q = copy.deepcopy(request)
    validate_request(q)  # Restores compact storage only in this private copy.
    require(q.get("compiler_configuration", {}).get("backend") == "poseidon_cpu", "Explicit Poseidon CPU baseline")
    require(not any(k in q for k in ("construction_exercise", "construction_profile", "upstream_helpers")),
            "Free native rule baseline only")
    model = q["model"]
    checked = validate(model)
    floor = budget_floor(q, checked)
    require(floor["total"] <= 1024, "Replicated-cell primitive lower bound exceeds existing operation budget")
    em = ReplicatedEmitter(q, literal_sources(model, checked))
    values = {name:np.asarray(value, dtype=np.float64) for name, value in model["constants"].items()}
    input_bindings = []
    for spec in model["inputs"]:
        bindings = [b for b in q["layout"]["inputs"] if b["name"] == spec["name"]]
        offset = 0
        parts = []
        for b in bindings:
            count = b.get("elements", math.prod(spec["shape"]))
            require(b.get("offset", 0) == offset and 0 < count <= em.period, "Input cell chunk coverage")
            parts.append(em.input(b["dsl_name"], count))
            offset += count
        require(offset == math.prod(spec["shape"]), "Input logical cell count")
        values[spec["name"]] = np.concatenate(parts).reshape(spec["shape"])
        input_bindings.append(dict(name=spec["name"], shape=spec["shape"], chunks=copy.deepcopy(bindings),
                                   representation="independent_replicated_logical_cells"))
    for node in model["nodes"]:
        op, attrs = node["op"], node["attrs"]
        xs = [values[ref] for ref in node["inputs"]]
        x = xs[0]
        if op == "add": y = x+xs[1]
        elif op == "subtract": y = x-xs[1]
        elif op == "multiply": y = x*xs[1]
        elif op == "negate": y = -x
        elif op == "square": y = x*x
        elif op == "power": y = x**attrs["exponent"]
        elif op == "linear":
            w = xs[1]
            y = np.empty((*x.shape[:-1], w.shape[0]), dtype=object)
            for index in np.ndindex(y.shape):
                total = em.zero
                for j in range(w.shape[1]):
                    total = total+x[(*index[:-1], j)]*float(w[index[-1], j])
                y[index] = total+(float(xs[2][index[-1]]) if len(xs) == 3 else 0)
        elif op == "flatten": y = x.reshape(-1)
        elif op == "reshape": y = x.reshape(attrs["shape"])
        elif op == "transpose": y = np.swapaxes(x, attrs["dim0"], attrs["dim1"])
        elif op == "permute": y = x.transpose(attrs["dims"])
        elif op == "rotate": y = np.roll(x.reshape(-1), -attrs["step"]).reshape(x.shape)
        elif op == "concat": y = np.concatenate(xs, axis=attrs["axis"])
        elif op == "stack": y = np.stack(xs, axis=attrs["axis"])
        elif op == "slice":
            index = [slice(None)]*x.ndim
            index[attrs["axis"]] = slice(attrs["start"], attrs["stop"], attrs["step"])
            y = x[tuple(index)]
        elif op == "split": y = np.split(x, np.cumsum(attrs["sections"])[:-1], axis=attrs["axis"])
        elif op in ("sum", "mean"):
            axes = tuple(i % x.ndim for i in attrs["axes"])
            y = np.asarray(np.sum(x, axis=axes, keepdims=attrs["keepdims"]), dtype=object)
            if op == "mean": y = np.asarray(y/math.prod(x.shape[i] for i in axes), dtype=object)
            if y.ndim == 0: y = y.reshape(1)
        elif op == "batch_norm":
            mean, var, gamma, beta = xs[1:]
            shape = [1, x.shape[1]]+[1]*(x.ndim-2)
            gain = np.asarray([float(g)/math.sqrt(float(v)+attrs["eps"]) for v, g in zip(var, gamma)])
            bias = np.asarray([float(b)-float(m)*g for m, b, g in zip(mean, beta, gain)])
            y = x*gain.reshape(shape)+bias.reshape(shape)
        elif op == "polynomial":
            coeff = xs[1]
            y = np.empty(x.shape, dtype=object)
            if attrs["basis"] == "power":
                for index in np.ndindex(x.shape): y[index] = power_polynomial(x[index], coeff, em.zero)
            else:
                from chebyshev_lowering import balanced
                for index in np.ndindex(x.shape): y[index] = balanced(x[index], coeff, em.zero)
        elif op.startswith(("conv", "avg_pool")): y = emit_spatial(op, xs, attrs, em)
        else: raise ValueError("No replicated-cell rule emitter for "+op)
        for name, result in zip(node["outputs"], y if op == "split" else [y]):
            values[name] = np.asarray(result, dtype=object)
            require(list(values[name].shape) == checked["shapes"][name], "Rule logical shape mismatch")
    outputs = []
    output_bindings = []
    for output, binding in zip(model["outputs"], q["layout"]["outputs"]):
        value = values[output["value"]].reshape(-1)
        require(value.size == math.prod(binding["shape"]), "Output logical cell count")
        chunks = binding.get("chunks", [dict(ciphertext=len(outputs), offset=0, elements=value.size)])
        offset = 0
        for chunk in chunks:
            require(chunk["ciphertext"] == len(outputs) and chunk["offset"] == offset and
                    0 < chunk["elements"] <= em.period, "Output physical chunk coverage")
            stop = offset+chunk["elements"]
            outputs.append(em.pack(value[offset:stop]).name)
            output_bindings.append(dict(name=output["name"], **copy.deepcopy(chunk)))
            offset = stop
        require(offset == value.size, "Output complete C-order coverage")
    require(len(outputs) == q["layout"]["output_ciphertexts"], "Output physical ciphertext count")
    names = [b["dsl_name"] for b in q["layout"]["inputs"]]+["zero_ct"]
    source = '@hc.func("'+','.join('c' for _ in names)+'")\ndef golden('+', '.join(names)+'):\n'
    source += '\n'.join(em.lines)+'\n    return ['+', '.join(outputs)+']\n'
    raw_sha = hashlib.sha256(source.encode()).hexdigest()
    source = capacity_lowering.compact_scalar_source(source)
    from unified_capacity_constants import request_version, INDEX_VERSION
    from unified_capacity_layout import ABIS
    output_identity = None
    if q["layout"]["execution_abi"] in ABIS and request_version(q) == INDEX_VERSION:
        from capacity_output_identity import distinct_outputs
        source, output_identity = distinct_outputs(source)
    candidate = dict(schema=1, request_id=q["request_id"], hecate_source=source)
    static = validate_candidate(candidate, q)
    require(digest(request) == original_sha, "Caller request changed")
    fallback = literal_map(source, em.fallbacks)
    require(all(row["source_locations"] for row in fallback), "Literal source occurrence missing")
    record = dict(format=FORMAT, version=VERSION, source_kind=source_kind,
        strategy="independent_replicated_logical_cells_final_physical_pack",
        request_id=q["request_id"], input_request_sha256=original_sha, restored_request_sha256=digest(q),
        model_sha256=digest(model), public_constants_sha256=digest(q["public_constants"]),
        layout_sha256=digest(q["layout"]), compiler_configuration_sha256=digest(q["compiler_configuration"]),
        implementation=implementation(), raw_source_sha256=raw_sha,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inputs=input_bindings, outputs=output_bindings,
        missing_literals=fallback, literal_source_map_sha256=digest(fallback),
        primitive_work_lower_bound=floor, emitted_operations=len(em.lines), operations=source_statistics(source),
        expanded_native_work=static["functions"]["golden"]["expanded_cost"],
        output_identity=output_identity, physical_input_chunks=len(q["layout"]["inputs"]),
        physical_output_chunks=len(outputs), logical_cell_representation="replicated_over_full_period",
        request_changed=False, model_changed=False, public_registry_changed=False,
        compiler_profile_changed=False, reference_changed=False, provider_candidate_rewritten=False,
        new_agent_generation=False, compiler_run=False, encrypted_execution=False,
        depth_is_capacity_proof=False)
    record["binding"] = digest(record)
    return source, record
