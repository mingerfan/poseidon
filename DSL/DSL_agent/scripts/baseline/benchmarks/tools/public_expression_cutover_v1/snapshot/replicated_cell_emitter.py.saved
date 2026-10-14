"""Public-origin scalar emission for deterministic layout baselines.

Provider candidates bypass this emitter and keep the ordinary candidate checks.
Missing literals require an original public constant or registered arithmetic
origin, followed by the unchanged source/type validator.
"""
import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import unified_graph_lowering as graph_lowering
import unified_capacity_lowering as capacity_lowering
from benchmark_graph import require
from unified_graph_lowering import Emitter
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
