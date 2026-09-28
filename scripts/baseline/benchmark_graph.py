"""Versioned, data-only logical graphs. No provider, frontend or execution imports."""
import hashlib
import itertools
import json
import math
import re

FORMAT = "poseidon-model-graph-v1"
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
OPS = {
    "add": set(), "subtract": set(), "multiply": set(), "negate": set(),
    "square": set(), "power": {"exponent"}, "linear": set(),
    "flatten": set(), "reshape": {"shape"}, "transpose": {"dim0", "dim1"},
    "permute": {"dims"}, "rotate": {"step"}, "concat": {"axis"}, "stack": {"axis"},
    "slice": {"axis", "start", "stop", "step"}, "split": {"axis", "sections"},
    "sum": {"axes", "keepdims"}, "mean": {"axes", "keepdims"},
    "batch_norm": {"eps"}, "polynomial": {"basis"},
    "conv1d": {"stride", "padding", "dilation", "groups"},
    "conv2d": {"stride", "padding", "dilation", "groups"},
    "avg_pool1d": {"kernel", "stride", "padding", "count_include_pad"},
    "avg_pool2d": {"kernel", "stride", "padding", "count_include_pad"},
}
BOUNDS = dict(inputs=4, input_elements=256, outputs=4, output_elements=256,
              nodes=64, constants=32, constant_elements=4096, rank=4, json_bytes=131072)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def shape(value):
    if type(value) is list:
        require(bool(value), "Empty public array")
        shapes = [shape(x) for x in value]
        require(all(s == shapes[0] for s in shapes), "Ragged public array")
        out = (len(value), *shapes[0])
        require(len(out) <= 4 and math.prod(out) <= 4096, "Public array budget")
        return out
    require(type(value) in (float, int) and math.isfinite(value) and abs(value) <= 1024,
            "Public values must be finite bounded real numbers")
    return ()


def dimensions(value):
    require(type(value) is list and 1 <= len(value) <= 4 and
            all(type(n) is int and 1 <= n <= 256 for n in value) and math.prod(value) <= 256,
            "Tensor shape budget")
    return tuple(value)


def axis(value, rank, insertion=False):
    n = rank + int(insertion)
    require(type(value) is int and -n <= value < n, "Invalid axis")
    return value % n


def broadcast(a, b):
    pairs = itertools.zip_longest(reversed(a), reversed(b), fillvalue=1)
    out = []
    for x, y in pairs:
        require(x == y or x == 1 or y == 1, "Broadcast shape mismatch")
        out.append(max(x, y))
    return tuple(reversed(out))


def infer(op, ss, a):
    s = ss[0]
    if op in ("add", "subtract", "multiply"):
        require(len(ss) == 2, "Binary arity")
        return [broadcast(*ss)]
    if op in ("negate", "square", "power", "flatten", "reshape", "transpose", "permute",
              "rotate", "slice", "split", "sum", "mean"):
        require(len(ss) == 1, "Unary arity")
    if op in ("negate", "square"):
        return [s]
    if op == "power":
        require(type(a["exponent"]) is int and a["exponent"] in (2, 4), "Power exponent")
        return [s]
    if op == "flatten":
        return [(math.prod(s),)]
    if op == "reshape":
        target = a["shape"]
        require(type(target) is list and 1 <= len(target) <= 4 and
                all(type(x) is int and (x == -1 or 1 <= x <= 256) for x in target) and
                target.count(-1) <= 1, "Reshape dimensions")
        known = math.prod(x for x in target if x != -1)
        require(math.prod(s) % known == 0, "Reshape product")
        out = tuple(math.prod(s)//known if x == -1 else x for x in target)
        require(math.prod(out) == math.prod(s), "Reshape changes element count")
        return [out]
    if op in ("transpose", "permute"):
        dims = list(range(len(s)))
        if op == "transpose":
            i, j = axis(a["dim0"], len(s)), axis(a["dim1"], len(s))
            dims[i], dims[j] = dims[j], dims[i]
        else:
            require(type(a["dims"]) is list and len(a["dims"]) == len(s), "Permutation rank")
            dims = [axis(x, len(s)) for x in a["dims"]]
            require(sorted(dims) == list(range(len(s))), "Permutation axes")
        return [tuple(s[i] for i in dims)]
    if op == "rotate":
        require(type(a["step"]) is int and 0 < abs(a["step"]) < math.prod(s), "Rotation step")
        return [s]
    if op in ("concat", "stack"):
        require(1 <= len(ss) <= 8 and all(len(t) == len(s) for t in ss), "Join arity/rank")
        i = axis(a["axis"], len(s), op == "stack")
        if op == "stack":
            require(all(t == s for t in ss), "Stack shape mismatch")
            out = list(s); out.insert(i, len(ss))
        else:
            require(all(all(t[j] == s[j] for j in range(len(s)) if j != i) for t in ss),
                    "Concat shape mismatch")
            out = list(s); out[i] = sum(t[i] for t in ss)
        return [tuple(out)]
    if op == "slice":
        i = axis(a["axis"], len(s))
        require(all(type(a[k]) is int for k in ("start", "stop", "step")) and a["step"] != 0,
                "Static slice bounds")
        size = len(range(*slice(a["start"], a["stop"], a["step"]).indices(s[i])))
        require(size > 0, "Empty slice not supported")
        out = list(s); out[i] = size
        return [tuple(out)]
    if op == "split":
        i = axis(a["axis"], len(s)); parts = a["sections"]
        require(type(parts) is list and 2 <= len(parts) <= 4 and
                all(type(x) is int and x > 0 for x in parts) and sum(parts) == s[i], "Split sections")
        return [(*s[:i], n, *s[i+1:]) for n in parts]
    if op in ("sum", "mean"):
        require(type(a["axes"]) is list and a["axes"] and type(a["keepdims"]) is bool,
                "Reduction attributes")
        aa = [axis(i, len(s)) for i in a["axes"]]
        require(len(set(aa)) == len(aa), "Duplicate reduction axes")
        out = tuple(1 if i in aa else n for i, n in enumerate(s) if a["keepdims"] or i not in aa)
        return [out or (1,)]
    if op == "linear":
        require(len(ss) in (2, 3) and len(ss[1]) == 2 and ss[1][1] == s[-1], "Linear weight")
        require(len(ss) == 2 or ss[2] == (ss[1][0],), "Linear bias")
        return [(*s[:-1], ss[1][0])]
    if op == "batch_norm":
        require(len(ss) == 5 and len(s) >= 2 and all(t == (s[1],) for t in ss[1:]) and
                type(a["eps"]) in (float, int) and 0 < a["eps"] <= 1, "BatchNorm attributes")
        return [s]
    if op == "polynomial":
        require(len(ss) == 2 and len(ss[1]) == 1 and ss[1][0] <= 128 and
                a["basis"] in ("power", "chebyshev"), "Polynomial coefficients")
        return [s]
    rank = 1 if op.endswith("1d") else 2
    require(len(s) in (rank+1, rank+2), "Spatial input rank")
    batched = len(s) == rank+2
    c = s[-rank-1]
    for k in ("stride", "padding"):
        require(type(a[k]) is list and len(a[k]) == rank and
                all(type(x) is int and (0 if k == "padding" else 1) <= x <= 256 for x in a[k]),
                "Spatial integer parameters")
    if op.startswith("conv"):
        require(len(ss) in (2, 3) and len(ss[1]) == rank+2, "Conv weight rank")
        w = ss[1]; g = a["groups"]
        require(type(g) is int and g > 0 and c % g == 0 and w[0] % g == 0 and w[1] == c//g,
                "Conv groups")
        require(len(ss) == 2 or ss[2] == (w[0],), "Conv bias")
        require(type(a["dilation"]) is list and len(a["dilation"]) == rank and
                all(type(x) is int and 0 < x <= 256 for x in a["dilation"]), "Conv dilation")
        kernel, dilation, co = w[2:], a["dilation"], w[0]
    else:
        require(len(ss) == 1 and type(a["count_include_pad"]) is bool, "Pool arity/divisor")
        kernel, dilation, co = a["kernel"], [1]*rank, c
        require(type(kernel) is list and len(kernel) == rank and
                all(type(x) is int and 0 < x <= 256 for x in kernel) and math.prod(kernel)<=256 and
                all(p <= k//2 for p, k in zip(a["padding"], kernel)), "Pool kernel/padding")
    spatial = tuple((n+2*p-d*(k-1)-1)//st+1 for n,p,d,k,st in
                    zip(s[-rank:], a["padding"], dilation, kernel, a["stride"]))
    require(all(n > 0 for n in spatial), "Empty spatial output")
    return [((s[0],) if batched else ()) + (co,) + spatial]


def validate(model):
    require(type(model) is dict and set(model) == {"format", "id", "inputs", "constants", "nodes", "outputs"}
            and model["format"] == FORMAT, "Invalid model graph format/fields")
    require(type(model["id"]) is str and NAME.fullmatch(model["id"]), "Model id")
    require(len(canonical(model)) <= BOUNDS["json_bytes"], "Model JSON budget")
    inputs, constants, nodes, outputs = (model[k] for k in ("inputs", "constants", "nodes", "outputs"))
    require(type(inputs) is list and 1 <= len(inputs) <= 4 and type(constants) is dict and
            len(constants) <= 32 and type(nodes) is list and 1 <= len(nodes) <= 64 and
            type(outputs) is list and 1 <= len(outputs) <= 4, "Graph count bounds")
    values, kinds = {}, {}
    def declare(n, s, kind):
        require(type(n) is str and NAME.fullmatch(n) and n not in values, "Invalid/duplicate name")
        values[n], kinds[n] = s, kind
    for item in inputs:
        require(type(item) is dict and set(item) == {"name", "shape"}, "Input declaration")
        declare(item["name"], dimensions(item["shape"]), "cipher")
    require(sum(math.prod(s) for s in values.values()) <= 256, "Total input budget")
    for n, v in constants.items():
        declare(n, shape(v), "plain")
    metadata = []
    for node in nodes:
        require(type(node) is dict and set(node) == {"id", "op", "inputs", "attrs", "outputs"},
                "Node fields")
        op, refs, attrs, names = (node[k] for k in ("op", "inputs", "attrs", "outputs"))
        require(type(node["id"]) is str and NAME.fullmatch(node["id"]) and
                node["id"] not in {x["id"] for x in metadata}, "Node identity")
        require(type(op) is str and op in OPS and type(attrs) is dict and set(attrs) == OPS[op],
                "Operator/attributes")
        require(type(refs) is list and 1 <= len(refs) <= 8 and
                all(type(r) is str and r in values for r in refs), "Undefined/forward input")
        require(kinds[refs[0]] == "cipher", "First operand must be encrypted")
        if op in ("linear", "batch_norm", "polynomial") or op.startswith("conv"):
            require(all(kinds[r] == "plain" for r in refs[1:]), "Weights/stats must be public")
        if op in ("stack", "concat"):
            require(all(kinds[r] == "cipher" for r in refs), "Join requires encrypted tensors")
        shapes = infer(op, [values[r] for r in refs], attrs)
        if op in ("add", "subtract", "multiply") and kinds[refs[1]] == "plain":
            require(broadcast(values[refs[0]], values[refs[1]]) == values[refs[0]],
                    "Public broadcast cannot expand encrypted shape")
        if op == "batch_norm":
            require(all(v >= 0 for v in constants[refs[2]]), "Negative BN variance")
        require(type(names) is list and len(names) == len(shapes), "Node output arity")
        for n, s in zip(names, shapes):
            dimensions(list(s))
            declare(n, s, "cipher")
        metadata.append(dict(id=node["id"], shapes=[list(s) for s in shapes]))
    labels = set()
    for out in outputs:
        require(type(out) is dict and set(out) == {"name", "value"} and
                type(out["name"]) is str and NAME.fullmatch(out["name"]) and out["name"] not in labels and
                type(out["value"]) is str and kinds.get(out["value"]) == "cipher", "Output declaration")
        labels.add(out["name"])
    require(sum(math.prod(values[o["value"]]) for o in outputs) <= 256, "Total output budget")
    return dict(shapes={n:list(s) for n,s in values.items()}, kinds=kinds, nodes=metadata,
                output_shapes={o["name"]:list(values[o["value"]]) for o in outputs})


def signature(model, topology=False):
    """Ignore labels and weights; topology additionally ignores sizes and op parameters."""
    checked = validate(model)
    producers = {out:(node,port) for node in model["nodes"] for port,out in enumerate(node["outputs"])}
    values = {spec["name"]:(["input",i] if topology else ["input",i,spec["shape"]])
              for i,spec in enumerate(model["inputs"])}
    constants = {}; nodes = {}; records = []
    def visit(ref):
        if ref in values: return values[ref]
        if ref in model["constants"]:
            if ref not in constants: constants[ref] = len(constants)
            values[ref] = (["public",constants[ref]] if topology else
                           ["public",constants[ref],checked["shapes"][ref]])
        else:
            node,port = producers[ref]
            if node["id"] not in nodes:
                operands = [visit(x) for x in node["inputs"]]
                nodes[node["id"]] = len(records)
                records.append([node["op"],operands,len(node["outputs"])] +
                               ([] if topology else [node["attrs"]]))
                for j,out in enumerate(node["outputs"]): values[out] = ["node",nodes[node["id"]],j]
        return values[ref]
    # Output-rooted traversal canonicalizes independent node ordering and removes
    # unreachable padding from the diversity count; shared nodes remain shared.
    outputs = [visit(o["value"]) for o in model["outputs"]]
    return digest(dict(inputs=len(model["inputs"]) if topology else [s["shape"] for s in model["inputs"]],
                       nodes=records,outputs=outputs))


def samples(model, count=16):
    """Private-to-runner deterministic probes; no candidate inputs or answers."""
    import random
    require(type(count) is int and 4 <= count <= 16, "Probe count")
    import numpy as np
    batches = []
    for test in range(count):
        item = {}
        for j, spec in enumerate(model["inputs"]):
            n = math.prod(spec["shape"])
            rng = random.Random(918273 + test*1009 + j*97)
            if test == 0: flat = [0.0]*n
            elif test == 1: flat = [(-1 if (i+j)%2 else 1)*(i+1)/(n+1) for i in range(n)]
            elif test == 2: flat = [rng.uniform(-1, 1) for _ in range(n)]
            elif test == 3: flat = [(-1.0 if (i+j)%2 else 1.0) for i in range(n)]
            elif test < 8: flat = [float(i == (test-4+j)%n) for i in range(n)]
            else: flat = [rng.uniform(-0.5, 0.5) for _ in range(n)]
            item[spec["name"]] = np.asarray(flat, dtype=np.float64).reshape(spec["shape"])
        batches.append(item)
    return batches
