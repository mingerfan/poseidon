"""Data-only, user-defined static graphs; no Python import, pickle or eval.

Schemas 2/3 are model input formats, NOT upstream Hecate language versions.
Schema 2 retains one input; schema 3 has 2..4 ordered independent encrypted
inputs. Each still has four logical elements. Bounded Conv/AvgPool use explicit
channel-first semantics; larger packing and unsupported spatial modes fail closed.
"""
import copy
import math
import re

from seal_artifact_gate import require
from spatial_ops import OPS as SPATIAL_OPS, INPUT_SHAPES, geometry

OPS = frozenset(("add", "multiply", "subtract", "negate", "square", "power", "linear", "flatten", "rotate", "batch_norm")) | SPATIAL_OPS
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,47}\Z")
CASE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
LINEAR_WIDTH_LIMIT = 8


def array_shape(value, depth=0):
    require(depth <= 4, "Public arrays have at most four dimensions")
    if type(value) in (int, float):
        require(abs(value) <= 1024 and math.isfinite(value), "Invalid public number")
        return ()
    require(type(value) is list and 1 <= len(value) <= 128, "Invalid public array")
    children = [array_shape(x, depth + 1) for x in value]
    require(all(x == children[0] for x in children), "Ragged public arrays are forbidden")
    shape = (len(value), *children[0])
    require(math.prod(shape) <= 128, "Public array element limit")
    return shape


def input_specs(data):
    """Ordered user names/shapes; never infer argument order from a JSON object."""
    if data.get("schema") == 2:
        return [{"name": "x", "shape": data["input_shape"]}]
    return data["inputs"]


def validate_graph(data):
    require(type(data) is dict and type(data.get("schema")) is int and data["schema"] in (2, 3),
            "Unknown graph schema")
    field = "input_shape" if data["schema"] == 2 else "inputs"
    require(set(data) == {"schema", "id", field, "constants", "nodes", "output"}, "Unexpected graph fields")
    require(type(data["id"]) is str and CASE_ID.fullmatch(data["id"]), "Invalid graph id")
    specs = input_specs(data)
    require(type(specs) is list and (len(specs) == 1 if data["schema"] == 2 else 2 <= len(specs) <= 4),
            "Invalid encrypted input count")
    symbols = {}
    for spec in specs:
        require(type(spec) is dict and set(spec) == {"name", "shape"}, "Invalid input descriptor")
        name, shape = spec["name"], spec["shape"]
        require(type(name) is str and NAME.fullmatch(name) and name not in symbols, "Invalid or duplicate input name")
        require(type(shape) is list and all(type(v) is int for v in shape) and
                tuple(shape) in INPUT_SHAPES, "Current encrypted ABI requires four input elements")
        symbols[name] = ("cipher", tuple(shape), True)
    constants = data["constants"]
    require(type(constants) is dict and len(constants) <= 32, "Public constant count limit")
    for name, value in constants.items():
        require(type(name) is str and NAME.fullmatch(name) and name not in symbols,
                "Invalid or duplicate public constant name")
        symbols[name] = ("plain", array_shape(value), False)
    nodes = data["nodes"]
    require(type(nodes) is list and 1 <= len(nodes) <= 64, "Graph node count limit")
    for node in nodes:
        require(type(node) is dict and type(node.get("op")) is str and node["op"] in OPS,
                "Unsupported graph operator")
        op = node["op"]
        extra = ({"exponent"} if op == "power" else {"weight", "bias"} if op == "linear"
                 else {"step"} if op == "rotate" else set())
        if op == "batch_norm":
            extra={"running_mean","running_var","weight","bias","eps"}
        if op in SPATIAL_OPS:
            extra = {"weight", "bias", "stride", "padding"} if op.startswith("conv") else {
                "kernel", "stride", "padding", "count_include_pad"}
            if op.startswith("conv"):
                extra |= set(node) & {"dilation", "groups"}
        require(set(node) == {"id", "op", "inputs"} | extra, "Unexpected operator fields")
        name = node["id"]
        require(type(name) is str and NAME.fullmatch(name) and name not in symbols,
                "Invalid or duplicate graph value")
        inputs = node["inputs"]
        require(type(inputs) is list and len(inputs) == (2 if op in ("add", "multiply", "subtract") else 1),
                "Invalid operator arity")
        require(all(type(v) is str and v in symbols for v in inputs),
                "Undefined value, forward reference or cycle")
        args = [symbols[v] for v in inputs]
        left = args[0]
        if op in ("add", "multiply", "subtract"):
            right = args[1]
            require(left[0] == "cipher" and (len(left[1]) == 1 or not left[2]),
                    "Binary left operand must be a cipher vector; flatten explicitly")
            if right[0] == "cipher":
                require(right[1:] == left[1:], "Cipher shape or packing mismatch")
            else:
                require(right[1] in ((), (1,), left[1]), "Unsupported plaintext broadcast")
            result = left
        elif op == "batch_norm":
            from batch_norm_ops import coefficients
            require(left[0]=="cipher", "BatchNorm requires ciphertext input")
            args_bn=[]
            for field in ("running_mean","running_var","weight","bias"):
                key=node[field]
                require((key is None and field in ("weight","bias")) or
                        type(key) is str and key in constants, "Invalid BatchNorm public "+field)
                args_bn.append(None if key is None else constants[key])
            coefficients(left[1],*args_bn,node["eps"])
            result=left
        elif op in SPATIAL_OPS:
            require(left[0] == "cipher", "Spatial operand must be ciphertext")
            if op.startswith("conv"):
                weight, bias = node["weight"], node["bias"]
                require("dilation" not in node or type(node["dilation"]) is list,
                        "Conv dilation must be an explicit per-axis list")
                require(type(weight) is str and weight in constants, "Conv weight must name a public array")
                wshape = symbols[weight][1]
                kernel = wshape[2:]
                require(bias is None or type(bias) is str and bias in constants and len(wshape) >= 1 and
                        symbols[bias][1] == (wshape[0],), "Invalid Conv bias")
            else:
                wshape, kernel = None, node["kernel"]
            out = geometry(op, left[1], wshape, kernel, node["stride"], node["padding"], node.get("count_include_pad", True),
                           dilation=node.get("dilation"), groups=node.get("groups", 1))
            result = ("cipher", out, False)
        elif op == "linear":
            weight, bias = node["weight"], node["bias"]
            require(type(weight) is str and weight in constants, "Linear weight must name a public array")
            wshape = symbols[weight][1]
            require(left[0] == "cipher" and len(left[1]) == 1 and len(wshape) == 2 and
                    wshape[1] == left[1][0] and 1 <= wshape[0] <= LINEAR_WIDTH_LIMIT, "Invalid Linear shape")
            require(bias is None or (type(bias) is str and bias in constants and
                                    symbols[bias][1] == (wshape[0],)), "Invalid Linear bias")
            result = ("cipher", (wshape[0],), False)
        elif op == "rotate":
            require(left == ("cipher", (4,), True), "Rotation requires a packed four-element vector")
            require(type(node["step"]) is int and node["step"] in (-3, -2, -1, 1, 2, 3), "Unsupported rotation step")
            result = left
        elif op == "flatten":
            require(left[0] == "cipher", "Flatten requires ciphertext tensor")
            result = ("cipher", (math.prod(left[1]),), left[2])
        else:
            require(left[0] == "cipher" and (len(left[1]) == 1 or not left[2]), "Unary operand must be a cipher vector or scalar-neuron tensor")
            if op == "power":
                require(type(node["exponent"]) is int and node["exponent"] in (2, 4), "Only powers 2 and 4")
            result = left
        symbols[name] = result
    output = data["output"]
    require(type(output) is str and output in symbols and symbols[output][0] == "cipher" and
            len(symbols[output][1]) == 1 and 1 <= symbols[output][1][0] <= 4,
            "Output must name a cipher vector with at most four elements")
    return {"schema": data["schema"], "operators": sorted({n["op"] for n in nodes}),
            "output_shape": list(symbols[output][1]), "input_slot_period": 4,
            "compiled": False, "encrypted_execution": False}


def build_graph_model(data):
    """Construct only trusted Torch operations from checked data, not user code."""
    validate_graph(data)
    import torch
    import torch.nn.functional as F

    class DataGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.definition = copy.deepcopy(data)
            for name, value in data["constants"].items():
                self.register_buffer("public_" + name, torch.tensor(value, dtype=torch.float64))

        def evaluate(self, args):
            values = {name: getattr(self, "public_" + name) for name in self.definition["constants"]}
            values.update({spec["name"]: value for spec, value in zip(input_specs(self.definition), args)})
            for node in self.definition["nodes"]:
                op = node["op"]
                args = [values[v] for v in node["inputs"]]
                if op == "add":
                    result = args[0] + args[1]
                elif op == "multiply":
                    result = args[0] * args[1]
                elif op == "subtract":
                    result = args[0] - args[1]
                elif op == "negate":
                    result = -args[0]
                elif op == "square":
                    result = torch.square(args[0])
                elif op == "power":
                    result = torch.pow(args[0], node["exponent"])
                elif op == "flatten":
                    result = torch.flatten(args[0])
                elif op == "rotate":
                    result = torch.roll(args[0], -node["step"], 0)
                elif op == "batch_norm":
                    result=F.batch_norm(args[0],values[node["running_mean"]],values[node["running_var"]],
                        None if node["weight"] is None else values[node["weight"]],
                        None if node["bias"] is None else values[node["bias"]],False,0.1,node["eps"])
                elif op == "linear":
                    result = F.linear(args[0], values[node["weight"]],
                                      None if node["bias"] is None else values[node["bias"]])
                elif op in ("conv1d", "conv2d"):
                    fn = F.conv1d if op == "conv1d" else F.conv2d
                    result = fn(args[0], values[node["weight"]], None if node["bias"] is None else values[node["bias"]],
                                tuple(node["stride"]), tuple(node["padding"]),
                                tuple(node.get("dilation", [1]*(1 if op == "conv1d" else 2))), node.get("groups", 1))
                elif op in ("avg_pool1d", "avg_pool2d"):
                    fn = F.avg_pool1d if op == "avg_pool1d" else F.avg_pool2d
                    result = fn(args[0], tuple(node["kernel"]), tuple(node["stride"]), tuple(node["padding"]),
                                False, node["count_include_pad"])
                else:
                    raise ValueError("Unvalidated graph operator")
                values[node["id"]] = result
            return values[self.definition["output"]]

    # Fixed trusted Python signatures, not exec-generated code or variadic FX placeholders.
    class One(DataGraph):
        def forward(self, x):
            return self.evaluate((x,))
    class Two(DataGraph):
        def forward(self, x, y):
            return self.evaluate((x, y))
    class Three(DataGraph):
        def forward(self, x, y, z):
            return self.evaluate((x, y, z))
    class Four(DataGraph):
        def forward(self, x, y, z, t):
            return self.evaluate((x, y, z, t))
    cls = (One, Two, Three, Four)[len(input_specs(data))-1]
    shape = list(data["input_shape"]) if data["schema"] == 2 else {"inputs": copy.deepcopy(data["inputs"])}
    return cls().eval(), shape


def evaluate_reference(data, logical_input):
    """Independent float64-style Python arithmetic; no Torch/DSL/runtime imports.

    Linear uses explicit row dot products, independently of Hecate rotation or
    the Torch graph. This reference never consumes ciphertexts or decrypted data.
    """
    validate_graph(data)
    values = copy.deepcopy(data["constants"])
    supplied = {"x": logical_input} if data["schema"] == 2 else logical_input
    specs = input_specs(data)
    require(type(supplied) is dict and set(supplied) == {s["name"] for s in specs}, "Reference input name mismatch")
    for spec in specs:
        require(array_shape(supplied[spec["name"]]) == tuple(spec["shape"]), "Reference input shape mismatch")
        values[spec["name"]] = copy.deepcopy(supplied[spec["name"]])

    def flat(value):
        if type(value) is list:
            return [item for child in value for item in flat(child)]
        return [float(value)]

    for node in data["nodes"]:
        op = node["op"]
        a = flat(values[node["inputs"][0]])
        if op in ("add", "multiply", "subtract"):
            b = flat(values[node["inputs"][1]])
            if len(b) == 1:
                b = b * len(a)
            if op == "add":
                result = [u + v for u, v in zip(a, b)]
            elif op == "multiply":
                result = [u * v for u, v in zip(a, b)]
            else:
                result = [u - v for u, v in zip(a, b)]
        elif op in SPATIAL_OPS:
            from spatial_ops import reference as spatial_reference
            value = values[node["inputs"][0]]
            weight = values[node["weight"]] if op.startswith("conv") else None
            bias = None if not op.startswith("conv") or node["bias"] is None else values[node["bias"]]
            kernel = array_shape(weight)[2:] if weight is not None else node["kernel"]
            result = spatial_reference(op, value, array_shape(value), weight, bias, kernel,
                                       node["stride"], node["padding"], node.get("count_include_pad", True),
                                       dilation=node.get("dilation"), groups=node.get("groups", 1))
        elif op == "batch_norm":
            shape=array_shape(values[node["inputs"][0]])
            spatial=math.prod(shape[2:])
            mean=values[node["running_mean"]];var=values[node["running_var"]]
            gamma=[1.]*shape[1] if node["weight"] is None else values[node["weight"]]
            beta=[0.]*shape[1] if node["bias"] is None else values[node["bias"]]
            result=[]
            for i,x in enumerate(a):
                ch=(i//spatial)%shape[1]
                # Independent formula, not folded gain/offset from the compiler.
                result.append((x-mean[ch])/math.sqrt(var[ch]+node["eps"])*gamma[ch]+beta[ch])
        elif op == "linear":
            result = [math.fsum(float(w) * x for w, x in zip(row, a))
                      for row in values[node["weight"]]]
            if node["bias"] is not None:
                result = [u + float(v) for u, v in zip(result, values[node["bias"]])]
        elif op == "negate":
            result = [-u for u in a]
        elif op in ("square", "power"):
            exponent = 2 if op == "square" else node["exponent"]
            result = [u ** exponent for u in a]
        elif op == "flatten":
            result = a
        elif op == "rotate":
            step = node["step"] % len(a)
            result = a[step:] + a[:step]
        else:
            raise ValueError("Unvalidated reference operator")
        if op not in SPATIAL_OPS | {"linear", "flatten"} and len(array_shape(values[node["inputs"][0]])) > 1:
            from spatial_ops import nested
            result = nested(result, array_shape(values[node["inputs"][0]]))
        require(all(math.isfinite(x) for x in flat(result)), "Nonfinite reference output")
        values[node["id"]] = result
    return flat(values[data["output"]])
