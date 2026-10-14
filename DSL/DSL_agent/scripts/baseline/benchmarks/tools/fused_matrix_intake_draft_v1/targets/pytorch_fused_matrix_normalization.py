"""Registered fused matrix calls over a trusted, metadata-only FX graph."""
import copy, hashlib, math, operator
from pathlib import Path
from benchmark_graph import require, digest, broadcast, dimensions
from model_resources import resolve

POLICY = "poseidon-fx-fused-matrix-draft-v1"
OPERATIONS = ("addmm", "baddbmm", "addbmm")
SCALAR_LIMIT = 1024.

def arguments(args, kwargs, operation, *, method=False):
    require(operation in OPERATIONS, "Registered fused matrix operation")
    names = ("input", "mat1", "mat2") if operation == "addmm" else ("input", "batch1", "batch2")
    allowed = {*names, "alpha", "beta"} | (set() if method else {"out"})
    require(len(args) <= 3 and not (set(kwargs) - allowed), "Fused matrix argument fields")
    values = dict(zip(names, args))
    for name in names:
        if name in kwargs:
            require(name not in values, "Duplicate fused matrix argument")
            values[name] = kwargs[name]
    require(set(values) == set(names), "Fused matrix required operands")
    require(kwargs.get("out") is None, "Fused matrix out mutation is unsupported")
    for name in ("alpha", "beta"):
        scalar = kwargs.get(name, 1)
        require(type(scalar) in (int, float) and math.isfinite(scalar) and abs(scalar) <= SCALAR_LIMIT,
                "Fused matrix finite public scalar budget")
        values[name] = scalar
    return values, names

def plan(operation, shapes, *, alpha=1, beta=1, resource_profile="general-tensors-v1"):
    limits = resolve(resource_profile)
    arguments((0, 0, 0), dict(alpha=alpha, beta=beta), operation)
    require(type(shapes) is list and len(shapes) == 3, "Fused matrix shape inventory")
    for s in shapes:
        require(type(s) is list and len(s) <= limits.rank and
                all(type(n) is int and 0 < n <= limits.axis_size for n in s) and
                math.prod(s) <= limits.tensor_elements, "Fused matrix operand shape budget")
    bias, left, right = shapes
    rank = 2 if operation == "addmm" else 3
    require(len(left) == len(right) == rank, "Fused matrix operand rank")
    require(left[-1] == right[-2], "Fused matrix contraction mismatch")
    if rank == 3: require(left[0] == right[0], "Fused matrix batch mismatch")
    product = ([left[0]] if rank == 3 else []) + [left[-2], right[-1]]
    result = product[-2:] if operation == "addbmm" else product
    dimensions(product, resource_profile=resource_profile)
    dimensions(result, resource_profile=resource_profile)
    try:
        combined = list(broadcast(bias, result))
    except ValueError as error:
        raise ValueError("Fused matrix bias broadcast") from error
    require(combined == result, "Fused matrix bias broadcast")
    return dict(version=POLICY, operation=operation, source_shapes=shapes,
                product_shape=product, result_shape=result, alpha=alpha, beta=beta,
                bias_policy="ignored" if beta == 0 else "scaled_broadcast",
                product_operator="mm" if operation == "addmm" else "bmm",
                reduction_axis=0 if operation == "addbmm" else None,
                resource_profile=resource_profile)

def _state(module):
    import torch
    records = {}
    for name, value in module.state_dict().items():
        require(isinstance(value, torch.Tensor), "Fused matrix public state tensor")
        raw = value.detach().cpu().contiguous().numpy()
        records[name] = dict(shape=list(raw.shape), dtype=raw.dtype.str,
                             sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    return records

def normalize(traced, inputs, *, resource_profile="general-tensors-v1"):
    import torch
    from pytorch_static_trace import ALIAS_POLICY, policy_record
    from pytorch_arithmetic_normalization import clone_public_state
    require(isinstance(traced, torch.fx.GraphModule), "Trusted fused matrix FX graph")
    limits = resolve(resource_profile)
    require(traced.meta.get("fused_matrix_policy") == POLICY, "Explicit fused matrix trace policy")
    require(traced.meta.get("ordered_alias_capture_policy") ==
            policy_record(inputs, resource_profile, policy=ALIAS_POLICY),
            "Ordered fused matrix trace policy")
    source_text = str(traced.graph); state = _state(traced); result = clone_public_state(traced)
    static = {n.name: dict(shape=list(i["shape"]), dtype=i["dtype"]) for n in result.graph.nodes
              if (i := n.meta.get("poseidon_static_tensor")) is not None}
    functions = {torch.addmm:"addmm", torch.baddbmm:"baddbmm", torch.addbmm:"addbmm"}
    roles = {}; entries = []
    def refs(value):
        if isinstance(value, torch.fx.Node):return [value]
        if type(value) in (list, tuple):return [n for v in value for n in refs(v)]
        if type(value) is dict:return [n for v in value.values() for n in refs(v)]
        return []
    def metadata(node, shape):
        node.meta["poseidon_static_tensor"] = dict(policy=ALIAS_POLICY, shape=list(shape), dtype="float64")
        roles[node] = "cipher"
        return node
    for node in list(result.graph.nodes):
        if node.op == "placeholder":roles[node] = "cipher"; continue
        if node.op == "get_attr":roles[node] = "plain"; continue
        op = functions.get(node.target) if node.op == "call_function" else (
             node.target if node.op == "call_method" and node.target in OPERATIONS else None)
        if op is None:
            roles[node] = "cipher" if any(roles.get(n) == "cipher" for n in refs((node.args,node.kwargs))) else "plain"
            continue
        values, names = arguments(node.args, node.kwargs, op, method=node.op == "call_method")
        tensors = [values[n] for n in names]
        require(all(isinstance(v, torch.fx.Node) and v.name in static and static[v.name]["dtype"] == "float64"
                    for v in tensors), "Fused matrix float64 tensor metadata")
        shapes = [static[v.name]["shape"] for v in tensors]
        p = plan(op, shapes, alpha=values["alpha"], beta=values["beta"], resource_profile=resource_profile)
        require(roles.get(tensors[1]) == "cipher" or roles.get(tensors[2]) == "cipher",
                "Fused matrix product requires an encrypted operand")
        require(static.get(node.name) == dict(shape=p["result_shape"], dtype="float64"),
                "Fused matrix result metadata")
        before = dict(fx_node=node.name, call_kind=node.op, operation=op,
                      operands=[v.name for v in tensors], input_kinds=[roles[v] for v in tensors], plan=p)
        emitted = []
        with result.graph.inserting_before(node):
            product = result.graph.call_function(torch.mm if op == "addmm" else torch.bmm,
                                                   args=tuple(tensors[1:]), kwargs={})
            metadata(product, p["product_shape"]); emitted.append(product.name)
            if op == "addbmm":
                product = result.graph.call_function(torch.sum, args=(product,), kwargs=dict(dim=0,keepdim=False))
                metadata(product, p["result_shape"]); emitted.append(product.name)
            if values["alpha"] != 1:
                product = result.graph.call_function(operator.mul, args=(product,values["alpha"]), kwargs={})
                metadata(product, p["result_shape"]); emitted.append(product.name)
        # beta=0 ignores input values in pinned Torch, rather than multiplying
        # them by zero. Earlier source writes remain in their original order.
        node.op = "call_function"; node.target = torch.add
        node.args = (product, 0. if values["beta"] == 0 else tensors[0])
        node.kwargs = {} if values["beta"] == 0 else dict(alpha=values["beta"])
        roles[node] = "cipher"; before["emitted_nodes"] = emitted; entries.append(before)
    count = sum(n.op in ("call_module","call_function","call_method") for n in result.graph.nodes)
    require(count <= limits.original_nodes, "Fused matrix normalization expansion budget")
    result.graph.lint(); result.recompile()
    require(_state(traced) == state == _state(result) and str(traced.graph) == source_text,
            "Fused matrix normalization changed source or public state")
    record = dict(format=POLICY, source_fx_text=source_text,
                  source_fx_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
                  normalized_fx_text=str(result.graph),
                  normalized_fx_sha256=hashlib.sha256(str(result.graph).encode()).hexdigest(),
                  public_state=state, input_signature=copy.deepcopy(inputs), resource_profile=resource_profile,
                  entries=entries, dead_code_policy="retain-source-effects-and-order",
                  source_graph_preserved=True, public_state_preserved=True,
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record["binding_sha256"] = digest(record)
    return result, record

def check_record(record, source, normalized):
    require(type(record) is dict and record.get("format") == POLICY and record.get("binding_sha256") ==
            digest({k:v for k,v in record.items() if k != "binding_sha256"}), "Fused matrix record binding")
    expected, evidence = normalize(source, record["input_signature"], resource_profile=record["resource_profile"])
    require(record == evidence and str(normalized.graph) == str(expected.graph) and
            _state(normalized) == _state(expected), "Fused matrix normalization provenance")
    return record
