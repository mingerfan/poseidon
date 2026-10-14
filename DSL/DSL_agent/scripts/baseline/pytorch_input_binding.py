"""Explicit structured forward calls; trusted FX capture and data-only replay."""
import copy, inspect, math, re
from dataclasses import dataclass
from benchmark_graph import require, canonical, digest

SOURCE_FORMAT = "poseidon-trusted-pytorch-v26"
CALL_FORMAT = "poseidon-pytorch-call-v1"
PLAN_FORMAT = "poseidon-pytorch-input-plan-v1"
FLOW_FORMAT = "poseidon-pytorch-input-flow-v1"
KINDS = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

@dataclass(frozen=True)
class TensorRef:
    name: str

def _literal(value):
    require(value is None or type(value) in (bool, int, float, str),
            "input_call_public_literal_type")
    if type(value) is int:
        require(abs(value) <= 2**53, "input_call_integer_precision")
    if type(value) is float:
        require(math.isfinite(value) and abs(value) <= 1024, "input_call_finite_scalar")
    if type(value) is str:
        require(len(value) <= 256, "input_call_string_budget")
    return value

def _tree(value, names, leaves, *, depth=0, budget=None):
    if budget is None: budget = [256]
    budget[0] -= 1
    require(depth <= 8 and budget[0] >= 0, "input_call_tree_capacity")
    require(type(value) is dict and type(value.get("kind")) is str, "input_call_tree_fields")
    kind = value["kind"]
    if kind == "tensor":
        require(set(value) == {"kind", "name"} and type(value["name"]) is str
                and value["name"] in names, "input_call_tensor_binding")
        leaves.append(value["name"])
    elif kind == "literal":
        require(set(value) == {"kind", "value"}, "input_call_literal_fields")
        _literal(value["value"])
    elif kind in ("tuple", "list", "dict"):
        require(set(value) == {"kind", "items"} and type(value["items"]) is list
                and len(value["items"]) <= 32, "input_call_container")
        items = value["items"]
        if kind == "dict":
            require(all(type(v) is list and len(v) == 2 and type(v[0]) is str
                        and len(v[0]) <= 256 for v in items), "input_call_dictionary_keys")
            require(len({v[0] for v in items}) == len(items), "input_call_duplicate_key")
            items = [v[1] for v in items]
        for item in items: _tree(item, names, leaves, depth=depth+1, budget=budget)
    else:
        raise ValueError("input_call_tree_kind")
    return value

def validate_call(call, inputs):
    require(type(inputs) is list and 1 <= len(inputs) <= 4
            and all(type(i) is dict and type(i.get("name")) is str for i in inputs),
            "input_call_input_inventory")
    names = [i["name"] for i in inputs]
    require(len(set(names)) == len(names), "input_call_duplicate_input")
    require(type(call) is dict and set(call) == {"format", "args", "kwargs"}
            and call["format"] == CALL_FORMAT, "input_call_fields")
    require(type(call["args"]) is list and len(call["args"]) <= 16
            and type(call["kwargs"]) is dict and len(call["kwargs"]) <= 16
            and all(type(n) is str and NAME.fullmatch(n) for n in call["kwargs"]),
            "input_call_arguments")
    leaves = []; budget = [256]
    for value in [*call["args"], *call["kwargs"].values()]:
        _tree(value, names, leaves, budget=budget)
    require(len(leaves) == len(names) and set(leaves) == set(names),
            "input_call_missing_or_duplicate_tensor")
    require(len(canonical(call)) <= 8192, "input_call_byte_budget")
    return copy.deepcopy(call)

def materialize(tree, tensors):
    kind = tree["kind"]
    if kind == "tensor": return tensors[tree["name"]]
    if kind == "literal": return tree["value"]
    values = tree["items"]
    if kind == "dict": return {k: materialize(v, tensors) for k, v in values}
    values = [materialize(v, tensors) for v in values]
    return tuple(values) if kind == "tuple" else values

def public_tree(value, *, depth=0):
    require(depth <= 8, "input_call_default_depth")
    if type(value) in (tuple, list):
        require(len(value) <= 32, "input_call_default_capacity")
        return dict(kind="tuple" if type(value) is tuple else "list",
                    items=[public_tree(v, depth=depth+1) for v in value])
    if type(value) is dict:
        require(len(value) <= 32 and all(type(k) is str and len(k) <= 256 for k in value),
                "input_call_default_dictionary")
        return dict(kind="dict", items=[[k, public_tree(v, depth=depth+1)] for k, v in value.items()])
    return dict(kind="literal", value=_literal(value))

def fx_default_tree(value):
    from torch.fx.immutable_collections import immutable_list,immutable_dict
    def plain(v,depth=0):
        require(depth<=8,"input_call_default_depth")
        if type(v)in(list,tuple,immutable_list):
            values=[plain(x,depth+1)for x in v]
            return tuple(values)if type(v)is tuple else values
        if type(v)in(dict,immutable_dict):return {k:plain(x,depth+1)for k,x in v.items()}
        return v
    return public_tree(plain(value))

def _bound_tree(value):
    if type(value) is TensorRef: return dict(kind="tensor", name=value.name)
    if type(value) in (tuple, list):
        return dict(kind="tuple" if type(value) is tuple else "list", items=[_bound_tree(v) for v in value])
    if type(value) is dict: return dict(kind="dict", items=[[k, _bound_tree(v)] for k, v in value.items()])
    return public_tree(value)

def signature_record(module):
    parameters = inspect.signature(module.forward).parameters.values()
    result = []
    for p in parameters:
        require(p.kind.name in KINDS and NAME.fullmatch(p.name), "input_call_variadic_or_parameter")
        row = dict(name=p.name, kind=p.kind.name)
        if p.default is not inspect.Parameter.empty: row["default"] = public_tree(p.default)
        result.append(row)
    require(1 <= len(result) <= 16 and len(canonical(result)) <= 8192, "input_call_signature_budget")
    return result

def plan(spec, signature):
    call = validate_call(spec["call"], spec["inputs"])
    require(type(signature) is list and 1 <= len(signature) <= 16, "input_call_signature")
    parameters = []
    for row in signature:
        require(type(row) is dict and set(row) in ({"name", "kind"}, {"name", "kind", "default"})
                and type(row["name"]) is str and NAME.fullmatch(row["name"])
                and row["kind"] in KINDS, "input_call_signature_parameter")
        default = inspect.Parameter.empty
        if "default" in row:
            leaves = []; _tree(row["default"], [], leaves)
            default = materialize(row["default"], {})
        parameters.append(inspect.Parameter(row["name"], getattr(inspect.Parameter, row["kind"]), default=default))
    try:
        sig = inspect.Signature(parameters)
        tensors = {i["name"]: TensorRef(i["name"]) for i in spec["inputs"]}
        bound = sig.bind(*(materialize(v, tensors) for v in call["args"]),
                         **{n: materialize(v, tensors) for n, v in call["kwargs"].items()})
        bound.apply_defaults()
    except (ValueError, TypeError) as error:
        raise ValueError("input_call_signature_mismatch") from error
    rows = [dict(name=p.name, kind=p.kind.name, tree=_bound_tree(bound.arguments[p.name]))
            for p in parameters]
    placeholders = []
    for row in rows:
        flat = []
        def visit(tree):
            if tree["kind"] in ("tensor", "literal"): flat.append(tree); return
            for item in tree["items"]: visit(item[1] if tree["kind"] == "dict" else item)
        visit(row["tree"])
        for index, leaf in enumerate(flat, 1):
            slot = dict(target=f'{row["name"]}_{index}', parameter=row["name"], **copy.deepcopy(leaf))
            placeholders.append(slot)
    require(len(placeholders) <= 256, "input_call_placeholder_capacity")
    result = dict(format=PLAN_FORMAT, call=call, inputs=copy.deepcopy(spec["inputs"]),
                  signature=copy.deepcopy(signature), parameters=rows, placeholders=placeholders)
    require(len(canonical(result)) <= 32768, "input_call_plan_budget")
    result["binding"] = digest(result)
    return result

def placeholder_defaults(record):
    defaults={r["name"]:r.get("default")for r in record["signature"]}
    return {s["target"]:[]if defaults[s["parameter"]]is None else[copy.deepcopy(defaults[s["parameter"]])]
            for s in record["placeholders"]}

def concrete_args(record):
    from torch.fx import PH
    return {r["name"]: materialize(r["tree"], {i["name"]: PH for i in record["inputs"]})
            for r in record["parameters"]}

def invoke(module, record, values):
    require(type(values) is dict and set(values) == {i["name"] for i in record["inputs"]},
            "input_call_runtime_names")
    args = []; kwargs = {}
    for row in record["parameters"]:
        v = materialize(row["tree"], values)
        if row["kind"] == "KEYWORD_ONLY": kwargs[row["name"]] = v
        else: args.append(v)
    by_id = {id(v):n for n,v in values.items()}
    def snapshot(value, depth=0):
        require(depth <= 8, "input_call_runtime_tree_depth")
        if id(value) in by_id: return dict(kind="tensor",name=by_id[id(value)])
        if type(value) in (tuple,list):
            return dict(kind="tuple"if type(value)is tuple else"list",
                        items=[snapshot(v,depth+1)for v in value])
        if type(value)is dict:
            require(all(type(k)is str for k in value),"input_call_runtime_keys")
            return dict(kind="dict",items=[[k,snapshot(v,depth+1)]for k,v in value.items()])
        return public_tree(value)
    before = snapshot((args,kwargs))
    result = module(*args, **kwargs)
    require(snapshot((args,kwargs)) == before, "input_call_input_container_mutated")
    return result

def reference_module(module, record, *, input_effect_policy="immutable-inputs-v1"):
    import torch
    require(input_effect_policy in("immutable-inputs-v1","registered-working-copy-inputs-v1"),
            "input_call_reference_effect_policy")
    class BoundReference(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.source = module
        def forward(self, *values):
            named = dict(zip([i["name"] for i in record["inputs"]], values))
            require(len(values) == len(named), "input_call_reference_arity")
            saved = {n: v.detach().clone() for n, v in named.items()}
            working={n:v.clone()for n,v in named.items()}if input_effect_policy=="registered-working-copy-inputs-v1"else named
            result = invoke(self.source, record, working)
            require(all(torch.equal(v, saved[n]) for n, v in named.items()),
                    "input_call_reference_tensor_mutated")
            return result
    return BoundReference().eval()

def flatten_samples(record, samples):
    """Bind caller structures to named inputs; public arguments stay frozen."""
    require(type(samples)is list and 1<=len(samples)<=16,"input_call_runtime_sample_count")
    parameters=[inspect.Parameter(r["name"],getattr(inspect.Parameter,r["kind"]),
                default=inspect.Parameter.empty if "default"not in r else materialize(r["default"],{}))
                for r in record["signature"]]
    sig=inspect.Signature(parameters);result=[]
    for sample in samples:
        require(type(sample)is dict and set(sample)=={"args","kwargs"}
                and type(sample["args"])in(list,tuple)and len(sample["args"])<=16
                and type(sample["kwargs"])is dict and len(sample["kwargs"])<=16,
                "input_call_runtime_arguments")
        try:
            bound=sig.bind(*sample["args"],**sample["kwargs"]);bound.apply_defaults()
        except(TypeError,ValueError)as error:raise ValueError("input_call_runtime_signature")from error
        named={}
        def bind(tree,value):
            kind=tree["kind"]
            if kind=="tensor":
                require(tree["name"]not in named,"input_call_runtime_duplicate_tensor")
                named[tree["name"]]=value;return
            if kind=="literal":
                require(type(value)is type(tree["value"])and value==tree["value"],
                        "input_call_runtime_public_value_changed")
                return
            if kind=="dict":
                require(type(value)is dict and set(value)=={k for k,_ in tree["items"]},
                        "input_call_runtime_dictionary")
                for key,t in tree["items"]:bind(t,value[key])
            else:
                require(type(value)is(tuple if kind=="tuple"else list)and len(value)==len(tree["items"]),
                        "input_call_runtime_container")
                for t,v in zip(tree["items"],value):bind(t,v)
        for row in record["parameters"]:bind(row["tree"],bound.arguments[row["name"]])
        require(set(named)=={i["name"]for i in record["inputs"]},"input_call_runtime_names")
        result.append(named)
    return result

def _encoded_literal(value):
    if value is None: return dict(kind="none")
    return dict(kind={bool:"bool", int:"int", float:"float", str:"str"}[type(value)], value=value)

def output_template(codegen):
    info = getattr(codegen, "pytree_info", None)
    if info is None: return None
    from torch.utils._pytree import tree_unflatten
    require(info.out_spec is not None and info.out_spec.num_leaves <= 32, "input_call_output_capacity")
    def encode(value):
        if type(value) is int: return dict(kind="leaf", index=value)
        if type(value) in (tuple, list):
            return dict(kind="tuple" if type(value) is tuple else "list", items=[encode(v) for v in value])
        require(type(value) is dict and all(type(k) is str for k in value), "input_call_output_tree")
        return dict(kind="dict", items=[[k, encode(v)] for k, v in value.items()])
    return encode(tree_unflatten(list(range(info.out_spec.num_leaves)), info.out_spec))

def _restore(template, flat, *, depth=0, used=None):
    if used is None: used = []
    require(depth <= 8 and type(template) is dict, "input_call_output_template")
    kind = template.get("kind")
    if kind == "leaf":
        i = template.get("index")
        require(set(template) == {"kind", "index"} and type(i) is int and 0 <= i < len(flat)
                and i not in used, "input_call_output_leaf")
        used.append(i); return flat[i]
    require(set(template) == {"kind", "items"} and kind in ("tuple", "list", "dict")
            and type(template["items"]) is list and len(template["items"]) <= 32,
            "input_call_output_template")
    if kind == "dict":
        require(all(type(v) is list and len(v) == 2 and type(v[0]) is str for v in template["items"])
                and len({v[0] for v in template["items"]}) == len(template["items"]),
                "input_call_output_keys")
        return {k: _restore(v, flat, depth=depth+1, used=used) for k, v in template["items"]}
    values = [_restore(v, flat, depth=depth+1, used=used) for v in template["items"]]
    return tuple(values) if kind == "tuple" else values

def expected_inventory(source, record, template):
    from pytorch_public_expression_flow import _decode, Ref
    require(type(source) is list and len(source) <= 5000, "input_call_inventory_budget")
    require(all(type(n)is dict and type(n.get("name"))is str for n in source)
            and len({n["name"]for n in source})==len(source),"input_call_unique_nodes")
    require(sum(n.get("op")=="output"for n in source)==1 and source[-1].get("op")=="output",
            "input_call_output_inventory")
    placeholders = [n for n in source if n.get("op") == "placeholder"]
    slots = record["placeholders"]
    require(len(placeholders) == len(slots), "input_call_placeholder_count")
    substitution = {}; tensors = {}; public = {}; guards = set(); equality = {}
    names = {}
    for row, slot in zip(placeholders, slots):
        require(row["target"] == slot["target"], "input_call_placeholder_order")
        names[row["name"]] = Ref(row["name"])
        if slot["kind"] == "tensor":
            declaration = next(i for i in record["inputs"] if i["name"] == slot["name"])
            require(row["tensor"] is not None and row["tensor"]["shape"] == declaration["shape"]
                    and row["tensor"]["dtype"] == "float64", "input_call_placeholder_shape")
            tensors[slot["name"]] = row
        else:
            require(row["tensor"] is None, "input_call_public_placeholder_type")
            substitution[row["name"]] = _encoded_literal(slot["value"])
            public[row["name"]] = slot
    for row in source:
        require(type(row) is dict and set(row) == {"name","op","target","args","kwargs","tensor"}
                and row["name"] not in (set(names)-{p["name"] for p in placeholders}),
                "input_call_inventory_node")
        if row["op"] == "placeholder": continue
        args = _decode(row["args"], names); kwargs = _decode(row["kwargs"], names)
        require(type(args) is tuple and type(kwargs) is dict, "input_call_inventory_arguments")
        target = row["target"]
        if row["op"] == "call_function" and target in ("_operator.eq", "operator.eq"):
            if len(args) == 2 and type(args[0]) is Ref and args[0].name in public:
                slot = public[args[0].name]
                require(not kwargs and slot["value"] is not None
                        and type(args[1]) is type(slot["value"]) and args[1] == slot["value"],
                        "input_call_specialization_guard")
                equality[row["name"]] = slot; guards.add(row["name"])
        if row["op"] == "call_function" and target in ("torch._assert", "torch.fx._symbolic_trace._assert_is_none"):
            require(not kwargs and len(args) == 2 and type(args[0]) is Ref
                    and type(args[1]) is str, "input_call_specialization_guard")
            if target == "torch._assert":
                require(args[0].name in equality, "input_call_non_public_assert")
                slot = equality[args[0].name]
            else:
                require(args[0].name in public and public[args[0].name]["value"] is None,
                        "input_call_non_public_assert")
                slot = public[args[0].name]
            require(args[1] == f'{slot["parameter"]} has been specialized to have value {slot["value"]} but got another value',
                    "input_call_specialization_message")
            guards.add(row["name"])
        names[row["name"]] = Ref(row["name"])
    for name,slot in public.items():
        if slot["value"]is None:
            count=sum(row["op"]=="call_function"and row["target"]=="torch.fx._symbolic_trace._assert_is_none"
                      and row["args"]["items"][0]==dict(kind="fx_value",name=name)for row in source)
        else:count=sum(v["target"]==slot["target"]for v in equality.values())
        require(count==1,"input_call_missing_or_duplicate_guard")
    require(all(sum(row["op"] == "call_function" and row["target"] == "torch._assert"
                    and row["args"]["items"][0] == dict(kind="fx_value", name=n)
                    for row in source) == 1 for n in equality), "input_call_guard_pair")
    def refs(v):
        if type(v) is dict:
            if v.get("kind") == "fx_value": return [v["name"]]
            return [n for value in v.values() for n in refs(value)]
        if type(v) in (list,tuple): return [n for x in v for n in refs(x)]
        return []
    def replace(v):
        if type(v) is dict:
            if v.get("kind") == "fx_value" and v["name"] in substitution: return copy.deepcopy(substitution[v["name"]])
            return {k: replace(x) for k,x in v.items()}
        if type(v) is list: return [replace(x) for x in v]
        return v
    result = []
    for declaration in record["inputs"]:
        row = copy.deepcopy(tensors[declaration["name"]])
        row["target"]=declaration["name"]
        row["args"] = dict(kind="tuple", items=[]); row["kwargs"] = dict(kind="dict", items=[])
        result.append(row)
    for row in source:
        if row["op"] == "placeholder" or row["name"] in guards: continue
        require(not set(refs((row["args"],row["kwargs"]))).intersection(guards | set(public)),
                "input_call_public_placeholder_used_outside_guard")
        item = replace(row)
        if row["op"] == "output" and template is not None:
            values = item["args"]["items"]
            require(len(values) == 1 and values[0]["kind"] in ("list", "tuple"), "input_call_flat_output")
            flat = values[0]["items"]; used = []
            restored = _restore(template, flat, used=used)
            require(sorted(used) == list(range(len(flat))), "input_call_output_leaves")
            def encoded(t):
                if t["kind"] == "leaf": return flat[t["index"]]
                if t["kind"] == "dict":
                    return dict(kind="dict",items=[[k,encoded(v)]for k,v in t["items"]])
                return dict(kind=t["kind"],items=[encoded(v)for v in t["items"]])
            item["args"] = dict(kind="tuple", items=[encoded(template)])
        result.append(item)
    return result, sorted(guards)

def normalize(traced, record):
    import torch
    from pytorch_fused_matrix_normalization import inventory
    source = inventory(traced); template = output_template(traced.graph._codegen)
    defaults=traced.meta.get("input_call_placeholder_defaults")
    require(defaults==placeholder_defaults(record),"input_call_placeholder_defaults")
    expected, guards = expected_inventory(source, record, template)
    graph = torch.fx.Graph(); nodes = {}; source_nodes = {n.name:n for n in traced.graph.nodes}
    for row in expected:
        if row["op"] != "placeholder": break
        node = source_nodes[row["name"]]
        nodes[node] = graph.create_node("placeholder",row["target"],(),{},name=node.name)
        nodes[node].meta = copy.deepcopy(node.meta)
    for node in traced.graph.nodes:
        if node.op == "placeholder" or node.name in guards: continue
        if node.op == "output" and template is not None:
            value = _restore(template, list(node.args[0]))
            def convert(v):
                if isinstance(v,torch.fx.Node): return nodes[v]
                if type(v) in (list,tuple): return type(v)(convert(x) for x in v)
                if type(v) is dict: return {k:convert(x) for k,x in v.items()}
                return v
            copied = graph.output(convert(value))
        else:
            copied = graph.node_copy(node, lambda n:nodes[n])
        copied.meta = copy.deepcopy(node.meta); nodes[node] = copied
    graph.lint(); result = torch.fx.GraphModule(traced, graph)
    result.meta = copy.deepcopy(traced.meta)
    require(inventory(result) == expected, "input_call_normalized_inventory")
    flow = dict(format=FLOW_FORMAT, plan=record, source_inventory=source,
                output_template=template, placeholder_defaults=defaults,
                removed_guards=guards, normalized_inventory=expected)
    flow["binding"] = digest(flow)
    return result, flow

def check_flow(flow, spec):
    require(type(flow) is dict and set(flow) == {"format","plan","source_inventory","output_template",
            "removed_guards","normalized_inventory","binding","placeholder_defaults"} and flow["format"] == FLOW_FORMAT
            and flow["binding"] == digest({k:v for k,v in flow.items() if k != "binding"}),
            "input_call_flow_binding")
    require(flow["plan"] == plan(spec, flow["plan"]["signature"]), "input_call_plan_binding")
    require(flow["placeholder_defaults"]==placeholder_defaults(flow["plan"]),"input_call_placeholder_defaults")
    expected, guards = expected_inventory(flow["source_inventory"], flow["plan"], flow["output_template"])
    require(flow["removed_guards"] == guards and flow["normalized_inventory"] == expected,
            "input_call_flow_changed")
    return flow
