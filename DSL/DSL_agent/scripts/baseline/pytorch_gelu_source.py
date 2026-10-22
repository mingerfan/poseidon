"""Explicit GELU source identities over unchanged polynomial graph semantics.

Only trusted capture selects this policy. Model JSON does not register callbacks
or select a source function, and historical capture keeps its exact-GELU rule.
"""
import copy
from benchmark_graph import require, digest
from model_decomposition import polynomial_profile
from model_resources import resolve

LEGACY_POLICY = "exact-gelu-source-v1"
POLICY = "poseidon-gelu-source-variants-v1"
FORMAT = "poseidon-pytorch-gelu-source-capture-v1"
VARIANTS = {
    "none": "torch-2.0.1-gelu-erf-v1",
    "tanh": "torch-2.0.1-gelu-tanh-0.044715-v1",
}

def variant(value, policy=LEGACY_POLICY):
    require(policy in (LEGACY_POLICY, POLICY), "Explicit GELU source policy")
    if policy == LEGACY_POLICY:
        require(value == "none", "Exact GELU source required")
    else:
        require(type(value) is str and value in VARIANTS, "Registered GELU source variant")
    return value

def entry(node, selected, approximation):
    variant(selected, POLICY)
    polynomial_profile(approximation)
    require(node.op in ("call_module", "call_function"), "Registered GELU source call")
    return dict(emission_node_id="fx_"+node.name, variant=selected,
                source_identity=VARIANTS[selected],
                call_kind=node.op, module_path=node.target if node.op == "call_module" else None,
                approximation_sha256=digest(approximation))

def record(graph, emissions, entries, *, normalized_origins=None,
           resource_profile="legacy-v1"):
    require(type(entries) is list and len(entries) <= resolve(resource_profile).original_nodes,
            "GELU source inventory bound")
    emission_nodes={n["id"]:n for n in emissions["nodes"]}
    graph_nodes={n["id"]:n for n in graph["nodes"]}
    require(len(emission_nodes)==len(emissions["nodes"]) and
            len(graph_nodes)==len(graph["nodes"]), "GELU graph identifiers")
    require({row["emission_node_id"] for row in entries} ==
            {n["id"] for n in emissions["nodes"] if n["op"]=="gelu"} and
            len({row["emission_node_id"] for row in entries})==len(entries),
            "Complete GELU source inventory")
    rows=[]
    for row in entries:
        require(type(row) is dict and set(row)=={
            "emission_node_id","variant","source_identity","call_kind","module_path","approximation_sha256"},
            "GELU source fields")
        variant(row["variant"], POLICY)
        require(row["source_identity"]==VARIANTS[row["variant"]] and
                row["call_kind"] in ("call_module","call_function") and
                (type(row["module_path"]) is str and 0<len(row["module_path"])<=256
                 if row["call_kind"]=="call_module" else row["module_path"] is None),
                "GELU source identity")
        source=emission_nodes[row["emission_node_id"]]
        require(source["op"]=="gelu" and row["approximation_sha256"]==
                digest(polynomial_profile(source["attrs"]["approximation"])),
                "GELU source polynomial binding")
        names=([row["emission_node_id"]] if normalized_origins is None else
               [n for n,origin in normalized_origins.items()
                if origin.get("source_id")==row["emission_node_id"] and
                   n in graph_nodes and graph_nodes[n]["op"]=="gelu"])
        require(len(names)==1 and names[0] in graph_nodes, "GELU normalized source mapping")
        target=graph_nodes[names[0]]
        require(target["op"]=="gelu" and target["attrs"]==source["attrs"],
                "GELU normalized polynomial identity")
        rows.append(dict(copy.deepcopy(row),node_id=names[0]))
    require({row["node_id"] for row in rows}==
            {n["id"] for n in graph["nodes"] if n["op"]=="gelu"},
            "Complete normalized GELU inventory")
    value=dict(format=FORMAT,policy=POLICY,graph_sha256=digest(graph),
               emission_graph_sha256=digest(emissions),entries=rows)
    value["binding"]=digest(value)
    return value

def check_record(value, graph, *, emissions=None, normalized_origins=None,
                 resource_profile="legacy-v1"):
    require(type(value) is dict and set(value)=={
        "format","policy","graph_sha256","emission_graph_sha256","entries","binding"} and
        value["format"]==FORMAT and value["policy"]==POLICY and
        value["binding"]==digest({k:v for k,v in value.items() if k!="binding"}) and
        type(value["entries"]) is list, "GELU source record")
    rows=[]
    for row in value["entries"]:
        require(type(row) is dict and set(row)=={
            "emission_node_id","node_id","variant","source_identity","call_kind","module_path","approximation_sha256"},
            "GELU source row")
        rows.append({k:v for k,v in row.items() if k!="node_id"})
    require(value==record(graph,graph if emissions is None else emissions,rows,
                         normalized_origins=normalized_origins,resource_profile=resource_profile),
            "GELU source record binding")
    return value
