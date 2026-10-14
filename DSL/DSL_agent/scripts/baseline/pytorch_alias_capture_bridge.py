"""Agent-owned ordered effects over the existing registered capture emissions."""
import copy,hashlib,operator
from pathlib import Path
from benchmark_graph import require,digest,canonical,NAME
from model_resources import resolve
from pytorch_alias_composition import SOURCE_FORMAT,SCALAR_SOURCE_FORMAT,plan

FORMAT="poseidon-pytorch-ordered-alias-capture-staging-v2"
SCALAR_FORMAT="poseidon-pytorch-ordered-alias-capture-v3"

def basic_index(value):
    if type(value) in (int,slice) or value is None or value is Ellipsis:return True
    return type(value) is tuple and all(basic_index(x) for x in value)

def origin(node,item,traced):
    import torch
    target=node.target
    result=dict(fx_node=node.name,fx_kind=node.op,operation=item["op"],action=None)
    op=item["op"]
    if op=="reshape":
        result["action"]="view" if target=="view" else "reshape"
    elif op in ("transpose","permute"):result["action"]=op
    elif op=="broadcast":result["action"]="expand"
    elif op=="slice":result["action"]="axis_slice"
    elif op=="dropout":
        result["action"]="clone" if target=="clone" else "contiguous" if target=="contiguous" else "alias"
    elif op=="public_gather" and (
          target in ("select","narrow",torch.select,torch.narrow) or
          target is operator.getitem and len(node.args)==2 and basic_index(node.args[1])):
        result["action"]="view_gather"
    mutable=False
    if node.op=="call_module":
        child=traced.get_submodule(target)
        mutable=bool(getattr(child,"inplace",False)) and op in ("relu","silu","leaky_relu")
    elif node.op=="call_function":
        mutable=(target is torch.relu_ or target in (torch.nn.functional.relu,
                 torch.nn.functional.silu,torch.nn.functional.leaky_relu) and
                 (node.kwargs.get("inplace") is True or
                  len(node.args)>(2 if target is torch.nn.functional.leaky_relu else 1) and
                  node.args[2 if target is torch.nn.functional.leaky_relu else 1] is True))
    if mutable:
        result["mutation_target"]=item["inputs"][0]
    return result

def normalize_emissions(graph,origins,*,resource_profile,constant_dtypes,state_constant_sources,public_constant_views=None,scalar_views=False,spatial_source_policy=None):
    """Only emitted registered operators enter this bridge; no source callbacks."""
    require(type(scalar_views)is bool,"Explicit scalar alias policy")
    limits=resolve(resource_profile)
    import numpy as np
    public_constant_views={} if public_constant_views is None else public_constant_views
    require(type(public_constant_views) is dict and len(public_constant_views)<=limits.constants,
            "Public emission view inventory")
    for name,view in public_constant_views.items():
        require(type(view) is dict and set(view)=={"source","source_shape","result_shape","indices"}
                and name in graph["constants"] and view["source"] in graph["constants"],
                "Public emission view source")
        raw=np.asarray(graph["constants"][view["source"]],dtype="<f8")
        indices=np.asarray(view["indices"])
        require(indices.dtype.kind in "iu" and list(raw.shape)==view["source_shape"]
                and list(indices.shape)==view["result_shape"] and 0<indices.size<=limits.constant_elements
                and bool((indices>=0).all() and (indices<raw.size).all()),"Public emission view indices")
        require(canonical(raw.ravel()[indices].tolist())==canonical(graph["constants"][name]),
                "Public emission view differs from source")
    require(type(graph) is dict and type(origins) is dict,"Capture emission stream")
    require(set(origins)=={n["id"] for n in graph["nodes"]},"Every emission must have source provenance")
    source=dict(format=SCALAR_SOURCE_FORMAT if scalar_views else SOURCE_FORMAT,inputs=copy.deepcopy(graph["inputs"]),
                constants=copy.deepcopy(graph["constants"]),nodes=[],outputs=copy.deepcopy(graph["outputs"]))
    used={i["name"] for i in graph["inputs"]}|set(graph["constants"])
    used.update(v for n in graph["nodes"] for v in n["outputs"])
    used.update(n["id"] for n in graph["nodes"])
    counter=0
    def fresh():
        nonlocal counter
        while True:
            counter+=1;name="capture_alias_"+str(counter)
            if name not in used:used.add(name);return name
    node_map={}
    def add(identifier,op,inputs,attrs,output,fx_origin):
        require(NAME.fullmatch(identifier) and NAME.fullmatch(output),"Generated capture identifiers")
        require(len(source["nodes"])<limits.original_nodes,"Ordered capture node budget")
        source["nodes"].append(dict(id=identifier,op=op,inputs=inputs,attrs=attrs,output=output))
        node_map[identifier]=copy.deepcopy(fx_origin)
    for item in graph["nodes"]:
        annotation=origins[item["id"]]
        op=item["op"];refs=list(item["inputs"]);attrs=copy.deepcopy(item["attrs"])
        if op=="split":
            require(set(attrs)=={"axis","sections"} and len(item["outputs"])==len(attrs["sections"]),"Split capture emissions")
            start=0
            for size,output in zip(attrs["sections"],item["outputs"]):
                add(fresh(),"axis_slice",refs,dict(axis=attrs["axis"],start=start,stop=start+size,step=1),output,annotation)
                start+=size
            continue
        require(len(item["outputs"])==1,"Single registered result required")
        output=item["outputs"][0]
        scalar_result=scalar_views and annotation.get("logical_shape")==[]
        result=fresh() if "mutation_target" in annotation or scalar_result else output
        action=annotation.get("action")
        if action=="diagonal":
            require(scalar_views and set(annotation.get("diagonal",{}))=={"offset","dim1","dim2"},"Generated diagonal provenance")
            add(item["id"],"diagonal",refs,copy.deepcopy(annotation["diagonal"]),result,annotation)
        elif action in ("alias","clone","contiguous"):
            add(item["id"],action,refs,{},result,annotation)
        elif action in ("view","reshape"):
            require(set(attrs)=={"shape"},"Reshape capture emission")
            add(item["id"],action,refs,dict(shape=attrs["shape"] if scalar_views else attrs["shape"] or [1]),result,annotation)
        elif action in ("permute","transpose","expand","axis_slice","view_gather"):
            if action=="expand":attrs=dict(shape=attrs["shape"])
            add(item["id"],action,refs,attrs,result,annotation)
        elif action=="copy":
            add(item["id"],"copy",refs,{},result,annotation)
        elif op in ("add","subtract","multiply"):
            require(not attrs and len(refs)==2,"Arithmetic capture emission")
            add(item["id"],op,refs,dict(inplace=False,alpha=1),result,annotation)
        elif op=="square":
            require(not attrs,"Square capture emission")
            add(item["id"],"square",refs,{},result,annotation)
        else:
            add(item["id"],"registered",refs,dict(operator=op,parameters=attrs),result,annotation)
        if scalar_result:
            logical=fresh() if "mutation_target" in annotation else output
            add(fresh(),"view",[result],dict(shape=[]),logical,annotation)
            result=logical
        if "mutation_target" in annotation:
            update=[annotation["mutation_target"],result]
            if "mutation_other" in annotation:update.append(annotation["mutation_other"])
            add(fresh(),"assign",update,{},output,annotation)
    value=plan(source,{},resource_profile=resource_profile,
        **({'spatial_source_policy':spatial_source_policy} if spatial_source_policy is not None else {}))
    state_sources={};derived_state_sources={};dtypes={}
    for name,record in value["public_constant_sources"].items():
        original=record["source"]
        if original in constant_dtypes:dtypes[name]=constant_dtypes[original]
        if original in state_constant_sources:
            if record["full_identity"]:state_sources[name]=state_constant_sources[original]
            else:derived_state_sources[name]=dict(state=state_constant_sources[original],view=record)
    metadata=dict(format=SCALAR_FORMAT if scalar_views else FORMAT,source_graph=source,source_graph_sha256=digest(source),
        capture_emissions=copy.deepcopy(graph),capture_emissions_sha256=digest(graph),
        capture_emission_constant_dtypes=copy.deepcopy(constant_dtypes),
        capture_emission_state_constant_sources=copy.deepcopy(state_constant_sources),
        capture_public_constant_views=copy.deepcopy(public_constant_views),
        capture_source_origins=copy.deepcopy(origins),
        source_node_origins=node_map,normalization=value,
        constant_dtypes=dtypes,state_constant_sources=state_sources,
        derived_state_constant_sources=derived_state_sources,
        implementation={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
          (Path(__file__),Path(__file__).with_name("pytorch_model_capture.py"),
           Path(__file__).with_name("pytorch_alias_composition.py"))},
        input_storage_assumption="separate contiguous CPU float64 inputs",
        production_intake_integrated=False,paid_calls=0,physical_layout_validated=False,
        compiler_validated=False,encrypted_execution_validated=False)
    metadata["binding"]=digest(metadata)
    return value["mathematical_graph"],value["lowered_graph"],copy.deepcopy(value["decomposition"]),metadata
