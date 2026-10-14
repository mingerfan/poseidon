"""Deterministic, explicit approximation selection for trusted PyTorch capture."""
import copy
import re
from benchmark_graph import require, canonical, digest, NAME
from model_decomposition import profile, polynomial_profile

FORMAT = "poseidon-pytorch-approximations-v1"
REPORT = "poseidon-pytorch-approximation-selection-v1"
LEGACY_POLICY = "operator-defaults-v1"
POLICY = "node-module-default-v1"
FUNCTIONS = ("relu","gelu","silu","sigmoid","tanh","log","abs","exp","reciprocal","rsqrt")
FX_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MODULE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*")

def validate(value):
    require(type(value) is dict and set(value)=={"format","defaults","modules","nodes"}
            and value["format"]==FORMAT, "Approximation selection fields/version")
    require(len(canonical(value))<=131072, "Approximation selection byte budget")
    defaults=value["defaults"]
    require(type(defaults) is dict and set(defaults)<=set(FUNCTIONS), "Approximation default function")
    for name,item in defaults.items(): _profile(name,item)
    require(type(value["nodes"]) is list and type(value["modules"]) is list and
            len(value["nodes"])+len(value["modules"])<=128, "Approximation selector budget")
    for kind,key in (("nodes","name"),("modules","path")):
        seen=set()
        for item in value[kind]:
            require(type(item) is dict and set(item)=={key,"function","profile"},
                    "Approximation selector fields")
            label=item[key];function=item["function"]
            require(type(label) is str and len(label)<=256 and
                    (FX_NAME.fullmatch(label) if key=="name" else MODULE.fullmatch(label)),
                    "Approximation selector location")
            require(function in FUNCTIONS, "Approximation selector function")
            require((label,function) not in seen, "Conflicting approximation selectors")
            seen.add((label,function));_profile(function,item["profile"])
    return value

def _profile(function,value):
    checked=profile(function,value) if function in ("exp","reciprocal","rsqrt") else polynomial_profile(value)
    if function=="log":require(checked["domain"][0]>0,"Log domain must be positive")
    return checked

def module_paths(node):
    paths=[]
    stack=node.meta.get("nn_module_stack",{})
    require(isinstance(stack,dict), "FX module source metadata")
    # Pinned Torch 2.0.1 records OrderedDict[path, module_type].
    for path,module_type in stack.items():
        require(type(path) is str and isinstance(module_type,type), "FX module source metadata")
        if path and path not in paths:paths.append(path)
    if node.op=="call_module" and node.target not in paths:paths.append(node.target)
    _paths(paths)
    return paths

def _paths(paths):
    require(type(paths) is list and len(paths)<=64 and
            all(type(p) is str and len(p)<=256 and MODULE.fullmatch(p) for p in paths) and
            len(paths)==len(set(paths)),
            "Approximation module source paths")

class Resolver:
    def __init__(self,configuration):
        self.configuration=copy.deepcopy(validate(configuration))
        self.matched={"defaults":set(),"nodes":set(),"modules":set()}
        self.selections=[]
    def select(self,node,operator,function,paths):
        require(type(node) is str and FX_NAME.fullmatch(node) and type(operator) is str,
                "Approximation node source")
        require(function in FUNCTIONS,"Approximation function")
        _paths(paths)
        config=self.configuration
        found={}
        for kind,key,values in (("nodes","name",[node]),("modules","path",paths)):
            matches=[(i,item) for i,item in enumerate(config[kind])
                     if item[key] in values and item["function"]==function]
            require(len(matches)<=1,"Conflicting approximation selectors at "+node+" for "+function)
            for i,item in matches:self.matched[kind].add(i)
            if matches:
                index,item=matches[0]
                found[kind]=(item["profile"],dict(kind=kind,selector=index))
        if function in config["defaults"]:
            self.matched["defaults"].add(function)
            found["defaults"]=(config["defaults"][function],dict(kind="defaults",selector=function))
        require(found,"Explicit approximation required for "+function+" at "+node)
        selected=next(found[k] for k in ("nodes","modules","defaults") if k in found)
        value,source=selected
        self.selections.append(dict(node_id="fx_"+node,operator=operator,function=function,
            module_paths=copy.deepcopy(paths),source=source,profile_sha256=digest(value)))
        return copy.deepcopy(value)
    def finish(self):
        for kind in ("nodes","modules"):
            require(self.matched[kind]==set(range(len(self.configuration[kind]))),
                    "Unmatched approximation "+kind+" selector")
        require(self.matched["defaults"]==set(self.configuration["defaults"]),
                "Unmatched approximation default function")
        return dict(format=REPORT,policy=POLICY,configuration_sha256=digest(self.configuration),
                    selections=copy.deepcopy(self.selections))

def _components(node):
    op=node["op"];attrs=node["attrs"]
    if op in ("relu","gelu","silu","sigmoid","tanh","log","exp","reciprocal","rsqrt"):
        return [(op,attrs["approximation"])]
    if op in ("leaky_relu","prelu"):return [("relu",attrs["approximation"])]
    if op in ("layer_norm","group_norm","instance_norm","rms_norm","sqrt"):
        return [("rsqrt",attrs["approximation"])]
    if op in ("maximum","minimum","max","min","max_pool1d","max_pool2d"):
        return [("abs",attrs["approximation"])]
    if op=="softmax":return [(n,attrs[n]) for n in ("exp","reciprocal")]
    if op=="logsoftmax":return [(n,attrs[n]) for n in ("exp","log")]
    return []

def check_record(record,original,configuration):
    require(type(record) is dict and set(record)=={"format","policy","configuration_sha256","selections"}
            and type(record["selections"]) is list,"Approximation selection metadata")
    resolver=Resolver(configuration);rows=iter(record["selections"])
    for node in original["nodes"]:
        for function,expected in _components(node):
            row=next(rows,None)
            require(type(row) is dict and set(row)=={"node_id","operator","function","module_paths","source","profile_sha256"}
                    and row["node_id"]==node["id"] and row["operator"]==node["op"] and row["function"]==function,
                    "Approximation selection graph binding")
            require(node["id"].startswith("fx_"),"Approximation FX node identity")
            selected=resolver.select(node["id"][3:],node["op"],function,row["module_paths"])
            require(canonical(selected)==canonical(expected),"Selected approximation differs from graph")
    require(next(rows,None) is None,"Extra approximation selection")
    require(record==resolver.finish(),"Approximation selection record binding")
    return record
