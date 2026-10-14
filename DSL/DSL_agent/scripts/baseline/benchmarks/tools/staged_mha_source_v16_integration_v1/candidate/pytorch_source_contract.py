"""Versioned trusted-source parameters, float policy and static output trees."""
import math
import re
from benchmark_graph import require, canonical, NAME
LEGACY = "poseidon-trusted-pytorch-v1"
FORMAT = "poseidon-trusted-pytorch-v2"
WEIGHT_FORMAT = "poseidon-trusted-pytorch-v3"
STATIC_FORMAT = "poseidon-trusted-pytorch-v4"
APPROXIMATION_FORMAT = "poseidon-trusted-pytorch-v5"
INDEX_FORMAT = "poseidon-trusted-pytorch-v6"
MATRIX_FORMAT = "poseidon-trusted-pytorch-v7"
INDEX_FORMATS = (INDEX_FORMAT,MATRIX_FORMAT)
SCOPED_FORMATS = (APPROXIMATION_FORMAT,*INDEX_FORMATS)
STATIC_FORMATS = (STATIC_FORMAT,*SCOPED_FORMATS)
STATE_FORMATS = (WEIGHT_FORMAT,*STATIC_FORMATS)
EXTENDED = (FORMAT,*STATE_FORMATS)
FLOAT_POLICIES = ("strict-float64-v1", "convert-float64-v1")
COMMON = {"format", "files", "entry", "inputs", "outputs", "approximations"}

def validate_manifest(spec):
    require(type(spec) is dict, "Trusted PyTorch manifest")
    version=spec.get("format")
    require(version in (LEGACY,*EXTENDED), "Trusted PyTorch format")
    expected=COMMON if version==LEGACY else COMMON|{"factory","dtype_policy"}
    if version in STATE_FORMATS:expected|={"weights"}
    require(set(spec)==expected, "Trusted PyTorch manifest fields")
    require(type(spec["inputs"]) is list and type(spec["outputs"]) is list
            and type(spec["approximations"]) is dict, "Trusted signature/approximations")
    if version in SCOPED_FORMATS:
        from pytorch_approximation import validate
        validate(spec["approximations"])
    if version in EXTENDED:
        require(spec["dtype_policy"] in FLOAT_POLICIES, "Explicit float policy")
        factory=spec["factory"]
        require(type(factory) is dict and set(factory)=={"args","kwargs"}
                and type(factory["args"]) is list and type(factory["kwargs"]) is dict,
                "Factory parameters")
        count=0
        def visit(value,depth=0):
            nonlocal count
            count+=1
            require(depth<=8 and count<=1024,"Factory parameter structure budget")
            if type(value) is bool or value is None: return
            if type(value) is int:
                require(abs(value)<=2**53,"Factory integer precision");return
            if type(value) is float:
                require(math.isfinite(value),"Nonfinite factory parameter");return
            if type(value) is str:
                require(len(value)<=256,"Factory string budget");return
            if type(value) in (list,dict):
                require(len(value)<=64,"Factory collection budget")
                if type(value) is dict:
                    require(all(type(k) is str and len(k)<=256 for k in value),"Factory mapping keys")
                    for item in value.values():visit(item,depth+1)
                else:
                    for item in value:visit(item,depth+1)
                return
            raise ValueError("Only data values in factory parameters")
        require(all(type(k) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",k)
                    for k in factory["kwargs"]),"Factory keyword names")
        visit(factory)
        require(len(canonical(factory))<=8192,"Factory parameter byte budget")
    if version in STATE_FORMATS:
        from pytorch_weight_package import declarations
        # The actual selected resource profile is checked by the admission host.
        declarations(spec["weights"],resource_profile="general-tensors-v1")
    return spec

def factory_parameters(spec):
    validate_manifest(spec)
    return spec.get("factory",dict(args=[],kwargs={}))

def output_tree(value,is_tensor,names,*,scalar_tensor=None):
    """Flatten only tensors and bounded static containers, preserving order."""
    require(type(names) is list and 1<=len(names)<=4 and
            all(type(n) is str and NAME.fullmatch(n) for n in names)
            and len(set(names))==len(names),"Named output signature")
    leaves=[]
    def visit(item,depth=0):
        require(depth<=8,"Output structure depth")
        if is_tensor(item):
            require(len(leaves)<len(names),"Output leaf count")
            name=names[len(leaves)];leaves.append(item)
            return dict(kind="tensor",name=name,**({"shape":[]} if scalar_tensor is not None and scalar_tensor(item) else {}))
        if isinstance(item,(tuple,list)):
            require(1<=len(item)<=4,"Static output sequence")
            return dict(kind="tuple" if isinstance(item,tuple) else "list",
                        items=[visit(x,depth+1) for x in item])
        if isinstance(item,dict):
            require(1<=len(item)<=4 and
                    all(type(k) is str and 0<len(k)<=128 for k in item),
                    "String-key output dictionary")
            return dict(kind="dict",items=[dict(key=k,value=visit(v,depth+1)) for k,v in item.items()])
        raise ValueError("Output leaves must be tensors")
    tree=visit(value)
    require(len(leaves)==len(names),"Output leaf count")
    return leaves,tree

def validate_output_tree(tree,names,*,allow_scalars=False,source_format=None):
    if source_format=="poseidon-trusted-pytorch-v16":
        from pytorch_mha_outputs import validate
        validate(tree)
        require(tree["source_format"]==source_format and tree["tensor_names"]==names,
                "Versioned MHA output signature")
        return tree
    collected=[]
    def visit(node,depth=0):
        require(type(node) is dict and depth<=8,"Output tree node")
        kind=node.get("kind")
        if kind=="tensor":
            require((set(node)=={"kind","name"} or allow_scalars and
                     set(node)=={"kind","name","shape"} and node["shape"]==[])
                    and type(node["name"]) is str,
                    "Output tree tensor")
            collected.append(node["name"]);return
        require(kind in ("tuple","list","dict") and set(node)=={"kind","items"}
                and type(node["items"]) is list and 1<=len(node["items"])<=4,
                "Output tree container")
        if kind=="dict":
            keys=set()
            for item in node["items"]:
                require(type(item) is dict and set(item)=={"key","value"}
                        and type(item["key"]) is str and 0<len(item["key"])<=128
                        and item["key"] not in keys,"Output tree dictionary key")
                keys.add(item["key"]);visit(item["value"],depth+1)
        else:
            for item in node["items"]:visit(item,depth+1)
    visit(tree)
    require(collected==names,"Output tree signature")
    return tree

def restore_outputs(values,tree):
    if type(tree) is dict and tree.get("format")=="poseidon-static-output-tree-v16-draft-v1":
        from pytorch_mha_outputs import restore,SOURCE_FORMAT
        require(tree.get("source_format")==SOURCE_FORMAT,"Explicit MHA output restoration version")
        return restore(values,tree)
    validate_output_tree(tree,list(values),allow_scalars=True)
    def visit(node):
        if node["kind"]=="tensor":
            value=values[node["name"]]
            if "shape" in node:
                import numpy as np
                array=np.asarray(value)
                require(array.shape==(1,), "Scalar output ABI shape")
                return array.reshape(()).tolist()
            return value
        if node["kind"]=="dict":
            return {item["key"]:visit(item["value"]) for item in node["items"]}
        items=[visit(item) for item in node["items"]]
        return tuple(items) if node["kind"]=="tuple" else items
    return visit(tree)
