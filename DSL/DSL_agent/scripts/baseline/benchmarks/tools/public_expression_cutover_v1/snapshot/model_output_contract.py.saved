"""Data-only output delivery over the existing Agent task and program interfaces.

This module neither captures source nor compiles or executes a program. The host
must use the existing package checker and execution adapter first. Source/output
identity is separate from the unchanged compilation and numerical identities.
"""
import hashlib,math,re,sys
from pathlib import Path
from benchmark_graph import NAME,canonical,digest,validate,graph_resource_profile
from model_resources import resolve,identity

FORMAT="poseidon-model-output-contract-v1"
DELIVERY="poseidon-output-delivery-v1"
EXECUTION="poseidon-structured-execution-v1"
TREE_DEPTH=32
# A dictionary edge occupies node/items/entry in JSON. Eight additional levels
# cover the fixed contract and delivery envelopes; the semantic tree cap stays 32.
SNAPSHOT_DEPTH=3*TREE_DEPTH+8
TREE_NODES=1024
CONTAINER_ITEMS=64
CONTRACT_BYTES=131072
PACKET_BYTES=4*1024**2
MESSAGES={
    "output_source_rejected":"The output contract must contain registered static data.",
    "output_structure_rejected":"Only bounded Tensor, tuple, list, string-key dictionary and static None structures are registered.",
    "output_signature_rejected":"Named tensor leaves, scalar markers and model output shapes must agree.",
    "output_values_rejected":"Decoded tensors must match the named float64 output signature and contain finite values.",
    "output_binding_rejected":"The output contract must match its trusted binding, source and current implementation.",
    "output_execution_identity_rejected":"The execution result must match the checked program package, request and validation identity.",
}
class OutputError(ValueError):
    def __init__(self,code,position=-1):
        if code not in MESSAGES or type(position) is not int or position < -1:
            raise ValueError("Invalid trusted output rejection")
        self.code,self.position=code,position
        super().__init__(MESSAGES[code])
    def feedback(self):
        return dict(code=self.code,layer="delivery",location=dict(component="output",position=self.position),
            context=dict(constraint=MESSAGES[self.code]),owner="host",retry_policy="never_automatic")
def check(ok,code,position=-1):
    if not ok:raise OutputError(code,position)
def implementation():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def is_hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def snapshot(value,*,limit=CONTRACT_BYTES,code="output_source_rejected"):
    count=0
    def copy(v,depth):
        nonlocal count
        count+=1
        check(depth<=SNAPSHOT_DEPTH and count<=limit//2+64,code)
        if type(v) is dict:
            check(all(type(k) is str for k in v),code)
            return {k:copy(x,depth+1) for k,x in v.items()}
        if type(v) in (list,tuple):return [copy(x,depth+1) for x in v]
        if v is None or type(v) in (str,bool):return v
        if type(v) is int:
            check(v.bit_length()<=256,code);return v
        if type(v) is float:
            check(math.isfinite(v),code);return v
        raise OutputError(code)
    result=copy(value,0)
    try:raw=canonical(result)
    except (ValueError,TypeError,OverflowError):raise OutputError(code)
    check(len(raw)<=limit,code)
    return result

def tensor_leaves(tree):
    leaves=[];count=0
    def visit(node,depth):
        nonlocal count
        count+=1;position=count-1
        check(type(node) is dict and depth<=TREE_DEPTH and count<=TREE_NODES,
              "output_structure_rejected",position)
        kind=node.get("kind")
        if kind=="tensor":
            check(set(node) in ({"kind","name"},{"kind","name","shape"}) and
                  type(node["name"]) is str and NAME.fullmatch(node["name"]) and
                  ("shape" not in node or node["shape"]==[]),"output_structure_rejected",position)
            leaves.append(node);return
        if kind=="none":
            check(set(node)=={"kind"},"output_structure_rejected",position);return
        check(kind in ("tuple","list","dict") and set(node)=={"kind","items"} and
              type(node["items"]) is list and 1<=len(node["items"])<=CONTAINER_ITEMS,
              "output_structure_rejected",position)
        if kind=="dict":
            keys=set()
            for item in node["items"]:
                check(type(item) is dict and set(item)=={"key","value"} and
                      type(item["key"]) is str and 1<=len(item["key"])<=128 and
                      item["key"] not in keys,"output_structure_rejected",position)
                keys.add(item["key"]);visit(item["value"],depth+1)
        else:
            for item in node["items"]:visit(item,depth+1)
    visit(tree,0);return leaves

def capture_tree(value,is_tensor,names,*,scalar_tensor=None):
    """Host-owned callbacks inspect tensors inside the trusted capture worker.

    Returned leaves remain local. Only the data-only tree belongs in provenance.
    No generated or uploaded callable is accepted as a request field.
    """
    check(callable(is_tensor) and (scalar_tensor is None or callable(scalar_tensor)),
          "output_source_rejected")
    check(type(names) is list and 1<=len(names)<=4 and all(type(n) is str and NAME.fullmatch(n) for n in names)
          and len(set(names))==len(names),"output_signature_rejected")
    leaves=[];count=0
    def visit(v,depth):
        nonlocal count
        count+=1;position=count-1
        check(depth<=TREE_DEPTH and count<=TREE_NODES,"output_structure_rejected",position)
        if v is None:return dict(kind="none")
        if type(v) in (tuple,list):
            check(1<=len(v)<=CONTAINER_ITEMS,"output_structure_rejected",position)
            return dict(kind="tuple" if type(v) is tuple else "list",items=[visit(x,depth+1) for x in v])
        if type(v) is dict:
            check(1<=len(v)<=CONTAINER_ITEMS and all(type(k) is str and 1<=len(k)<=128 for k in v),
                  "output_structure_rejected",position)
            return dict(kind="dict",items=[dict(key=k,value=visit(x,depth+1)) for k,x in v.items()])
        check(is_tensor(v) is True and len(leaves)<len(names),"output_structure_rejected",position)
        name=names[len(leaves)];leaves.append(v);node=dict(kind="tensor",name=name)
        if scalar_tensor is not None:
            marker=scalar_tensor(v);check(type(marker) is bool,"output_signature_rejected",position)
            if marker:node["shape"]=[]
        return node
    tree=visit(value,0)
    check(len(leaves)==len(names),"output_signature_rejected")
    tensor_leaves(tree)
    return leaves,tree

def model_signature(model):
    checked_model=snapshot(model,limit=32*1024**2)
    try:
        profile=graph_resource_profile(checked_model);metadata=validate(checked_model)
        signature=[dict(name=o["name"],abi_shape=metadata["shapes"][o["value"]],dtype="float64")
                   for o in checked_model["outputs"]]
        resolve(profile)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError):raise OutputError("output_signature_rejected")
    return checked_model,profile,signature

def build(model,tree,*,source_binding):
    check(is_hash(source_binding),"output_binding_rejected")
    model,profile,signature=model_signature(model)
    tree=snapshot(tree)
    leaves=tensor_leaves(tree);names=[s["name"] for s in signature]
    check([n["name"] for n in leaves]==names,"output_signature_rejected")
    for leaf,s in zip(leaves,signature):
        if "shape" in leaf:check(s["abi_shape"]==[1],"output_signature_rejected")
        s["logical_shape"]=[] if "shape" in leaf else list(s["abi_shape"])
    result=dict(format=FORMAT,model_sha256=digest(model),source_binding=source_binding,
        resource_profile=identity(profile),signature=signature,tree=tree,
        implementation_sha256=implementation())
    result["binding"]=digest(result)
    check(len(canonical(result))<=CONTRACT_BYTES,"output_source_rejected")
    return result

def verify(contract,*,expected_binding,model=None,source_binding=None):
    clean=snapshot(contract,code="output_binding_rejected")
    check(type(clean) is dict and set(clean)=={"format","model_sha256","source_binding","resource_profile",
          "signature","tree","implementation_sha256","binding"} and clean["format"]==FORMAT,"output_binding_rejected")
    check(is_hash(expected_binding) and clean["binding"]==expected_binding and
          digest({k:v for k,v in clean.items() if k!="binding"})==expected_binding and
          clean["implementation_sha256"]==implementation() and is_hash(clean["model_sha256"])
          and is_hash(clean["source_binding"]),"output_binding_rejected")
    if source_binding is not None:check(clean["source_binding"]==source_binding,"output_binding_rejected")
    try:actual_profile=identity(clean["resource_profile"]["name"])
    except (TypeError,ValueError,KeyError):raise OutputError("output_binding_rejected")
    check(canonical(clean["resource_profile"])==canonical(actual_profile),"output_binding_rejected")
    limits=resolve(actual_profile["name"])
    sig=clean["signature"];leaves=tensor_leaves(clean["tree"])
    check(type(sig) is list and 1<=len(sig)<=limits.outputs and len(sig)==len(leaves),"output_signature_rejected")
    names=[]
    for s,leaf in zip(sig,leaves):
        check(type(s) is dict and set(s)=={"name","abi_shape","logical_shape","dtype"} and
              s["dtype"]=="float64" and type(s["name"]) is str and NAME.fullmatch(s["name"]) and
              s["name"]==leaf["name"],"output_signature_rejected")
        shape=s["abi_shape"]
        check(type(shape) is list and 1<=len(shape)<=limits.rank and
              all(type(n) is int and 1<=n<=limits.axis_size for n in shape) and
              math.prod(shape)<=limits.tensor_elements,"output_signature_rejected")
        logical=s["logical_shape"]
        check(type(logical) is list and all(type(n) is int and n>0 for n in logical) and
              logical==([] if "shape" in leaf else shape) and
              ("shape" not in leaf or shape==[1]),"output_signature_rejected")
        names.append(s["name"])
    check(len(set(names))==len(names) and sum(math.prod(s["abi_shape"]) for s in sig)<=limits.output_elements,
          "output_signature_rejected")
    if model is not None:
        rebuilt=build(model,clean["tree"],source_binding=clean["source_binding"])
        check(canonical(rebuilt)==canonical(clean),"output_binding_rejected")
    return clean

def decoded_tensor(value,shape):
    # ndarray is admitted only by exact type; user conversion hooks never run.
    if type(value) is not list:
        np=sys.modules.get("numpy")
        check(np is not None and type(value) is np.ndarray and value.dtype==np.dtype("float64") and
              list(value.shape)==shape,"output_values_rejected")
        check(np.isfinite(value).all(),"output_values_rejected")
        return value.tolist()
    def visit(v,dims):
        if dims:
            check(type(v) is list and len(v)==dims[0],"output_values_rejected")
            return [visit(x,dims[1:]) for x in v]
        check(type(v) in (int,float),"output_values_rejected")
        if type(v) is int:check(-(2**53)<=v<=2**53,"output_values_rejected")
        check(math.isfinite(v),"output_values_rejected")
        return float(v)
    return visit(value,shape)

def restore(contract,values,*,expected_binding,model=None,source_binding=None):
    contract=verify(contract,expected_binding=expected_binding,model=model,source_binding=source_binding)
    names=[s["name"] for s in contract["signature"]]
    check(type(values) is dict and set(values)==set(names),"output_values_rejected")
    decoded={s["name"]:decoded_tensor(values[s["name"]],s["abi_shape"]) for s in contract["signature"]}
    def visit(node):
        if node["kind"]=="none":return None
        if node["kind"]=="tensor":
            value=decoded[node["name"]]
            return value[0] if "shape" in node else value
        if node["kind"]=="dict":return {i["key"]:visit(i["value"]) for i in node["items"]}
        items=[visit(i) for i in node["items"]]
        return tuple(items) if node["kind"]=="tuple" else items
    return visit(contract["tree"])

def checked_validation(value):
    """Exact validation forms returned by the existing program package adapter."""
    validation=snapshot(value,limit=PACKET_BYTES,code="output_execution_identity_rejected")
    check(type(validation) is dict and set(validation)=={"level","encrypted_execution","numerically_validated"} and
          type(validation["level"]) is str and type(validation["encrypted_execution"]) is bool and
          type(validation["numerically_validated"]) is bool and
          validation in (dict(level="compiled",encrypted_execution=False,numerically_validated=False),
                         dict(level="numerical",encrypted_execution=True,numerically_validated=True)),
          "output_execution_identity_rejected")
    return validation

def bind_delivery(contract,*,request_id,bundle_binding,package_validation):
    """Bind identities supplied by the existing checked package adapter.

    This is delivery metadata, not a replacement for package verification.
    """
    contract=verify(contract,expected_binding=contract.get("binding") if type(contract) is dict else None)
    check(is_hash(request_id) and is_hash(bundle_binding),"output_execution_identity_rejected")
    validation=checked_validation(package_validation)
    result=dict(format=DELIVERY,contract=contract,contract_binding=contract["binding"],
        request_id=request_id,bundle_binding=bundle_binding,package_validation=validation)
    result["binding"]=digest(result);return result

def verify_delivery(value,*,expected_binding,model=None,source_binding=None):
    value=snapshot(value,limit=PACKET_BYTES,code="output_binding_rejected")
    check(type(value) is dict and set(value)=={"format","contract","contract_binding","request_id",
          "bundle_binding","package_validation","binding"} and value["format"]==DELIVERY and
          is_hash(expected_binding) and value["binding"]==expected_binding and
          digest({k:v for k,v in value.items() if k!="binding"})==expected_binding,"output_binding_rejected")
    check(is_hash(value["request_id"]) and is_hash(value["bundle_binding"]),"output_execution_identity_rejected")
    checked_validation(value["package_validation"])
    verify(value["contract"],expected_binding=value["contract_binding"],model=model,source_binding=source_binding)
    return value

def structure_execution(delivery,report,*,expected_binding,model=None,source_binding=None):
    """Restore caller-facing structure without changing any validation claim."""
    delivery=verify_delivery(delivery,expected_binding=expected_binding,model=model,source_binding=source_binding)
    original=snapshot(report,limit=PACKET_BYTES,code="output_execution_identity_rejected")
    check(type(original) is dict and original.get("status") in
          ("executed","validation_failed","service_error","cancelled","timed_out"),
          "output_execution_identity_rejected")
    result=dict(format=EXECUTION,status=original["status"],delivery_binding=expected_binding,
        output_contract_binding=delivery["contract_binding"],execution=original)
    if original["status"]=="executed":
        checked_validation(original.get("package_validation"))
        check(original.get("backend") in ("poseidon_cpu","poseidon_cuda") and
              original.get("bundle_binding")==delivery["bundle_binding"] and
              original.get("request_id")==delivery["request_id"] and
              original.get("package_validation")==delivery["package_validation"] and
              type(original.get("encrypted_execution")) is bool and original["encrypted_execution"],
              "output_execution_identity_rejected")
        samples=original.get("samples")
        check(type(samples) is int and 1<=samples<=16 and type(original.get("outputs")) is list and
              len(original["outputs"])==samples,"output_values_rejected")
        result["structured_outputs"]=[restore(delivery["contract"],row,expected_binding=delivery["contract_binding"])
                                      for row in original["outputs"]]
    return result
