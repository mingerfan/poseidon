"""Versioned negative fixtures; validators only, never execute candidate Python."""
import copy,hashlib,json,struct
from benchmark_graph import validate,digest,require
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_request,validate_candidate
from seal_artifact_gate import inspect_artifacts

VERSION="semantic-rejections-v1"
PARTITIONS=("unknown_operator","nonfinite_constant","shape_mismatch","graph_cycle",
    "forward_reference","duplicate_name","private_weight","resource_overflow",
    "candidate_io","candidate_network","ciphertext_condition","unsupported_bootstrap",
    "unverified_upscale","compiler_configuration","request_integrity","layout_binding")
GRAPH=set(PARTITIONS[:8])
CANDIDATE=set(PARTITIONS[8:11])
ARTIFACT=set(PARTITIONS[11:13])
BASE_SOURCE='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n return x\n'

def model(context):
    b=Builder([(2,),(2,2)])
    x=b.node("linear",["input0",b.const([[1.,0.],[0.,1.]]),b.const([0.,0.])])
    if context==1:x=b.node("square",[x])
    elif context==2:x=b.node("subtract",[x,b.node("rotate",[x],step=1)])
    return b.finish(x)

def candidate(request,source=BASE_SOURCE):
    return dict(schema=1,request_id=request["request_id"],hecate_source=source)

def artifact(context):
    opcode=(2,1,8)[context];rhs=(0,1,0)[context];scale=80 if opcode==8 else 40
    body=40+8*(2*3+3)
    raw=struct.pack("<IIQQ",0x4845564D,24,3,1)+struct.pack("<5Q",body,1,4,0,13)
    raw+=struct.pack("<9Q",40,40,40,13,13,13,scale,13,3)
    raw+=struct.pack("<4H",opcode,3,0,rhs)
    return raw,struct.pack("<q",0)

def encoded(value):
    """NaN/Infinity intentionally retained only in negative input identity bytes."""
    if isinstance(value,bytes):return {"hex":value.hex()}
    if isinstance(value,tuple):return [encoded(v) for v in value]
    return value

def identity(value):
    return hashlib.sha256(json.dumps(encoded(value),sort_keys=True,separators=(",",":"),
                                     allow_nan=True).encode()).hexdigest()

def fixture(name,context):
    require(name in PARTITIONS and context in (0,1,2),"Unknown rejection fixture")
    g=model(context);r=prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"))
    if name in GRAPH:
        positive=g;negative=copy.deepcopy(g);layer="model_validation"
        if name=="unknown_operator":
            negative["nodes"][0]["op"]=("unknown","exec","relu")[context];reason="Operator/attributes"
        elif name=="nonfinite_constant":
            negative["constants"]["c0"][0][0]=(float("nan"),float("inf"),-float("inf"))[context]
            reason="Out of range float values"
        elif name=="shape_mismatch":
            negative["constants"]["c0"]=[[1.,0.,1.],[0.,1.,0.]];reason="Linear weight"
        elif name=="graph_cycle":
            negative["nodes"][0]["inputs"][0]=negative["nodes"][-1]["outputs"][0];reason="Undefined/forward input"
        elif name=="forward_reference":
            negative["nodes"].insert(0,dict(id="early",op="negate",inputs=[negative["nodes"][0]["outputs"][0]],attrs={},outputs=["early_value"]))
            reason="Undefined/forward input"
        elif name=="duplicate_name":
            negative["nodes"][0]["outputs"][0]="input0";reason="Invalid/duplicate name"
        elif name=="private_weight":
            negative["nodes"][0]["inputs"][1]="input1";reason="Weights/stats must be public"
        else:
            negative["inputs"][0]["shape"]=[257];reason="Tensor shape budget"
    elif name in CANDIDATE:
        positive=(r,candidate(r));layer="candidate_ast_contract"
        bad={"candidate_io":'open("forbidden-path", "w")',
             "candidate_network":'__import__("socket").socket()',
             "ciphertext_condition":"x if x else zero_ct"}[name]
        source=BASE_SOURCE.replace("return x","return "+bad)
        negative=(r,candidate(r,source));reason={"candidate_io":"Unknown/keyword native loop call",
            "candidate_network":"Unsupported native loop method","ciphertext_condition":"Unsupported native loop syntax"}[name]
    elif name in ARTIFACT:
        positive=artifact(context);raw=bytearray(positive[0])
        struct.pack_into("<H",raw,len(raw)-8,5 if name=="unsupported_bootstrap" else 10)
        negative=(bytes(raw),positive[1]);layer="artifact_preflight"
        reason="Forbidden opcode (bootstrap/upscale/unknown)"
    else:
        positive=r;negative=copy.deepcopy(r);layer="request_binding"
        if name=="compiler_configuration":
            field,value=(("waterline",46),("pipeline","other"),("backend","plaintext"))[context]
            negative["compiler_configuration"][field]=value;reason="Changed compiler configuration"
        elif name=="request_integrity":
            negative["request_id"]="0"*64;reason="Compiler configuration request hash mismatch"
        else:
            negative["layout"]["output_selectors"][0]=[0,1];reason="Changed unified physical layout"
        if name!="request_integrity":
            negative["request_id"]=digest({k:v for k,v in negative.items() if k!="request_id"})
    return dict(name=name,context=context,layer=layer,positive=positive,negative=negative,expected_reason=reason)

def check(value,layer):
    if layer=="model_validation":return validate(value)
    if layer=="candidate_ast_contract":return validate_candidate(value[1],value[0])
    if layer=="request_binding":return validate_request(value)
    if layer=="artifact_preflight":
        from unified_graph_contract import ABI
        return inspect_artifacts(*value,rotation_steps=(1,2),expected_inputs=3,execution_abi=ABI,input_period=4)
    raise ValueError("Unknown rejection layer")

def tasks():
    result=[]
    for name in PARTITIONS:
        for context in range(3):
            f=fixture(name,context)
            row=dict(id="reject_"+name+"_"+str(context),requirement="reject."+name,
                context=context,layer=f["layer"],positive_sha256=identity(f["positive"]),
                negative_sha256=identity(f["negative"]),expected_reason=f["expected_reason"],
                scope="static gate acceptance and rejection; no candidate Python, native code, sandbox escape or ciphertext execution",
                execution_status="not_run")
            row["task_sha256"]=digest(row);result.append(row)
    return result

def run_task(task, known_tasks=None):
    f=fixture(task["requirement"].removeprefix("reject."),task["context"])
    require(task in (tasks() if known_tasks is None else known_tasks),"Frozen rejection task identity")
    check(f["positive"],f["layer"])
    try:check(f["negative"],f["layer"])
    except ValueError as error:
        reason=str(error)
        require(f["expected_reason"] is not None and f["expected_reason"] in reason,
                "Unexpected rejection reason: "+reason)
        return dict(id=task["id"],task_sha256=task["task_sha256"],status="passed",
                    positive_control_accepted=True,negative_rejected=True,reason=reason,layer=f["layer"])
    raise ValueError("Negative fixture unexpectedly accepted")
