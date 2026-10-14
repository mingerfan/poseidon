"""Explicit GELU source-variant draft; not connected to production intake.

The source function and the requested CKKS polynomial are separate identities.
Only Agent-owned graph normalization is added; existing lowerers remain unchanged.
"""
import hashlib
import math
from pathlib import Path

from benchmark_graph import NAME, canonical, digest, dimensions
from model_decomposition import FORMAT as OPERATOR_FORMAT, decompose, polynomial_profile
from model_resources import resolve, identity

FORMAT = "poseidon-gelu-source-plan-draft-v1"
CALL_FORMAT = "poseidon-gelu-source-call-draft-v1"
FUNCTIONS = ("F.gelu", "torch.nn.functional.gelu")
VARIANTS = {"none": "torch-2.0.1-gelu-erf-v1", "tanh": "torch-2.0.1-gelu-tanh-0.044715-v1"}
MESSAGES = {
    "gelu_source_call_rejected": "The call must be a registered GELU function or exact GELU module.",
    "gelu_source_arguments_rejected": "Arguments must match the registered static call signature.",
    "gelu_source_variant_rejected": "The source variant must be explicitly identified as none or tanh.",
    "gelu_source_input_rejected": "The encrypted operand must have float64 name and static shape metadata.",
    "gelu_source_resource_rejected": "The shape must satisfy the selected resource profile.",
    "gelu_source_approximation_rejected": "An explicit bounded polynomial approximation is required.",
    "gelu_source_out_rejected": "Mutation through out is not registered.",
    "gelu_source_graph_rejected": "The polynomial graph must satisfy existing decomposition bounds.",
    "gelu_source_integrity_rejected": "The plan must match the trusted source identity and current implementation.",
    "gelu_source_domain_rejected": "Reference inputs must stay inside the explicitly declared approximation interval.",
}
PARAMETERS = {"call", "input", "input_shape", "resource_profile", "approximate", "approximation", "out", "plan"}

class GeluSourceError(ValueError):
    def __init__(self, code, parameter):
        if code not in MESSAGES or parameter not in PARAMETERS:
            raise ValueError("Invalid trusted GELU rejection")
        self.code, self.parameter = code, parameter
        super().__init__(MESSAGES[code])

    def feedback(self):
        return dict(format="poseidon-gelu-source-rejection-draft-v1", code=self.code,
                    layer="integrity" if self.code=="gelu_source_integrity_rejected" else "input",
                    location=dict(operator="gelu", parameter=self.parameter),
                    context=dict(constraint=MESSAGES[self.code]), retry_policy="never_automatic")

def fail(code, parameter):
    raise GeluSourceError(code, parameter)

def implementation():
    root=Path(__file__).parents[3]
    files=(Path(__file__),root/"benchmark_graph.py",root/"model_decomposition.py",root/"model_resources.py")
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def copy_data(value, limits, *, parameter):
    count=0
    budget=min(limits.constant_elements,limits.total_constant_elements)+4*limits.rank+32
    def walk(v, depth):
        nonlocal count
        count+=1
        if depth>limits.rank+6 or count>budget:
            fail("gelu_source_arguments_rejected",parameter)
        if type(v) is dict:
            if any(type(k) is not str for k in v):
                fail("gelu_source_arguments_rejected",parameter)
            return {k:walk(x,depth+1) for k,x in v.items()}
        if type(v) in (list,tuple):
            return [walk(x,depth+1) for x in v]
        if v is None or type(v) in (str,bool,int,float):
            return v
        fail("gelu_source_arguments_rejected",parameter)
    return walk(value,0)

def plan(operand, *, approximate, approximation, output_name="result",
         resource_profile="general-tensors-v1"):
    try: limits=resolve(resource_profile)
    except (TypeError,ValueError):fail("gelu_source_resource_rejected","resource_profile")
    operand=copy_data(operand,limits,parameter="input")
    if (type(operand) is not dict or set(operand)!={"kind","name","shape","dtype"}
            or operand["kind"]!="cipher" or operand["dtype"]!="float64"
            or type(operand["name"]) is not str or not NAME.fullmatch(operand["name"])
            or type(output_name) is not str or not NAME.fullmatch(output_name)):
        fail("gelu_source_input_rejected","input")
    if type(approximate) is not str or approximate not in VARIANTS:
        fail("gelu_source_variant_rejected","approximate")
    try:
        dims=dimensions(operand["shape"],resource_profile=resource_profile)
        if math.prod(dims)>min(limits.input_elements,limits.output_elements):
            raise ValueError("Logical input/output budget")
    except (TypeError,ValueError,OverflowError):
        fail("gelu_source_resource_rejected","input_shape")
    approximation=copy_data(approximation,limits,parameter="approximation")
    try: polynomial_profile(approximation)
    except (TypeError,ValueError,OverflowError):
        fail("gelu_source_approximation_rejected","approximation")
    used={operand["name"]}
    def fresh(base):
        name=base;i=0
        while name in used:i+=1;name=base+"_"+str(i)
        used.add(name);return name
    coeff=fresh("public_coefficients");value=fresh("approximate_output")
    graph=dict(format=OPERATOR_FORMAT,id="static_gelu_source",
        inputs=[dict(name=operand["name"],shape=list(dims))],
        constants={coeff:approximation["coefficients"]},
        nodes=[dict(id=fresh("polynomial_call"),op="polynomial",inputs=[operand["name"],coeff],
                    attrs=dict(basis=approximation["basis"]),outputs=[value])],
        outputs=[dict(name=output_name,value=value)])
    try: lowered,decomposition=decompose(graph,resource_profile=resource_profile)
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):
        fail("gelu_source_graph_rejected","plan")
    request=dict(operand=operand,approximate=approximate,approximation=approximation,
                 output_name=output_name,resource_profile=resource_profile)
    result=dict(format=FORMAT,source_request=request,source_request_sha256=digest(request),
                source_function=dict(operator="gelu",variant=approximate,identity=VARIANTS[approximate]),
                approximate_mathematical_graph=graph,lowered_graph=lowered,decomposition=decomposition,
                implementation=implementation(),resource_profile=identity(resource_profile),
                original_function_accuracy_certified=False,production_intake_integrated=False,
                physical_layout_status="unverified",compiler_status="unverified",
                encrypted_status="unverified",agent_status="unverified",no_dsl_answer=True)
    result["binding"]=digest(result)
    return result

def source_call(call,args,kwargs, *, approximation, module_variant=None,
                output_name="result",resource_profile="general-tensors-v1"):
    if type(call) is not str or call not in (*FUNCTIONS,"nn.GELU"):
        fail("gelu_source_call_rejected","call")
    if type(args) not in (list,tuple) or len(args)>1 or type(kwargs) is not dict:
        fail("gelu_source_arguments_rejected","call")
    names={"input","approximate","out"} if call in FUNCTIONS else {"input"}
    if any(type(k) is not str or k not in names for k in kwargs):
        fail("gelu_source_arguments_rejected","call")
    values=dict(kwargs)
    if args:
        if "input" in values:fail("gelu_source_arguments_rejected","call")
        values["input"]=args[0]
    if "input" not in values:fail("gelu_source_arguments_rejected","input")
    if call=="nn.GELU":
        variant=module_variant
    else:
        if module_variant is not None:fail("gelu_source_arguments_rejected","call")
        variant=values.get("approximate","none")
        if values.get("out") is not None:fail("gelu_source_out_rejected","out")
    result=plan(values["input"],approximate=variant,approximation=approximation,
                output_name=output_name,resource_profile=resource_profile)
    limits=resolve(resource_profile)
    request=dict(call=call,args=copy_data(args,limits,parameter="call"),
                 kwargs=copy_data(kwargs,limits,parameter="call"),
                 module_variant=module_variant,approximation=copy_data(approximation,limits,parameter="approximation"),
                 output_name=output_name,resource_profile=resource_profile)
    envelope=dict(format=CALL_FORMAT,source_call=request,source_call_sha256=digest(request),plan=result)
    envelope["binding"]=digest(envelope)
    return envelope

def verify_plan(value, *, expected_source_request_sha256):
    try:
        if type(value) is not dict or value.get("format")!=FORMAT:
            raise ValueError("Plan fields")
        if value["binding"]!=digest({k:v for k,v in value.items() if k!="binding"}):
            raise ValueError("Plan binding")
        if value["implementation"]!=implementation():
            raise ValueError("Implementation binding")
        if value["source_request_sha256"]!=digest(value["source_request"]) or value["source_request_sha256"]!=expected_source_request_sha256:
            raise ValueError("Source binding")
        if canonical(plan(**value["source_request"]))!=canonical(value):
            raise ValueError("Source semantics")
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):
        fail("gelu_source_integrity_rejected","plan")
    return value

def verify_call(value, *, expected_source_call_sha256):
    try:
        if type(value) is not dict or set(value)!={"format","source_call","source_call_sha256","plan","binding"} or value["format"]!=CALL_FORMAT:
            raise ValueError("Call fields")
        if value["binding"]!=digest({k:v for k,v in value.items() if k!="binding"}):
            raise ValueError("Call binding")
        if value["source_call_sha256"]!=digest(value["source_call"]) or value["source_call_sha256"]!=expected_source_call_sha256:
            raise ValueError("Trusted call binding")
        if canonical(source_call(**value["source_call"]))!=canonical(value):
            raise ValueError("Call semantics")
        verify_plan(value["plan"],expected_source_request_sha256=value["plan"]["source_request_sha256"])
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):
        fail("gelu_source_integrity_rejected","plan")
    return value

def original_scalar(x,variant):
    if type(variant) is not str or variant not in VARIANTS:
        fail("gelu_source_variant_rejected","approximate")
    if variant=="none":
        return .5*x*(1+math.erf(x/math.sqrt(2)))
    return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))

def check_domain(values, approximation):
    polynomial_profile(approximation)
    lo,hi=approximation["domain"]
    if any(type(x) not in (int,float) or not math.isfinite(x) or not lo<=x<=hi for x in values):
        fail("gelu_source_domain_rejected","input")

def from_fx_call(node, operands, *, module=None, approximation,
                 output_name="result",resource_profile="general-tensors-v1"):
    import torch
    from torch.nn import functional as F
    from torch.fx.immutable_collections import immutable_list
    if type(node) is not torch.fx.Node or type(operands) is not dict:
        fail("gelu_source_call_rejected","call")
    if node.op=="call_function" and node.target is F.gelu and module is None:
        call="F.gelu";variant=None
    elif node.op=="call_module" and type(module) is torch.nn.GELU:
        call="nn.GELU";variant=module.approximate
    else:
        fail("gelu_source_call_rejected","call")
    try:limits=resolve(resource_profile)
    except (TypeError,ValueError):fail("gelu_source_resource_rejected","resource_profile")
    def resolve_operand(value):
        if type(value) is torch.fx.Node:
            if value not in operands:fail("gelu_source_input_rejected","input")
            return copy_data(operands[value],limits,parameter="input")
        if type(value) in (list,tuple,immutable_list):
            return [resolve_operand(x) for x in value]
        if value is None or type(value) in (str,bool,int,float):
            return value
        fail("gelu_source_arguments_rejected","call")
    return source_call(call,[resolve_operand(x) for x in node.args],
        {k:resolve_operand(v) for k,v in node.kwargs.items()},module_variant=variant,
        approximation=approximation,output_name=output_name,resource_profile=resource_profile)
