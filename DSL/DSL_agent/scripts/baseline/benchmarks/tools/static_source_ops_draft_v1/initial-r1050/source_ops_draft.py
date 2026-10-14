"""Agent-owned static source-operation draft, separate from production intake.

Builds mathematical graphs and bindings. Does not generate DSL or execute code.
The existing decomposition/reference/backend implementations remain unchanged.
"""
import copy,hashlib,math
from pathlib import Path
from benchmark_graph import NAME,canonical,digest,dimensions,shape,broadcast,validate
from model_resources import resolve,identity
from model_decomposition import FORMAT as OPERATOR_FORMAT,decompose,polynomial_profile

VERSION="poseidon-static-source-op-plan-draft-v1"
DIVISION_CALLS=("operator.truediv","torch.div","torch.divide","torch.true_divide",
                "Tensor.div","Tensor.divide","Tensor.true_divide")
CONSTRAINTS={
"source_call_unregistered":"The call must be an explicitly registered static source operation.",
"source_shape_rejected":"The input shape must satisfy the selected resource profile.",
"source_name_rejected":"Input and output names must follow the registered identifier grammar.",
"division_rounding_unsupported":"Only true division with rounding_mode=None is registered.",
"source_out_mutation_unsupported":"Source out mutation is not registered.",
"division_public_type_rejected":"The divisor must be a bounded real scalar or rectangular public array.",
"division_public_zero":"Every public divisor element must be nonzero.",
"division_public_nonfinite":"Every public divisor element must be finite.",
"division_public_budget":"Public values and their reciprocals must satisfy existing constant bounds.",
"division_broadcast_rejected":"The encrypted input and public divisor must broadcast within the resource profile.",
"pool_public_parameters_rejected":"Pooling parameters must be static integers or rank-matched integer sequences.",
"pool_rank_rejected":"The existing pool contract requires N,C and one or two spatial dimensions.",
"pool_ceil_unsupported":"Only floor-mode pooling is registered.",
"pool_indices_unsupported":"Only pool values are registered; encrypted argmax indices are not supported.",
"pool_approximation_rejected":"An explicit valid absolute-value polynomial profile is required.",
"source_graph_contract_rejected":"The normalized graph must satisfy existing shape and expansion bounds.",
"source_plan_integrity_rejected":"The plan must match its trusted source request and current implementation."
}
PARAMETERS={"call","input_shape","names","rounding_mode","out","divisor","kernel_size","stride",
            "padding","dilation","ceil_mode","return_indices","approximation","plan","resource_profile"}

class SourcePlanError(ValueError):
    def __init__(self,code,operator,parameter):
        if code not in CONSTRAINTS or operator not in {"public_division","max_pool1d","max_pool2d","plan"} or parameter not in PARAMETERS:
            raise ValueError("Invalid trusted source rejection")
        self.code,self.operator,self.parameter=code,operator,parameter
        super().__init__(CONSTRAINTS[code])
    def feedback(self):
        return dict(format="poseidon-static-source-op-rejection-draft-v1",code=self.code,
            layer="integrity" if self.code=="source_plan_integrity_rejected" else "input",
            location=dict(operator=self.operator,parameter=self.parameter),
            context=dict(constraint=CONSTRAINTS[self.code]),retry_policy="never_automatic")
def fail(code,operator,parameter):raise SourcePlanError(code,operator,parameter)

def implementation():
    paths=[Path(__file__),Path(__file__).parents[3]/"benchmark_graph.py",
           Path(__file__).parents[3]/"model_decomposition.py",Path(__file__).parents[3]/"model_resources.py"]
    # The source lives in baseline/benchmarks/tools/<draft>/.
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def _profile(name,op):
    try:return resolve(name)
    except (TypeError,ValueError):fail("source_shape_rejected",op,"resource_profile")
def _input_shape(value,profile,op):
    limits=_profile(profile,op)
    try:
        dims=dimensions(value,resource_profile=profile)
        if math.prod(dims)>limits.input_elements:raise ValueError("Total input budget")
    except (TypeError,ValueError,OverflowError):fail("source_shape_rejected",op,"input_shape")
    return dims
def _names(input_name,output_name,op):
    if not all(type(n) is str and NAME.fullmatch(n) for n in (input_name,output_name)):
        fail("source_name_rejected",op,"names")
    used={input_name}
    def fresh(base):
        n=base;i=0
        while n in used:i+=1;n=base+"_"+str(i)
        used.add(n);return n
    return fresh
def _inverse(value,profile):
    limits=resolve(profile);seen=0
    def walk(v,depth):
        nonlocal seen
        if type(v) is list:
            if not v or depth>=limits.rank:fail("division_public_type_rejected","public_division","divisor")
            return [walk(x,depth+1) for x in v]
        if type(v) not in (int,float):fail("division_public_type_rejected","public_division","divisor")
        seen+=1
        if seen>min(limits.constant_elements,limits.total_constant_elements):fail("division_public_budget","public_division","divisor")
        if type(v) is float and not math.isfinite(v):fail("division_public_nonfinite","public_division","divisor")
        if abs(v)>1024:fail("division_public_budget","public_division","divisor")
        if v==0:fail("division_public_zero","public_division","divisor")
        reciprocal=1./v
        if not math.isfinite(reciprocal) or abs(reciprocal)>1024:
            fail("division_public_budget","public_division","divisor")
        return reciprocal
    result=walk(value,0)
    try:shape(value,resource_profile=profile);shape(result,resource_profile=profile)
    except (TypeError,ValueError,OverflowError):fail("division_public_type_rejected","public_division","divisor")
    return result

def _finish(request,graph,op):
    try:lowered,decomposition=decompose(graph,resource_profile=request["resource_profile"])
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):
        fail("source_graph_contract_rejected",op,"plan")
    result=dict(format=VERSION,source_request=copy.deepcopy(request),source_request_sha256=digest(request),
        mathematical_graph=graph,lowered_graph=lowered,decomposition=decomposition,
        implementation=implementation(),resource_profile=identity(request["resource_profile"]),
        physical_layout_status="unverified",compiler_status="unverified",encrypted_status="unverified",
        agent_status="unverified",production_intake_integrated=False,no_dsl_answer=True)
    result["binding"]=digest(result)
    return result

def public_division(input_shape,divisor,*,call="torch.div",rounding_mode=None,out=None,
                    input_name="x",output_name="result",resource_profile="general-tensors-v1"):
    op="public_division"
    if type(call) is not str or call not in DIVISION_CALLS:fail("source_call_unregistered",op,"call")
    if rounding_mode is not None:fail("division_rounding_unsupported",op,"rounding_mode")
    if out is not None:fail("source_out_mutation_unsupported",op,"out")
    dims=_input_shape(input_shape,resource_profile,op);fresh=_names(input_name,output_name,op)
    factor=_inverse(divisor,resource_profile)
    try:
        result=broadcast(dims,shape(factor,resource_profile=resource_profile))
        dimensions(list(result),resource_profile=resource_profile)
        if math.prod(result)>resolve(resource_profile).output_elements:raise ValueError("Output budget")
    except (TypeError,ValueError,OverflowError):fail("division_broadcast_rejected",op,"divisor")
    weight=fresh("public_factor");value=fresh("normalized_output");node=fresh("normalized_division")
    graph=dict(format=OPERATOR_FORMAT,id="static_source_operation",
        inputs=[dict(name=input_name,shape=list(dims))],constants={weight:factor},
        nodes=[dict(id=node,op="multiply",inputs=[input_name,weight],attrs={},outputs=[value])],
        outputs=[dict(name=output_name,value=value)])
    request=dict(operation=op,call=call,input_shape=list(dims),divisor=copy.deepcopy(divisor),
        rounding_mode=None,out=None,input_name=input_name,output_name=output_name,resource_profile=resource_profile)
    return _finish(request,graph,op)

def _spatial(value,rank,min_value,profile,op,parameter):
    if type(value) is int:value=[value]*rank
    elif type(value) in (list,tuple):value=list(value)
    else:fail("pool_public_parameters_rejected",op,parameter)
    if len(value)!=rank or any(type(v) is not int or not min_value<=v<=resolve(profile).axis_size for v in value):
        fail("pool_public_parameters_rejected",op,parameter)
    return value

def functional_max_pool(input_shape,kernel_size,*,rank=2,stride=None,padding=0,dilation=1,
                        ceil_mode=False,return_indices=False,approximation=None,
                        input_name="x",output_name="result",resource_profile="general-tensors-v1"):
    if type(rank) is not int or rank not in (1,2):fail("pool_rank_rejected","plan","input_shape")
    op="max_pool"+str(rank)+"d"
    dims=_input_shape(input_shape,resource_profile,op);fresh=_names(input_name,output_name,op)
    if len(dims)!=rank+2:fail("pool_rank_rejected",op,"input_shape")
    if type(ceil_mode) is not bool or ceil_mode:fail("pool_ceil_unsupported",op,"ceil_mode")
    if type(return_indices) is not bool or return_indices:fail("pool_indices_unsupported",op,"return_indices")
    kernel=_spatial(kernel_size,rank,1,resource_profile,op,"kernel_size")
    step=list(kernel) if stride is None else _spatial(stride,rank,1,resource_profile,op,"stride")
    pad=_spatial(padding,rank,0,resource_profile,op,"padding")
    dil=_spatial(dilation,rank,1,resource_profile,op,"dilation")
    try:polynomial_profile(approximation)
    except (TypeError,ValueError,OverflowError):fail("pool_approximation_rejected",op,"approximation")
    attrs=dict(kernel=kernel,stride=step,padding=pad,dilation=dil,approximation=copy.deepcopy(approximation))
    value=fresh("normalized_output");node=fresh("normalized_pool")
    graph=dict(format=OPERATOR_FORMAT,id="static_source_operation",
        inputs=[dict(name=input_name,shape=list(dims))],constants={},
        nodes=[dict(id=node,op=op,inputs=[input_name],attrs=attrs,outputs=[value])],
        outputs=[dict(name=output_name,value=value)])
    request=dict(operation="functional_max_pool",rank=rank,input_shape=list(dims),
        kernel_size=kernel,stride=step,padding=pad,dilation=dil,ceil_mode=False,return_indices=False,
        approximation=copy.deepcopy(approximation),input_name=input_name,output_name=output_name,resource_profile=resource_profile)
    return _finish(request,graph,op)

def verify_plan(plan,*,expected_source_request_sha256):
    try:
        if type(plan) is not dict or plan.get("format")!=VERSION or type(expected_source_request_sha256) is not str:
            raise ValueError("Plan fields")
        content={k:v for k,v in plan.items() if k!="binding"}
        if plan.get("binding")!=digest(content) or plan.get("implementation")!=implementation():
            raise ValueError("Plan identity")
        request=plan["source_request"]
        if plan["source_request_sha256"]!=digest(request) or plan["source_request_sha256"]!=expected_source_request_sha256:
            raise ValueError("Source identity")
        kwargs={k:v for k,v in request.items() if k!="operation"}
        rebuilt=public_division(**kwargs) if request["operation"]=="public_division" else (
            functional_max_pool(**kwargs) if request["operation"]=="functional_max_pool" else None)
        if rebuilt is None or canonical(rebuilt)!=canonical(plan):raise ValueError("Plan semantics")
    except (TypeError,ValueError,KeyError,IndexError,OverflowError):
        fail("source_plan_integrity_rejected","plan","plan")
    return plan
