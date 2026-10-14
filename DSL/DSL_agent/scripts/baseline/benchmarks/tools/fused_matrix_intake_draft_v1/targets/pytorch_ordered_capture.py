"""Trusted staging entry for registered capture with ordered tensor effects.

Call only inside the explicitly trusted local-model capture sandbox. Source
loading, factory arguments and weight-package I/O belong to the retained intake.
This adapter has no Provider, backend, shell, package loading or DSL-answer path.
"""
import copy,hashlib,math
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_resources import resolve
from pytorch_static_trace import trace,ALIAS_POLICY,INDEX_POLICY
from pytorch_arithmetic_normalization import normalize,bind_record,normalize_common,bind_common_record
from pytorch_model_capture import capture,REGISTERED_FUNCTION_POLICY
FORMAT="poseidon-trusted-ordered-capture-staging-v2"
COMMON_FORMAT="poseidon-trusted-ordered-common-capture-draft-v1"
def prepare_module(module,inputs,outputs,*,approximations=None,resource_profile="general-tensors-v1",
                   dtype_policy="strict-float64-v1",matrix_policy="rank-two-three-v1",
                   approximation_policy="operator-defaults-v1",functional_policy=REGISTERED_FUNCTION_POLICY,common_calls=False,fused_matrix_policy=None):
    import torch
    require(type(common_calls) is bool,"Explicit ordered common calls")
    from pytorch_fused_matrix_normalization import POLICY as FUSED_POLICY
    require(fused_matrix_policy is None or fused_matrix_policy == FUSED_POLICY and common_calls is True,
            "Explicit fused matrix capture policy")
    limits=resolve(resource_profile)
    require(isinstance(module,torch.nn.Module),"Explicitly trusted PyTorch module required")
    require(type(inputs) is list and 1<=len(inputs)<=limits.inputs,"Input count")
    total=0
    for spec in inputs:
        require(type(spec) is dict and set(spec)=={"name","shape"} and type(spec["name"]) is str,
                "Declared ordered capture signature")
        shape=spec["shape"]
        require(type(shape) is list and 1<=len(shape)<=limits.rank and
                all(type(n) is int and 0<n<=limits.axis_size for n in shape),"Input shape")
        total+=math.prod(shape)
    require(total<=limits.input_elements,"Input element budget")
    require(type(outputs) is list and 1<=len(outputs)<=limits.outputs,"Output count")
    from pytorch_source_contract import FLOAT_POLICIES
    require(dtype_policy=="legacy-float64-v1" or dtype_policy in FLOAT_POLICIES,"Float policy")
    source_dtypes={name:str(value.dtype).removeprefix("torch.") for name,value in module.state_dict().items()}
    if dtype_policy!="legacy-float64-v1":
        for value in module.state_dict().values():
            if value.is_floating_point():
                require(value.dtype in (torch.float32,torch.float64),"Supported source floating dtype")
                if dtype_policy=="strict-float64-v1":require(value.dtype is torch.float64,"Strict float64 source policy")
            else:require(value.dtype in (torch.int32,torch.int64,torch.bool),"Supported public dtype")
    # Work on a separate copy before tracing as well as ShapeProp. Public state
    # mutation is rejected, never absorbed into a replacement weight binding.
    source=copy.deepcopy(module).cpu().double().eval()
    before={k:v.detach().clone() for k,v in source.state_dict().items()}
    fx=trace(source,inputs,resource_profile=resource_profile,policy=ALIAS_POLICY,fused_matrix_policy=fused_matrix_policy)
    require(before.keys()==source.state_dict().keys() and
            all(torch.equal(v,source.state_dict()[k]) for k,v in before.items()),
            "State-mutating trace is unsupported")
    source_fx=str(fx.graph)
    fused=None
    if fused_matrix_policy is not None:
        from pytorch_fused_matrix_normalization import normalize as normalize_fused,check_record
        traced=fx
        fx,fused=normalize_fused(fx,inputs,resource_profile=resource_profile)
        check_record(fused,traced,fx)
    common=None
    if common_calls:
        fx,common=normalize_common(fx,inputs,resource_profile=resource_profile,ordered_effects=True)
    normalized,arithmetic=normalize(fx,resource_profile=resource_profile,
                                   preserve_unused_state=True,allow_ordered_mutations=True)
    require(len(source_fx.encode())<=limits.json_bytes and
            len(str(normalized.graph).encode())<=limits.json_bytes,"Ordered FX text budget")
    high,lowered,binding=capture(normalized,inputs,outputs,approximations=approximations,
        resource_profile=resource_profile,dtype_policy=dtype_policy,preserve_output_structure=True,
        record_state_sources=True,capture_policy=INDEX_POLICY,matrix_policy=matrix_policy,
        approximation_policy=approximation_policy,functional_policy=functional_policy,capture_aliases=True)
    binding["pytorch_capture"]["source_state_dtypes"]=source_dtypes
    arithmetic=bind_record(arithmetic,binding["pytorch_capture"]["ordered_alias_capture"]["capture_emissions"],
                           captured_fx=normalized)
    record=dict(format=COMMON_FORMAT if common_calls else FORMAT,source_state_dtypes=source_dtypes,source_fx_text=source_fx,source_fx_sha256=hashlib.sha256(source_fx.encode()).hexdigest(),
        normalized_fx_text=str(normalized.graph),normalized_fx_sha256=hashlib.sha256(str(normalized.graph).encode()).hexdigest(),
        arithmetic=arithmetic,resource_profile=resource_profile,graph_sha256=digest(high),
        independent_reference_required=True,reference_inputs="separate clones for every execution",
        production_intake_integrated=False,compiler_validated=False,encrypted_execution_validated=False,
        implementation={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name("pytorch_model_capture.py"),
             Path(__file__).with_name("pytorch_static_trace.py"),Path(__file__).with_name("pytorch_alias_capture_bridge.py"),
             Path(__file__).with_name("pytorch_alias_composition.py"),Path(__file__).with_name("pytorch_arithmetic_normalization.py"))})
    if common_calls:
        alias=binding["pytorch_capture"]["ordered_alias_capture"]
        record["common_normalization"]=bind_common_record(common,alias["capture_emissions"],arithmetic,
                                          alias["capture_emission_state_constant_sources"])
    if fused is not None:
        record["format"]="poseidon-trusted-ordered-fused-matrix-capture-draft-v1"
        record["fused_matrix_normalization"]=fused
        record["implementation"]["pytorch_fused_matrix_normalization.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_fused_matrix_normalization.py").read_bytes()).hexdigest()
    record["binding"]=digest(record)
    binding["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]=record
    alias=binding["pytorch_capture"]["ordered_alias_capture"]
    alias.pop("binding",None);alias["binding"]=digest(alias)
    binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    return high,lowered,binding,dict(original_module=source,normalized_module=normalized)

def capture_module(module,inputs,outputs,**kwargs):
    return prepare_module(module,inputs,outputs,**kwargs)[:3]
