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
FUSED_FORMAT="poseidon-trusted-ordered-fused-matrix-capture-draft-v1"
GELU_FORMAT="poseidon-trusted-ordered-gelu-source-capture-v1"
MHA_FORMAT="poseidon-trusted-ordered-mha-source-capture-v1"
MHA_POLICY="poseidon-mha-ordered-source-v1"
PARTITION_FORMAT="poseidon-trusted-ordered-partition-source-capture-v1"
PARTITION_POLICY="poseidon-pytorch-static-partition-v1"
SCALAR_PARTITION_POLICY="poseidon-pytorch-static-partition-v2"
SCALAR_PARTITION_FORMAT="poseidon-trusted-ordered-scalar-partition-source-capture-v1"
EXACT_ARITHMETIC_POLICY="poseidon-pytorch-exact-arithmetic-v1"
EXACT_ARITHMETIC_FORMAT="poseidon-trusted-ordered-exact-arithmetic-source-capture-v1"
EINSUM_FORMAT="poseidon-trusted-ordered-einsum-source-capture-v1"
EINSUM_POLICY="poseidon-pytorch-static-einsum-v1"
EINSUM_VIEW_POLICY="poseidon-pytorch-static-einsum-v2"
EINSUM_VIEW_FORMAT="poseidon-trusted-ordered-einsum-source-capture-v2"
MHA_IMPLEMENTATION=("pytorch_mha_source.py","pytorch_mha_math.py")
def prepare_module(module,inputs,outputs,*,approximations=None,resource_profile="general-tensors-v1",
                   dtype_policy="strict-float64-v1",matrix_policy="rank-two-three-v1",
                   approximation_policy="operator-defaults-v1",functional_policy=REGISTERED_FUNCTION_POLICY,common_calls=False,fused_matrix_policy=None,
                   gelu_source_policy="exact-gelu-source-v1",mha_source_policy=None,partition_source_policy=None,einsum_source_policy=None,exact_arithmetic_policy=None):
    import torch
    require(type(common_calls) is bool,"Explicit ordered common calls")
    from pytorch_fused_matrix_normalization import POLICY as FUSED_POLICY
    require(fused_matrix_policy is None or fused_matrix_policy == FUSED_POLICY and common_calls is True,
            "Explicit fused matrix capture policy")
    require(gelu_source_policy in ("exact-gelu-source-v1","poseidon-gelu-source-variants-v1"),
            "Explicit ordered GELU source policy")
    require(gelu_source_policy=="exact-gelu-source-v1" or fused_matrix_policy is not None,
            "GELU source capture requires registered ordered fused policy")
    require(mha_source_policy is None or mha_source_policy==MHA_POLICY and
            gelu_source_policy=="poseidon-gelu-source-variants-v1" and common_calls is True and
            fused_matrix_policy is not None,"Explicit registered ordered MHA source policy")
    require(partition_source_policy is None or partition_source_policy in (PARTITION_POLICY,SCALAR_PARTITION_POLICY) and mha_source_policy is not None,
            "Explicit registered ordered partition source policy")
    require(einsum_source_policy is None or einsum_source_policy in (EINSUM_POLICY,EINSUM_VIEW_POLICY) and
            partition_source_policy in (PARTITION_POLICY,SCALAR_PARTITION_POLICY),"Explicit registered ordered einsum source policy")
    require(partition_source_policy!=SCALAR_PARTITION_POLICY or einsum_source_policy==EINSUM_VIEW_POLICY,
            "Scalar partitions require the explicit scalar view contract")
    require(exact_arithmetic_policy is None or exact_arithmetic_policy==EXACT_ARITHMETIC_POLICY and
            partition_source_policy==SCALAR_PARTITION_POLICY,"Explicit registered exact arithmetic source policy")
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
    fx=trace(source,inputs,resource_profile=resource_profile,policy=ALIAS_POLICY,fused_matrix_policy=fused_matrix_policy,
             **({"mha_source_policy":mha_source_policy} if mha_source_policy is not None else {}),
             **({"partition_source_policy":partition_source_policy} if partition_source_policy is not None else {}),
             **({"einsum_source_policy":einsum_source_policy} if einsum_source_policy is not None else {}),
              **({"exact_arithmetic_policy":exact_arithmetic_policy} if exact_arithmetic_policy is not None else {}))
    require(before.keys()==source.state_dict().keys() and
            all(torch.equal(v,source.state_dict()[k]) for k,v in before.items()),
            "State-mutating trace is unsupported")
    source_fx=str(fx.graph)
    exact=None
    if exact_arithmetic_policy is not None:
        from pytorch_exact_arithmetic import normalize as normalize_exact
        fx,exact=normalize_exact(fx,inputs,resource_profile=resource_profile)
    einsum=None
    if einsum_source_policy is not None:
        from pytorch_einsum_normalization import normalize as normalize_einsum
        if einsum_source_policy==EINSUM_VIEW_POLICY:
            from pytorch_einsum_extension import normalize as normalize_einsum
        fx,einsum=normalize_einsum(fx,inputs,resource_profile=resource_profile)
    partition=None
    if partition_source_policy is not None:
        from pytorch_partition_normalization import normalize as normalize_partition
        fx,partition=normalize_partition(fx,inputs,resource_profile=resource_profile,source_policy=partition_source_policy)
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
        approximation_policy=approximation_policy,functional_policy=functional_policy,capture_aliases=True,
        gelu_source_policy=gelu_source_policy,
        **({"einsum_view_policy":einsum_source_policy} if einsum_source_policy==EINSUM_VIEW_POLICY else {}),
        **({"mha_source_policy":mha_source_policy} if mha_source_policy is not None else {}))
    binding["pytorch_capture"]["source_state_dtypes"]=source_dtypes
    arithmetic=bind_record(arithmetic,binding["pytorch_capture"]["ordered_alias_capture"]["capture_emissions"],
                           captured_fx=normalized,public_tensor_views=partition_source_policy==SCALAR_PARTITION_POLICY)
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
        record["format"]=FUSED_FORMAT
        record["fused_matrix_normalization"]=fused
        record["implementation"]["pytorch_fused_matrix_normalization.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_fused_matrix_normalization.py").read_bytes()).hexdigest()
    if gelu_source_policy=="poseidon-gelu-source-variants-v1":
        record["format"]=GELU_FORMAT
        record["gelu_source_policy"]=gelu_source_policy
        record["implementation"]["pytorch_gelu_source.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_gelu_source.py").read_bytes()).hexdigest()
    if mha_source_policy is not None:
        record["format"]=MHA_FORMAT
        record["mha_source_policy"]=mha_source_policy
        for name in MHA_IMPLEMENTATION:
            record["implementation"][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    if partition is not None:
        record["format"]=PARTITION_FORMAT
        record["partition_normalization"]=partition
        record["implementation"]["pytorch_partition_normalization.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_partition_normalization.py").read_bytes()).hexdigest()
    if einsum is not None:
        record["format"]=EINSUM_VIEW_FORMAT if einsum_source_policy==EINSUM_VIEW_POLICY else EINSUM_FORMAT
        record["einsum_normalization"]=einsum
        record["implementation"]["pytorch_einsum_normalization.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_einsum_normalization.py").read_bytes()).hexdigest()
    if einsum_source_policy==EINSUM_VIEW_POLICY:
        record["implementation"]["pytorch_einsum_extension.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_einsum_extension.py").read_bytes()).hexdigest()
    if partition_source_policy==SCALAR_PARTITION_POLICY:record["format"]=SCALAR_PARTITION_FORMAT
    if exact is not None:
        record["format"]=EXACT_ARITHMETIC_FORMAT
        record["exact_arithmetic_normalization"]=exact
        record["implementation"]["pytorch_exact_arithmetic.py"]=hashlib.sha256(
            Path(__file__).with_name("pytorch_exact_arithmetic.py").read_bytes()).hexdigest()
    record["binding"]=digest(record)
    binding["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]=record
    alias=binding["pytorch_capture"]["ordered_alias_capture"]
    alias.pop("binding",None);alias["binding"]=digest(alias)
    binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    context=dict(original_module=source,normalized_module=normalized)
    if mha_source_policy is not None:context["mha_source_policy"]=mha_source_policy
    return high,lowered,binding,context

def capture_module(module,inputs,outputs,**kwargs):
    return prepare_module(module,inputs,outputs,**kwargs)[:3]
