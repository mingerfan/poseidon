from pytorch_logical_view_source import SOURCE_FORMAT as VIEW_SOURCE_FORMAT, PACKAGE_FORMAT as VIEW_PACKAGE_FORMAT, REFERENCE_FORMAT as VIEW_REFERENCE_FORMAT
from pytorch_public_join_source import SOURCE_FORMAT as JOIN_SOURCE_FORMAT, PACKAGE_FORMAT as JOIN_PACKAGE_FORMAT, REFERENCE_FORMAT as JOIN_REFERENCE_FORMAT
from pytorch_public_mask_source import SOURCE_FORMAT as MASK_SOURCE_FORMAT, PACKAGE_FORMAT as MASK_PACKAGE_FORMAT, REFERENCE_FORMAT as MASK_REFERENCE_FORMAT
"""Versioned trusted PyTorch intake facade.

V1-V7 delegate to the retained capture implementation. V8 adds deterministic
arithmetic normalization; V9 opts into static metadata for arithmetic results.
V10 additionally registers operator result metadata and functional adaptive
average pooling. V11 adds registered common calls and public SDPA masks.
All retain the same isolated CPU capture boundary.
It never executes model source in the host and never calls a Provider.
"""
import copy,hashlib,json,math,os,re,signal,subprocess,sys
from pathlib import Path
from benchmark_graph import require,digest,canonical
from model_weight_package import regular_bytes
import trusted_pytorch_import as legacy
from pytorch_spatial_policy import SOURCE_FORMAT as SPATIAL_FORMAT, PACKAGE_FORMAT as SPATIAL_PACKAGE_FORMAT, REFERENCE_FORMAT as SPATIAL_REFERENCE_FORMAT
from pytorch_static_broadcast import SOURCE_FORMAT as BROADCAST_FORMAT, PACKAGE_FORMAT as BROADCAST_PACKAGE_FORMAT, REFERENCE_FORMAT as BROADCAST_REFERENCE_FORMAT
from pytorch_source_package_policy import SOURCE_FORMAT as SOURCE_PACKAGE_FORMAT, PACKAGE_FORMAT as SOURCE_PACKAGE_PROVENANCE, REFERENCE_FORMAT as SOURCE_PACKAGE_REFERENCE
from pytorch_static_tensor_transforms import SOURCE_FORMAT as TENSOR_SOURCE_FORMAT, PACKAGE_FORMAT as TENSOR_PACKAGE_FORMAT, REFERENCE_FORMAT as TENSOR_REFERENCE_FORMAT
from pytorch_public_constant_source import SOURCE_FORMAT as CONSTANT_SOURCE_FORMAT, PACKAGE_FORMAT as CONSTANT_PACKAGE_FORMAT, REFERENCE_FORMAT as CONSTANT_REFERENCE_FORMAT
from pytorch_public_tensor_transform_source import SOURCE_FORMAT as STATE_TRANSFORM_SOURCE_FORMAT, PACKAGE_FORMAT as STATE_TRANSFORM_PACKAGE_FORMAT, REFERENCE_FORMAT as STATE_TRANSFORM_REFERENCE_FORMAT
ORDERED_FORMAT="poseidon-trusted-pytorch-v12"
ORDERED_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v12"
ORDERED_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v12"
PARTITION_FORMAT="poseidon-trusted-pytorch-v17"
PARTITION_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v17"
PARTITION_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v17"
SCALAR_PARTITION_FORMAT="poseidon-trusted-pytorch-v20"
SCALAR_PARTITION_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v20"
SCALAR_PARTITION_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v20"
PUBLIC_TENSOR_FORMAT="poseidon-trusted-pytorch-v22"
PUBLIC_TENSOR_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v22"
PUBLIC_TENSOR_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v22"
PUBLIC_EXPRESSION_FORMAT="poseidon-trusted-pytorch-v23"
PUBLIC_EXPRESSION_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v23"
PUBLIC_EXPRESSION_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v23"
PUBLIC_DTYPE_FORMAT="poseidon-trusted-pytorch-v24"
PUBLIC_DTYPE_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v24"
PUBLIC_DTYPE_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v24"
INPUT_BINDING_FORMAT="poseidon-trusted-pytorch-v26"
INPUT_BINDING_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v26"
INPUT_BINDING_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v26"
PUBLIC_ARITHMETIC_FORMAT="poseidon-trusted-pytorch-v25"
PUBLIC_ARITHMETIC_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v25"
PUBLIC_ARITHMETIC_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v25"
EXACT_ARITHMETIC_FORMAT="poseidon-trusted-pytorch-v21"
EXACT_ARITHMETIC_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v21"
EXACT_ARITHMETIC_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v21"
EINSUM_VIEW_FORMAT="poseidon-trusted-pytorch-v19"
EINSUM_VIEW_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v19"
EINSUM_VIEW_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v19"
EINSUM_FORMAT="poseidon-trusted-pytorch-v18"
EINSUM_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v18"
EINSUM_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v18"
MHA_FORMAT="poseidon-trusted-pytorch-v16"
MHA_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v16"
MHA_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v16"
GELU_FORMAT="poseidon-trusted-pytorch-v15"
GELU_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v15"
GELU_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v15"
ORDERED_COMMON_FORMAT="poseidon-trusted-pytorch-v13"
ORDERED_COMMON_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v13"
ORDERED_COMMON_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v13"
ORDERED_FUSED_FORMAT="poseidon-trusted-pytorch-v14"
ORDERED_FUSED_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v14"
ORDERED_FUSED_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v14"
def ordered(spec):
    return type(spec) is dict and spec.get("format") in (ORDERED_FORMAT,ORDERED_COMMON_FORMAT,ORDERED_FUSED_FORMAT,GELU_FORMAT,MHA_FORMAT,PARTITION_FORMAT,EINSUM_FORMAT,EINSUM_VIEW_FORMAT,SCALAR_PARTITION_FORMAT,EXACT_ARITHMETIC_FORMAT,PUBLIC_TENSOR_FORMAT,PUBLIC_EXPRESSION_FORMAT,PUBLIC_DTYPE_FORMAT,PUBLIC_ARITHMETIC_FORMAT,INPUT_BINDING_FORMAT,SPATIAL_FORMAT,BROADCAST_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT,JOIN_SOURCE_FORMAT,VIEW_SOURCE_FORMAT)

FORMAT="poseidon-trusted-pytorch-v8"
PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v8"
REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v8"
SHAPE_FORMAT="poseidon-trusted-pytorch-v9"
SHAPE_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v9"
SHAPE_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v9"
OPERATOR_SHAPE_FORMAT="poseidon-trusted-pytorch-v10"
OPERATOR_SHAPE_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v10"
OPERATOR_SHAPE_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v10"
COMMON_FORMAT="poseidon-trusted-pytorch-v11"
COMMON_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v11"
COMMON_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v11"
SOURCE_FORMATS=(FORMAT,SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT,COMMON_FORMAT,ORDERED_FORMAT,ORDERED_COMMON_FORMAT,ORDERED_FUSED_FORMAT,GELU_FORMAT,MHA_FORMAT,PARTITION_FORMAT,EINSUM_FORMAT,EINSUM_VIEW_FORMAT,SCALAR_PARTITION_FORMAT,EXACT_ARITHMETIC_FORMAT,PUBLIC_TENSOR_FORMAT,PUBLIC_EXPRESSION_FORMAT,PUBLIC_DTYPE_FORMAT,PUBLIC_ARITHMETIC_FORMAT,INPUT_BINDING_FORMAT,SPATIAL_FORMAT,BROADCAST_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT,JOIN_SOURCE_FORMAT,VIEW_SOURCE_FORMAT)
SOURCE_SHAPE_FORMATS=(SHAPE_FORMAT,OPERATOR_SHAPE_FORMAT,COMMON_FORMAT)
PACKAGES={FORMAT:PACKAGE_FORMAT,SHAPE_FORMAT:SHAPE_PACKAGE_FORMAT,OPERATOR_SHAPE_FORMAT:OPERATOR_SHAPE_PACKAGE_FORMAT,COMMON_FORMAT:COMMON_PACKAGE_FORMAT,ORDERED_FORMAT:ORDERED_PACKAGE_FORMAT,ORDERED_COMMON_FORMAT:ORDERED_COMMON_PACKAGE_FORMAT,ORDERED_FUSED_FORMAT:ORDERED_FUSED_PACKAGE_FORMAT,GELU_FORMAT:GELU_PACKAGE_FORMAT,MHA_FORMAT:MHA_PACKAGE_FORMAT,PARTITION_FORMAT:PARTITION_PACKAGE_FORMAT,EINSUM_FORMAT:EINSUM_PACKAGE_FORMAT,EINSUM_VIEW_FORMAT:EINSUM_VIEW_PACKAGE_FORMAT,SCALAR_PARTITION_FORMAT:SCALAR_PARTITION_PACKAGE_FORMAT,EXACT_ARITHMETIC_FORMAT:EXACT_ARITHMETIC_PACKAGE_FORMAT,PUBLIC_TENSOR_FORMAT:PUBLIC_TENSOR_PACKAGE_FORMAT,PUBLIC_EXPRESSION_FORMAT:PUBLIC_EXPRESSION_PACKAGE_FORMAT,PUBLIC_DTYPE_FORMAT:PUBLIC_DTYPE_PACKAGE_FORMAT,PUBLIC_ARITHMETIC_FORMAT:PUBLIC_ARITHMETIC_PACKAGE_FORMAT,INPUT_BINDING_FORMAT:INPUT_BINDING_PACKAGE_FORMAT,SPATIAL_FORMAT:SPATIAL_PACKAGE_FORMAT,BROADCAST_FORMAT:BROADCAST_PACKAGE_FORMAT,SOURCE_PACKAGE_FORMAT:SOURCE_PACKAGE_PROVENANCE,TENSOR_SOURCE_FORMAT:TENSOR_PACKAGE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT:STATE_TRANSFORM_PACKAGE_FORMAT,CONSTANT_SOURCE_FORMAT:CONSTANT_PACKAGE_FORMAT,MASK_SOURCE_FORMAT:MASK_PACKAGE_FORMAT,JOIN_SOURCE_FORMAT:JOIN_PACKAGE_FORMAT,VIEW_SOURCE_FORMAT:VIEW_PACKAGE_FORMAT}
REFERENCES={FORMAT:REFERENCE_FORMAT,SHAPE_FORMAT:SHAPE_REFERENCE_FORMAT,OPERATOR_SHAPE_FORMAT:OPERATOR_SHAPE_REFERENCE_FORMAT,COMMON_FORMAT:COMMON_REFERENCE_FORMAT,ORDERED_FORMAT:ORDERED_REFERENCE_FORMAT,ORDERED_COMMON_FORMAT:ORDERED_COMMON_REFERENCE_FORMAT,ORDERED_FUSED_FORMAT:ORDERED_FUSED_REFERENCE_FORMAT,GELU_FORMAT:GELU_REFERENCE_FORMAT,MHA_FORMAT:MHA_REFERENCE_FORMAT,PARTITION_FORMAT:PARTITION_REFERENCE_FORMAT,EINSUM_FORMAT:EINSUM_REFERENCE_FORMAT,EINSUM_VIEW_FORMAT:EINSUM_VIEW_REFERENCE_FORMAT,SCALAR_PARTITION_FORMAT:SCALAR_PARTITION_REFERENCE_FORMAT,EXACT_ARITHMETIC_FORMAT:EXACT_ARITHMETIC_REFERENCE_FORMAT,PUBLIC_TENSOR_FORMAT:PUBLIC_TENSOR_REFERENCE_FORMAT,PUBLIC_EXPRESSION_FORMAT:PUBLIC_EXPRESSION_REFERENCE_FORMAT,PUBLIC_DTYPE_FORMAT:PUBLIC_DTYPE_REFERENCE_FORMAT,PUBLIC_ARITHMETIC_FORMAT:PUBLIC_ARITHMETIC_REFERENCE_FORMAT,INPUT_BINDING_FORMAT:INPUT_BINDING_REFERENCE_FORMAT,SPATIAL_FORMAT:SPATIAL_REFERENCE_FORMAT,BROADCAST_FORMAT:BROADCAST_REFERENCE_FORMAT,SOURCE_PACKAGE_FORMAT:SOURCE_PACKAGE_REFERENCE,TENSOR_SOURCE_FORMAT:TENSOR_REFERENCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT:STATE_TRANSFORM_REFERENCE_FORMAT,CONSTANT_SOURCE_FORMAT:CONSTANT_REFERENCE_FORMAT,MASK_SOURCE_FORMAT:MASK_REFERENCE_FORMAT,JOIN_SOURCE_FORMAT:JOIN_REFERENCE_FORMAT,VIEW_SOURCE_FORMAT:VIEW_REFERENCE_FORMAT}
EXTRA_IMPLEMENTATION=("pytorch_model_intake.py","pytorch_arithmetic_normalization.py")
IMPLEMENTATION=(*legacy.IMPLEMENTATION,*EXTRA_IMPLEMENTATION)

def shim(spec):
    if ordered(spec):
        from pytorch_ordered_contract import shim as selected
        return selected(spec)
    require(type(spec) is dict and spec.get("format") in SOURCE_FORMATS,"Trusted arithmetic source version")
    require(len(canonical(spec))<=131072,"Trusted manifest data/budget")
    result=copy.deepcopy(spec);result["format"]=legacy.SOURCE_V7
    return result

def source_shape_policy(spec):
    if ordered(spec):
        from pytorch_static_trace import ALIAS_POLICY
        return ALIAS_POLICY
    from pytorch_static_trace import INDEX_POLICY,ARITHMETIC_POLICY,OPERATOR_POLICY,COMMON_POLICY
    require(type(spec) is dict and spec.get("format") in SOURCE_FORMATS,"Trusted source shape version")
    return {FORMAT:INDEX_POLICY,SHAPE_FORMAT:ARITHMETIC_POLICY,OPERATOR_SHAPE_FORMAT:OPERATOR_POLICY,COMMON_FORMAT:COMMON_POLICY}[spec["format"]]

def source_identity():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in IMPLEMENTATION}

def package_format(spec):
    if ordered(spec):
        from pytorch_ordered_intake import package_format as selected
        return selected(spec)
    if type(spec) is dict and spec.get("format") in SOURCE_FORMATS:
        legacy.validate_manifest(shim(spec));return PACKAGES[spec["format"]]
    return legacy.package_format(spec)

def validate_package(spec,files):
    if ordered(spec):
        from pytorch_ordered_contract import validate_package as selected
        return selected(spec,files)
    if type(spec) is dict and spec.get("format") in SOURCE_FORMATS:
        legacy.validate_package(shim(spec),files)
        require(not set(files).intersection(IMPLEMENTATION),"Source module conflicts with trusted adapter")
        return spec
    return legacy.validate_package(spec,files)

def prepare(manifest_path):
    from strict_json_files import strict_file
    if ordered(strict_file(Path(manifest_path),131072)):
        from pytorch_ordered_contract import prepare as selected
        return selected(manifest_path)
    from strict_json_files import strict_file
    path=Path(manifest_path);spec=strict_file(path,131072)
    if spec.get("format") not in SOURCE_FORMATS:return legacy.prepare(path)
    converted=shim(spec);legacy.validate_manifest(converted)
    require(type(spec["files"]) is dict and 1<=len(spec["files"])<=8,"Source inventory")
    files={};paths={};total=0
    for name,expected in spec["files"].items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py",name) and
                name not in IMPLEMENTATION,"Flat source filename")
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}",expected),"Source hash")
        raw=regular_bytes(path.parent/name,65536-total);total+=len(raw)
        files[name]=raw.decode();paths[name]=(path.parent/name).resolve()
    validate_package(spec,files)
    return spec,paths

def preparation_options(provenance,model,options=None):
    if provenance.get("trusted_pytorch",{}).get("format") in (ORDERED_PACKAGE_FORMAT,ORDERED_COMMON_PACKAGE_FORMAT,ORDERED_FUSED_PACKAGE_FORMAT,GELU_PACKAGE_FORMAT,MHA_PACKAGE_FORMAT,PARTITION_PACKAGE_FORMAT,EINSUM_PACKAGE_FORMAT):
        from pytorch_ordered_intake import preparation_options as selected
        return selected(provenance,model,options)
    if provenance.get("trusted_pytorch",{}).get("format") in PACKAGES.values():
        require(options is None or type(options) is dict,"Component option fields")
        result=copy.deepcopy(options or {})
        if model.get("format")=="poseidon-model-graph-v2" and result.get("capacity_packing_version") is None:
            from unified_capacity_constants import INDEX_VERSION
            result["capacity_packing_version"]=INDEX_VERSION
        return result
    return legacy.preparation_options(provenance,model,options)

def check_capture_metadata(metadata,original,spec,*,resource_profile="legacy-v1",expected_weight_binding=None,expected_weight_arrays=None):
    if ordered(spec):
        from pytorch_ordered_contract import check_capture_metadata as selected
        return selected(metadata,original,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    require(expected_weight_arrays is None,"Public array payloads belong to explicit V22")
    require(expected_weight_binding is None,"Weight payload binding belongs to explicit V12")
    if spec.get("format") not in SOURCE_FORMATS:
        return legacy.check_capture_metadata(metadata,original,spec,resource_profile=resource_profile)
    from pytorch_arithmetic_normalization import check_record
    require(type(metadata) is dict and "arithmetic_capture" in metadata,"Arithmetic capture metadata")
    clean=copy.deepcopy(metadata);record=clean.pop("arithmetic_capture")
    if spec["format"] in SOURCE_SHAPE_FORMATS:
        from pytorch_static_trace import policy_record
        source_shape=clean.pop("source_shape_capture",None)
        require(type(source_shape) is dict and canonical(source_shape)==canonical(
            policy_record(spec["inputs"],resource_profile,policy=source_shape_policy(spec))),
            "Arithmetic source shape policy binding")
    require(record.get("resource_profile")==resource_profile,"Arithmetic resource binding")
    check_record(record,original)
    if spec["format"]==COMMON_FORMAT:
        from pytorch_arithmetic_normalization import check_common_record
        common=clean.pop("common_capture",None)
        check_common_record(common,original,record)
    legacy.check_capture_metadata(clean,original,shim(spec),resource_profile=resource_profile)
    return metadata

def capture_binding(original,spec,*,resource_profile="legacy-v1",capture_metadata=None,expected_weight_binding=None,expected_weight_arrays=None):
    if ordered(spec):
        from pytorch_ordered_contract import capture_binding as selected
        return selected(original,spec,resource_profile=resource_profile,capture_metadata=capture_metadata,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    require(expected_weight_arrays is None,"Public array payloads belong to explicit V22")
    require(expected_weight_binding is None,"Weight payload binding belongs to explicit V12")
    if spec.get("format") not in SOURCE_FORMATS:
        return legacy.capture_binding(original,spec,resource_profile=resource_profile,capture_metadata=capture_metadata)
    check_capture_metadata(capture_metadata,original,spec,resource_profile=resource_profile)
    clean=copy.deepcopy(capture_metadata);clean.pop("arithmetic_capture")
    if spec["format"] in SOURCE_SHAPE_FORMATS:clean.pop("source_shape_capture")
    if spec["format"]==COMMON_FORMAT:clean.pop("common_capture")
    lowered,binding=legacy.capture_binding(original,shim(spec),resource_profile=resource_profile,capture_metadata=clean)
    binding.update(manifest_sha256=digest(spec),capture_implementation=source_identity(),
                   pytorch_capture=capture_metadata,capture_policy_version=spec["format"])
    binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    return lowered,binding

def check_reference(report,original,lowered,spec,*,resource_profile="legacy-v1",expected_weight_binding=None,expected_weight_arrays=None):
    if ordered(spec):
        from pytorch_ordered_contract import check_reference as selected
        return selected(report,original,lowered,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    require(expected_weight_arrays is None,"Public array payloads belong to explicit V22")
    require(expected_weight_binding is None,"Weight payload binding belongs to explicit V12")
    if spec.get("format") not in SOURCE_FORMATS:
        return legacy.check_reference(report,original,lowered,spec,resource_profile=resource_profile)
    require(type(report) is dict and report.get("format")==REFERENCES[spec["format"]] and
            report.get("source_hashes")==source_identity() and report.get("manifest_sha256")==digest(spec),
            "Arithmetic reference source identity")
    require(report.get("normalized_module_checked") is True and
            report.get("arithmetic_semantics_preserved") is True,"Original/normalized module comparison")
    for key in ("max_arithmetic_normalization_error","max_arithmetic_normalization_ratio"):
        value=report.get(key);require(type(value) in (int,float) and math.isfinite(value) and value>=0,
                                     "Arithmetic reference metric")
    require(report["max_arithmetic_normalization_ratio"]<=1,"Arithmetic normalization differs from original module")
    check_capture_metadata(report.get("capture_metadata"),original,spec,resource_profile=resource_profile)
    if spec["format"]==COMMON_FORMAT:
        require(report.get("common_semantics_preserved") is True,"Common source normalization comparison")
    projected=copy.deepcopy(report)
    projected.update(format=legacy.REFERENCE_V7,manifest_sha256=digest(shim(spec)),source_hashes=legacy.source_identity())
    projected["capture_metadata"].pop("arithmetic_capture")
    if spec["format"] in SOURCE_SHAPE_FORMATS:projected["capture_metadata"].pop("source_shape_capture")
    if spec["format"]==COMMON_FORMAT:projected["capture_metadata"].pop("common_capture")
    legacy.check_reference(projected,original,lowered,shim(spec),resource_profile=resource_profile)
    return report

def run(manifest_path,output,*,resource_profile="legacy-v1",trusted=False):
    from strict_json_files import strict_file
    if ordered(strict_file(Path(manifest_path),131072)):
        from pytorch_ordered_intake import run as selected
        return selected(manifest_path,output,resource_profile=resource_profile,trusted=trusted)
    spec,paths=prepare(manifest_path)
    if spec.get("format") not in SOURCE_FORMATS:
        return legacy.run(manifest_path,output,resource_profile=resource_profile,trusted=trusted)
    require(trusted is True,"Explicit --trust-local-code required")
    from hecate_python_env import VENV,ROOT,WORK
    from process_resource_limits import limits
    from model_resources import resolve
    from pytorch_weight_package import read
    resolve(resource_profile)
    frozen=source_identity();_,weight_files=read(Path(manifest_path).parent,spec["weights"],resource_profile=resource_profile)
    output=Path(output)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh capture directory under results required")
    output.mkdir(mode=0o700);payload=output/"capture-input.json"
    payload.write_text(json.dumps(dict(manifest=spec,resource_profile=resource_profile,
                                      runtime=legacy.runtime_identity(),source_hashes=frozen)))
    if weight_files:
        (output/"weight-snapshot").mkdir()
        for name,raw in weight_files.items():(output/"weight-snapshot"/name).write_bytes(raw)
    library=os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    require(all(Path(p).is_relative_to("/nix/store") for p in library.split(":")),"Pinned library path")
    cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
        "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
        "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
        "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/model",
        "--ro-bind",str(payload),"/request.json","--bind",str(output),"/out"]
    for name in IMPLEMENTATION:cmd+=["--ro-bind",str(ROOT/"scripts/baseline"/name),"/app/"+name]
    for name,path in paths.items():cmd+=["--ro-bind",str(path),"/model/"+name]
    for name in weight_files:cmd+=["--ro-bind",str(output/"weight-snapshot"/name),"/model/"+name]
    for key,value in dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
            LD_LIBRARY_PATH=library,HOME="/nonexistent",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1").items():
        cmd+=["--setenv",key,value]
    cmd+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/pytorch_model_intake.py","--worker"]
    with (output/"capture.log").open("x") as log:
        proc=subprocess.Popen(cmd,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
                              preexec_fn=limits,start_new_session=True)
        try:proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise ValueError("Trusted arithmetic capture timeout")
    require(proc.returncode==0,"Trusted arithmetic capture failed; inspect capture.log")
    require(source_identity()==frozen,"Arithmetic capture implementation changed")
    original=json.loads(regular_bytes(output/"original.json",resolve(resource_profile).json_bytes))
    report=json.loads(regular_bytes(output/"capture-reference.json",131072))
    lowered,binding=capture_binding(original,spec,resource_profile=resource_profile,
                                    capture_metadata=report.get("capture_metadata"))
    check_reference(report,original,lowered,spec,resource_profile=resource_profile)
    (output/"trusted-python").mkdir()
    (output/"trusted-python/manifest.json").write_text(json.dumps(spec))
    for name,path in paths.items():
        raw=regular_bytes(path,65536);require(hashlib.sha256(raw).hexdigest()==spec["files"][name],"Source changed during capture")
        (output/"trusted-python"/name).write_bytes(raw)
    for name,raw in weight_files.items():
        require(regular_bytes(Path(manifest_path).parent/name,len(raw))==raw and
                regular_bytes(output/"weight-snapshot"/name,len(raw))==raw,"Weights changed during capture")
        (output/"trusted-python"/name).write_bytes(raw)
    for name,value in (("model.json",lowered),("decomposition.json",binding),("reference-report.json",report)):
        with (output/name).open("x") as f:json.dump(value,f)
    return dict(status="captured",resource_profile=resource_profile,model_sha256=digest(lowered),
                encrypted_execution=False,agent_generated=False,reference_verified=True,
                reference_scope="16 finite inputs; original and normalized nn.Module, independent math/Torch",
                normalization_policy="poseidon-fx-common-calls-v1" if spec["format"]==COMMON_FORMAT else "poseidon-fx-public-arithmetic-v1")

def worker():
    if ordered(json.loads(Path("/request.json").read_text()).get("manifest")):
        from pytorch_ordered_intake import worker as selected
        return selected()
    import importlib,random,numpy as np,torch
    from model_resources import identity
    from pytorch_weight_package import read,load_state
    from pytorch_static_trace import trace,INDEX_POLICY,ARITHMETIC_POLICY,policy_record
    from pytorch_arithmetic_normalization import normalize,rename_record,bind_record
    from pytorch_model_capture import capture,REGISTERED_FUNCTION_POLICY,ADAPTIVE_FUNCTION_POLICY
    from pytorch_source_contract import factory_parameters,output_tree
    from benchmark_graph import samples
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as independent_torch
    from operator_reference import evaluate as high
    request=json.loads(Path("/request.json").read_text());spec=request["manifest"];profile=request["resource_profile"]
    require(spec.get("format") in SOURCE_FORMATS and source_identity()==request["source_hashes"],"Arithmetic worker implementation")
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
                 cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
    require(runtime==request["runtime"],"Pinned capture runtime");torch.set_num_threads(1)
    files={name:Path("/model",name).read_text() for name in spec["files"]};validate_package(spec,files)
    module_name,entry=spec["entry"].split(".");parameters=factory_parameters(shim(spec))
    module=getattr(importlib.import_module(module_name),entry)(*parameters["args"],**parameters["kwargs"])
    arrays,_=read("/model",spec["weights"],resource_profile=profile)
    loaded=load_state(module,arrays,spec["weights"])
    require(isinstance(module,torch.nn.Module),"PyTorch nn.Module required")
    original_dtypes={name:str(value.dtype).removeprefix("torch.") for name,value in module.state_dict().items()}
    for value in module.state_dict().values():
        if value.is_floating_point():
            require(value.dtype in (torch.float32,torch.float64),"Supported floating dtype")
            require(spec["dtype_policy"]!="strict-float64-v1" or value.dtype is torch.float64,"Strict float64 source policy")
        else:require(value.dtype in (torch.int32,torch.int64,torch.bool),"Supported public dtype")
    module=module.cpu().double().eval()
    before={k:v.detach().clone() for k,v in module.state_dict().items()}
    rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    source_policy=source_shape_policy(spec)
    traced=trace(module,spec["inputs"],resource_profile=profile,policy=source_policy)
    common_record=None
    if spec["format"]==COMMON_FORMAT:
        from pytorch_arithmetic_normalization import normalize_common
        traced,common_record=normalize_common(traced,spec["inputs"],resource_profile=profile)
    normalized,record=normalize(traced,resource_profile=profile,
                                preserve_unused_state=spec["format"]==COMMON_FORMAT)
    canonical=trace(normalized,spec["inputs"],resource_profile=profile,policy=INDEX_POLICY)
    record=rename_record(record,normalized,canonical)
    original,lowered,captured=capture(canonical,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
        resource_profile=profile,dtype_policy=spec["dtype_policy"],preserve_output_structure=True,
        record_state_sources=True,capture_policy=INDEX_POLICY,approximation_policy="node-module-default-v1",
        matrix_policy="static-tensor-matmul-v1",
        functional_policy=ADAPTIVE_FUNCTION_POLICY if spec["format"] in (OPERATOR_SHAPE_FORMAT,COMMON_FORMAT) else REGISTERED_FUNCTION_POLICY)
    metadata=captured["pytorch_capture"]
    all_constant_sources=dict(metadata["state_constant_sources"])
    metadata["source_state_dtypes"]=original_dtypes
    metadata["state_constant_sources"]={name:state for name,state in metadata["state_constant_sources"].items()
                                       if state in spec["weights"]}
    metadata["weight_binding"]=loaded
    metadata["arithmetic_capture"]=bind_record(record,original,captured_fx=canonical)
    if common_record is not None:
        from pytorch_arithmetic_normalization import bind_common_record
        metadata["common_capture"]=bind_common_record(common_record,original,
            metadata["arithmetic_capture"],all_constant_sources)
    if spec["format"] in SOURCE_SHAPE_FORMATS:
        metadata["source_shape_capture"]=policy_record(spec["inputs"],profile,policy=source_policy)
    Path("/out/original.json").write_text(json.dumps(original))
    Path("/out/fx-source.txt").write_text(record["source_fx_text"])
    Path("/out/fx-normalized.txt").write_text(record["normalized_fx_text"])
    maximum=module_error=approximation_error=ratio=normal_error=normal_ratio=0.
    def compare(actual,expected):
        require(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),"Capture reference shape/nonfinite")
        measured=float(np.max(np.abs(actual-expected)/(1e-12+1e-12*np.abs(expected))))
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        return float(np.max(np.abs(actual-expected))),measured
    with torch.no_grad():
        for inputs in samples(lowered,16):
            arguments=[torch.from_numpy(inputs[s["name"]]) for s in spec["inputs"]]
            raw,structure=output_tree(module(*arguments),lambda v:isinstance(v,torch.Tensor),spec["outputs"],scalar_tensor=lambda v:v.ndim==0)
            normalized_raw,normalized_structure=output_tree(canonical(*arguments),lambda v:isinstance(v,torch.Tensor),spec["outputs"],scalar_tensor=lambda v:v.ndim==0)
            require(structure==normalized_structure==metadata["output_structure"],"Forward output structure")
            actual_module={name:v.detach().numpy().reshape(1) if v.ndim==0 else v.detach().numpy()
                           for name,v in zip(spec["outputs"],raw)}
            actual_normalized={name:v.detach().numpy().reshape(1) if v.ndim==0 else v.detach().numpy()
                               for name,v in zip(spec["outputs"],normalized_raw)}
            expected=high(original,inputs);ideal=high(original,inputs,ideal=True)
            actual=mathematical(lowered,inputs,resource_profile=profile)
            independent=independent_torch(lowered,inputs,resource_profile=profile)
            require(set(actual_module)==set(expected)==set(actual)==set(independent),"Capture output names")
            for name in expected:
                for value in (actual[name],independent[name]):
                    error,relative=compare(value,expected[name]);maximum=max(maximum,error);ratio=max(ratio,relative)
                error,relative=compare(actual_module[name],ideal[name]);module_error=max(module_error,error);ratio=max(ratio,relative)
                error,relative=compare(actual_normalized[name],actual_module[name]);normal_error=max(normal_error,error);normal_ratio=max(normal_ratio,relative)
                approximation_error=max(approximation_error,float(np.max(np.abs(expected[name]-actual_module[name]))))
            require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py_rng and
                    all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng)),"Random forward is not supported")
            require(before.keys()==module.state_dict().keys() and
                    all(torch.equal(v,module.state_dict()[k]) for k,v in before.items()),"State-mutating forward is not supported")
    report=dict(format=REFERENCES[spec["format"]],mode="isolated_trusted_pytorch_reference",
        original_sha256=digest(original),model_sha256=digest(lowered),manifest_sha256=digest(spec),
        source_hashes=source_identity(),runtime=runtime,probes=16,resource_profile=identity(profile),
        reference_status="passed",original_module_checked=True,normalized_module_checked=True,
        arithmetic_semantics_preserved=True,pure_forward_checked=True,
        comparison_tolerance=dict(atol=1e-12,rtol=1e-12),max_reference_tolerance_ratio=ratio,
        max_decomposition_error=maximum,max_original_module_error=module_error,
        max_arithmetic_normalization_error=normal_error,max_arithmetic_normalization_ratio=normal_ratio,
        max_original_function_approximation_error=approximation_error,approximation_accuracy_certified=False,
        runtime_layout_validated=False,encrypted_execution=False,agent_generated=False,paid_calls=0,
        capture_metadata=metadata,reference_dtype="float64",source_precision_execution_compared=False)
    if spec["format"]==COMMON_FORMAT:report["common_semantics_preserved"]=True
    Path("/out/capture-reference.json").write_text(json.dumps(report))

if __name__=="__main__":
    require(sys.argv[1:]==["--worker"],"Internal worker only");worker()
