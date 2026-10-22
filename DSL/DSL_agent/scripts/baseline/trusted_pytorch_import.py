"""Explicit trusted local capture and finite reference checks in one sandbox."""
import hashlib,json,math,os,re,signal,subprocess,sys
from pathlib import Path
from benchmark_graph import require,digest
from model_weight_package import regular_bytes
from pytorch_source_contract import (FORMAT as SOURCE_V2,WEIGHT_FORMAT as SOURCE_V3,STATIC_FORMAT as SOURCE_V4,APPROXIMATION_FORMAT as SOURCE_V5,INDEX_FORMAT as SOURCE_V6,MATRIX_FORMAT as SOURCE_V7,INDEX_FORMATS,SCOPED_FORMATS,STATIC_FORMATS,STATE_FORMATS,EXTENDED,LEGACY,validate_manifest,
                                     factory_parameters,validate_output_tree,output_tree)

PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v1"
REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v1"
PACKAGE_V2="poseidon-trusted-pytorch-provenance-v2"
REFERENCE_V2="poseidon-trusted-pytorch-reference-v2"
PACKAGE_V3="poseidon-trusted-pytorch-provenance-v3"
REFERENCE_V3="poseidon-trusted-pytorch-reference-v3"
PACKAGE_V4="poseidon-trusted-pytorch-provenance-v4"
REFERENCE_V4="poseidon-trusted-pytorch-reference-v4"
PACKAGE_V5="poseidon-trusted-pytorch-provenance-v5"
REFERENCE_V5="poseidon-trusted-pytorch-reference-v5"
PACKAGE_V6="poseidon-trusted-pytorch-provenance-v6"
REFERENCE_V6="poseidon-trusted-pytorch-reference-v6"
PACKAGE_V7="poseidon-trusted-pytorch-provenance-v7"
REFERENCE_V7="poseidon-trusted-pytorch-reference-v7"

def package_format(spec):
    validate_manifest(spec)
    return {SOURCE_V2:PACKAGE_V2,SOURCE_V3:PACKAGE_V3,SOURCE_V4:PACKAGE_V4,SOURCE_V5:PACKAGE_V5,SOURCE_V6:PACKAGE_V6,SOURCE_V7:PACKAGE_V7}.get(spec["format"],PACKAGE_FORMAT)
IMPLEMENTATION=("trusted_pytorch_import.py","process_resource_limits.py","pytorch_model_capture.py","model_resources.py",
    "benchmark_graph.py","model_decomposition.py","model_weight_package.py",
    "benchmark_math.py","benchmark_torch.py","operator_reference.py","pytorch_source_contract.py",
    "pytorch_weight_package.py","pytorch_static_trace.py","pytorch_approximation.py","pytorch_static_index.py","pytorch_matrix_plan.py","pytorch_split_plan.py")

def validate_package(spec,files):
    validate_manifest(spec)
    hashes=spec["files"]
    require(type(hashes) is dict and 1<=len(hashes)<=8 and type(files) is dict and set(files)==set(hashes),"Source inventory")
    require(type(spec["entry"]) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*",spec["entry"]),"Factory entry")
    total=0
    for name,expected in hashes.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py",name),"Flat Python filename")
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}",expected),"Source hash")
        require(type(files[name]) is str,"Source text")
        raw=files[name].encode();total+=len(raw)
        require(len(raw)<=65536 and total<=65536 and hashlib.sha256(raw).hexdigest()==expected,"Source hash/budget")
    require(spec["entry"].split(".")[0]+".py" in hashes,"Factory module missing")
    return spec

def prepare(manifest_path):
    path=Path(manifest_path)
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"Duplicate trusted manifest key");value[key]=item
        return value
    spec=json.loads(regular_bytes(path,131072),object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError("Nonfinite trusted manifest")))
    validate_manifest(spec)
    require(type(spec["files"]) is dict and 1<=len(spec["files"])<=8,"Source inventory")
    files={};paths={};total=0
    for name,expected in spec["files"].items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py",name),"Flat Python filename")
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}",expected),"Source hash")
        raw=regular_bytes(path.parent/name,65536-total)
        require(hashlib.sha256(raw).hexdigest()==expected,"Source hash/budget")
        total+=len(raw);files[name]=raw.decode()
        paths[name]=(path.parent/name).resolve()
    validate_package(spec,files)
    return spec,paths

def preparation_options(provenance,model,options=None):
    """Choose the versioned packing for an already validated v6 import."""
    options={} if options is None else options
    require(type(options) is dict,"Component option fields")
    result=dict(options)
    if (provenance.get("trusted_pytorch",{}).get("format")in (PACKAGE_V6,PACKAGE_V7)
        and model.get("format")=="poseidon-model-graph-v2"
        and result.get("capacity_packing_version") is None):
        from unified_capacity_constants import INDEX_VERSION
        result["capacity_packing_version"]=INDEX_VERSION
    return result

def source_identity(root=None):
    root=Path(__file__).parent if root is None else Path(root)
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in IMPLEMENTATION}

def runtime_identity():
    from platform_config import configuration
    return dict(python=[3,10,14],numpy="1.25.2",torch=configuration()["torch"],cuda=None,hip=None)

def check_capture_metadata(metadata,original,spec,*,resource_profile="legacy-v1"):
    fields={"torch_version","trusted_code","training","graph_sha256","dtype_policy","input_dtype",
            "source_state_dtypes","constant_dtypes","output_structure","source_precision_execution_compared"}
    if spec["format"] in STATE_FORMATS:fields|={"state_constant_sources","weight_binding"}
    if spec["format"] in STATIC_FORMATS:fields|={"static_capture"}
    if spec["format"] in SCOPED_FORMATS:fields|={"approximation_selection"}
    if spec["format"] in INDEX_FORMATS:fields|={"index_capture"}
    if spec["format"]==SOURCE_V7:fields|={"matrix_capture"}
    require(type(metadata) is dict and set(metadata)==fields,"Capture metadata")
    require(metadata["torch_version"]==runtime_identity()["torch"] and metadata["trusted_code"] is True
            and metadata["training"] is False and metadata["graph_sha256"]==digest(original),
            "Capture metadata source/runtime")
    require(metadata["dtype_policy"]==spec["dtype_policy"] and metadata["input_dtype"]=="float64"
            and metadata["source_precision_execution_compared"] is False,"Capture dtype policy")
    for name in ("source_state_dtypes","constant_dtypes"):
        value=metadata[name]
        require(type(value) is dict and len(value)<=1024 and all(
            type(k) is str and 0<len(k)<=256 and v in ("float32","float64","int32","int64","bool")
            for k,v in value.items()),"Capture dtype records")
    require(set(metadata["constant_dtypes"])<=set(original["constants"]),"Capture constant dtype binding")
    if spec["dtype_policy"]=="strict-float64-v1":
        require("float32" not in metadata["source_state_dtypes"].values()
                and "float32" not in metadata["constant_dtypes"].values(),"Strict float64 capture records")
    if spec["format"] in STATE_FORMATS:
        from pytorch_weight_package import declarations,check_binding
        records=declarations(spec["weights"],resource_profile="general-tensors-v1")
        check_binding(metadata["weight_binding"],records)
        dtype_names={"<f4":"float32","<f8":"float64","<i4":"int32","<i8":"int64","|b1":"bool"}
        require(metadata["source_state_dtypes"]=={name:dtype_names[item["dtype"]] for name,item in records.items()},
                "Loaded state dtype inventory")
        mapping=metadata["state_constant_sources"]
        require(type(mapping) is dict and set(mapping)<=set(original["constants"]) and
                all(type(v) is str and v in records for v in mapping.values()),"State constant source mapping")
        import numpy as np
        for name,state in mapping.items():
            normalized=np.asarray(original["constants"][name],dtype="<f8")
            require(list(normalized.shape)==records[state]["shape"] and
                    hashlib.sha256(normalized.tobytes(order="C")).hexdigest()==
                    metadata["weight_binding"]["normalized_float64_sha256"][state],
                    "Graph constant differs from loaded state")
    validate_output_tree(metadata["output_structure"],spec["outputs"],allow_scalars=spec["format"] in STATIC_FORMATS)
    if spec["format"] in STATIC_FORMATS:
        from pytorch_static_trace import policy_record
        require(metadata["static_capture"]==policy_record(spec["inputs"],resource_profile,policy="signature-static-index-v1" if spec["format"] in INDEX_FORMATS else "signature-static-v1"), "Static capture policy binding")
    if spec["format"] in SCOPED_FORMATS:
        from pytorch_approximation import check_record
        check_record(metadata["approximation_selection"],original,spec["approximations"])
    if spec["format"] in INDEX_FORMATS:
        from pytorch_static_index import check_record
        check_record(metadata["index_capture"],original,resource_profile=resource_profile)
    if spec["format"]==SOURCE_V7:
        from pytorch_matrix_plan import check_record
        check_record(metadata["matrix_capture"],original,resource_profile=resource_profile)
    return metadata

def capture_binding(original,spec,*,resource_profile="legacy-v1",capture_metadata=None):
    from model_decomposition import decompose
    require(original.get("inputs")==spec["inputs"] and [o["name"] for o in original.get("outputs",[])]==spec["outputs"],"Capture input/output binding")
    lowered,binding=decompose(original,resource_profile=resource_profile)
    binding["trusted_source_hashes"]=spec["files"];binding["manifest_sha256"]=digest(spec)
    binding["capture_implementation"]=source_identity()
    binding["capture_torch_version"]=runtime_identity()["torch"]
    if spec["format"] in EXTENDED:
        binding["pytorch_capture"]=check_capture_metadata(capture_metadata,original,spec,resource_profile=resource_profile)
    else:require(capture_metadata is None,"Legacy capture metadata")
    binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    return lowered,binding

def check_reference(report,original,lowered,spec,*,resource_profile="legacy-v1"):
    from model_resources import identity
    expected={SOURCE_V2:REFERENCE_V2,SOURCE_V3:REFERENCE_V3,SOURCE_V4:REFERENCE_V4,SOURCE_V5:REFERENCE_V5,SOURCE_V6:REFERENCE_V6,SOURCE_V7:REFERENCE_V7}.get(spec["format"],REFERENCE_FORMAT)
    require(type(report) is dict and report.get("format")==expected and
        report.get("mode")=="isolated_trusted_pytorch_reference","Trusted reference format")
    require(report.get("original_sha256")==digest(original) and report.get("model_sha256")==digest(lowered) and
        report.get("manifest_sha256")==digest(spec) and report.get("source_hashes")==source_identity(),"Trusted reference identity")
    require(report.get("runtime")==runtime_identity() and report.get("probes")==16 and
        report.get("resource_profile")==identity(resource_profile),"Trusted reference runtime/probes")
    require(report.get("reference_status")=="passed" and report.get("original_module_checked") is True and
        report.get("pure_forward_checked") is True and report.get("encrypted_execution") is False and
        report.get("agent_generated") is False and report.get("paid_calls")==0 and
        report.get("runtime_layout_validated") is False and report.get("approximation_accuracy_certified") is False,
        "Trusted reference acceptance scope")
    require(report.get("comparison_tolerance")==dict(atol=1e-12,rtol=1e-12),"Trusted reference tolerance")
    for name in ("max_decomposition_error","max_original_module_error","max_original_function_approximation_error","max_reference_tolerance_ratio"):
        value=report.get(name)
        require(type(value) in (int,float) and math.isfinite(value) and value>=0,"Trusted reference metric")
    require(report["max_reference_tolerance_ratio"]<=1,"Trusted reference comparison failed")
    if spec["format"] in EXTENDED:
        check_capture_metadata(report.get("capture_metadata"),original,spec,resource_profile=resource_profile)
        require(report.get("reference_dtype")=="float64" and
                report.get("source_precision_execution_compared") is False,
                "Normalized float64 reference scope")
    return report

def run(manifest_path,output,*,resource_profile="legacy-v1",trusted=False):
    require(trusted is True,"Explicit --trust-local-code required")
    from hecate_python_env import VENV,ROOT,WORK
    from process_resource_limits import limits
    from model_resources import resolve
    resolve(resource_profile)
    spec,paths=prepare(manifest_path);frozen=source_identity()
    weight_files={}
    if spec["format"] in STATE_FORMATS:
        from pytorch_weight_package import read
        _,weight_files=read(Path(manifest_path).parent,spec["weights"],resource_profile=resource_profile)
    output=Path(output)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh capture directory under results required")
    output.mkdir()
    payload=output/"capture-input.json"
    payload.write_text(json.dumps(dict(manifest=spec,resource_profile=resource_profile,runtime=runtime_identity())))
    if weight_files:
        (output/"weight-snapshot").mkdir()
        for name,raw in weight_files.items():
            with (output/"weight-snapshot"/name).open("xb") as stream:stream.write(raw)
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
    cmd+=["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/trusted_pytorch_import.py","--worker"]
    with (output/"capture.log").open("x") as log:
        proc=subprocess.Popen(cmd,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
            preexec_fn=limits,start_new_session=True)
        try:proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise ValueError("Trusted capture timeout")
    require(proc.returncode==0,"Trusted capture failed; inspect capture.log")
    require(source_identity()==frozen,"Capture implementation changed")
    original=json.loads(regular_bytes(output/"original.json",resolve(resource_profile).json_bytes))
    report=json.loads(regular_bytes(output/"capture-reference.json",131072))
    lowered,binding=capture_binding(original,spec,resource_profile=resource_profile,
                                    capture_metadata=report.get("capture_metadata"))
    check_reference(report,original,lowered,spec,resource_profile=resource_profile)
    # The package stores source provenance, not executable import hooks.
    files={}
    for name,path in paths.items():
        files[name]=regular_bytes(path,65536).decode()
    validate_package(spec,files)
    (output/"trusted-python").mkdir()
    (output/"trusted-python/manifest.json").write_text(json.dumps(spec))
    for name,text in files.items():(output/"trusted-python"/name).write_text(text)
    for name,raw in weight_files.items():
        require(regular_bytes(Path(manifest_path).parent/name,len(raw))==raw and
                regular_bytes(output/"weight-snapshot"/name,len(raw))==raw,"Weights changed during capture")
        with (output/"trusted-python"/name).open("xb") as stream:stream.write(raw)
    for name,value in (("model.json",lowered),("decomposition.json",binding),("reference-report.json",report)):
        with (output/name).open("x") as f:json.dump(value,f)
    return dict(status="captured",resource_profile=resource_profile,model_sha256=digest(lowered),
        encrypted_execution=False,agent_generated=False,reference_verified=True,
        reference_scope="16 finite inputs; original nn.Module and two independent decomposed references")

def worker():
    import importlib,random
    import numpy as np
    import torch
    from pytorch_model_capture import capture
    from benchmark_graph import samples
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as independent_torch
    from operator_reference import evaluate as high
    from model_resources import identity
    request=json.loads(Path("/request.json").read_text());spec=request["manifest"]
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
        cuda=torch.version.cuda,hip=getattr(torch.version,"hip",None))
    require(runtime==request["runtime"],"Pinned capture runtime")
    torch.set_num_threads(1)
    files={name:Path("/model",name).read_bytes().decode("utf-8") for name in spec["files"]}
    validate_package(spec,files)
    mod,entry=spec["entry"].split(".")
    parameters=factory_parameters(spec)
    module=getattr(importlib.import_module(mod),entry)(*parameters["args"],**parameters["kwargs"])
    extended=spec["format"] in EXTENDED
    loaded_weights=None
    if spec["format"] in STATE_FORMATS:
        from pytorch_weight_package import read,load_state
        arrays,_=read("/model",spec["weights"],resource_profile=request["resource_profile"])
        loaded_weights=load_state(module,arrays,spec["weights"])
    original,lowered,captured=capture(module,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
        resource_profile=request["resource_profile"],dtype_policy=spec.get("dtype_policy","legacy-float64-v1"),
        preserve_output_structure=extended,record_state_sources=spec["format"] in STATE_FORMATS,
        capture_policy=("signature-static-index-v1" if spec["format"] in INDEX_FORMATS else
                        "signature-static-v1" if spec["format"] in STATIC_FORMATS else "symbolic-v1"),
        approximation_policy="node-module-default-v1" if spec["format"] in SCOPED_FORMATS else "operator-defaults-v1",
        matrix_policy="static-tensor-matmul-v1" if spec["format"]==SOURCE_V7 else "rank-two-three-v1")
    if loaded_weights is not None:
        captured["pytorch_capture"]["weight_binding"]=loaded_weights
    Path("/out/original.json").write_text(json.dumps(original))
    before={k:v.detach().clone() for k,v in module.state_dict().items()}
    rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    maximum=module_error=approximation_error=ratio=0.
    def compare(actual,expected):
        nonlocal ratio
        require(actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),"Capture reference shape/nonfinite")
        ratio=max(ratio,float(np.max(np.abs(actual-expected)/(1e-12+1e-12*np.abs(expected)))))
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        return float(np.max(np.abs(actual-expected)))
    with torch.no_grad():
        for inputs in samples(lowered,16):
            raw=module(*(torch.from_numpy(inputs[i["name"]]) for i in spec["inputs"]))
            if extended:
                raw,structure=output_tree(raw,lambda v:isinstance(v,torch.Tensor),spec["outputs"],
                    scalar_tensor=(lambda v:v.ndim==0) if spec["format"] in STATIC_FORMATS else None)
                require(structure==captured["pytorch_capture"]["output_structure"],"Forward output structure changed")
            else:raw=list(raw) if isinstance(raw,(tuple,list)) else [raw]
            require(len(raw)==len(spec["outputs"]) and all(isinstance(v,torch.Tensor) and v.dtype==torch.float64 and v.device.type=="cpu" for v in raw),"Original module output")
            actual_module={name:(v.detach().numpy().reshape(1) if spec["format"] in STATIC_FORMATS and v.ndim==0
                                else v.detach().numpy()) for name,v in zip(spec["outputs"],raw)}
            expected=high(original,inputs);ideal=high(original,inputs,ideal=True)
            actual=mathematical(lowered,inputs,resource_profile=request["resource_profile"])
            independent=independent_torch(lowered,inputs,resource_profile=request["resource_profile"])
            require(set(actual_module)==set(expected)==set(ideal)==set(actual)==set(independent),"Capture reference names")
            for name in expected:
                maximum=max(maximum,compare(actual[name],expected[name]),compare(independent[name],expected[name]))
                module_error=max(module_error,compare(actual_module[name],ideal[name]))
                approximation_error=max(approximation_error,float(np.max(np.abs(expected[name]-actual_module[name]))))
            require(torch.equal(torch.get_rng_state(),rng) and random.getstate()==py_rng and
                all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),np_rng)),"Random forward is not supported")
            require(before.keys()==module.state_dict().keys() and
                all(torch.equal(v,module.state_dict()[k]) for k,v in before.items()),"State-mutating forward is not supported")
    report=dict(format={SOURCE_V2:REFERENCE_V2,SOURCE_V3:REFERENCE_V3,SOURCE_V4:REFERENCE_V4,SOURCE_V5:REFERENCE_V5,SOURCE_V6:REFERENCE_V6,SOURCE_V7:REFERENCE_V7}.get(spec["format"],REFERENCE_FORMAT),mode="isolated_trusted_pytorch_reference",
        original_sha256=digest(original),model_sha256=digest(lowered),manifest_sha256=digest(spec),
        source_hashes=source_identity(),runtime=runtime,probes=16,resource_profile=identity(request["resource_profile"]),
        reference_status="passed",original_module_checked=True,pure_forward_checked=True,
        comparison_tolerance=dict(atol=1e-12,rtol=1e-12),max_reference_tolerance_ratio=ratio,
        max_decomposition_error=maximum,max_original_module_error=module_error,
        max_original_function_approximation_error=approximation_error,approximation_accuracy_certified=False,
        runtime_layout_validated=False,encrypted_execution=False,agent_generated=False,paid_calls=0)
    if extended:
        report.update(capture_metadata=captured["pytorch_capture"],reference_dtype="float64",
                      source_precision_execution_compared=False)
    Path("/out/capture-reference.json").write_text(json.dumps(report))

if __name__=="__main__":
    require(sys.argv[1:]==["--worker"],"Internal worker only");worker()
