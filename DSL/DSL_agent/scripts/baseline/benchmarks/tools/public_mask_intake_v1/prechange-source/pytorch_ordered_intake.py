"""Explicit trusted-local PyTorch V12 worker and capture boundary.

No model source is imported in the caller. The worker runs with the existing
read-only SDK, no network, a fresh output directory and fixed resource limits.
"""
import copy, hashlib, json, os, signal, subprocess, sys
from pathlib import Path
from pytorch_source_package_policy import selected as packaged_source
from benchmark_graph import require, digest
from model_resources import resolve, identity
from model_weight_package import regular_bytes
from pytorch_ordered_contract import (
    FORMAT, PACKAGE_FORMAT, REFERENCE_FORMAT, SOURCE_FORMATS, COMMON_SOURCE_FORMAT, FUSED_SOURCE_FORMAT, GELU_SOURCE_FORMAT, MHA_SOURCE_FORMAT, PARTITION_SOURCE_FORMAT, EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,
    package_version, reference_version, prepare, validate_package, shim,
    source_identity, implementation_paths, capture_binding, check_reference,
)

def package_format(spec):
    import trusted_pytorch_import as legacy
    legacy.validate_manifest(shim(spec))
    return package_version(spec)

def preparation_options(provenance, model, options=None):
    require(options is None or type(options) is dict, "Component option fields")
    result = copy.deepcopy(options or {})
    if model.get("format") == "poseidon-model-graph-v2" and result.get("capacity_packing_version") is None:
        from unified_capacity_constants import INDEX_VERSION
        result["capacity_packing_version"] = INDEX_VERSION
    return result

def capture_request(request_path,model_root,output_root):
    import importlib, numpy as np, torch
    import trusted_pytorch_import as legacy
    from pytorch_source_contract import factory_parameters
    from pytorch_weight_package import read, load_state
    from pytorch_ordered_capture import prepare_module
    from pytorch_ordered_reference import validate as legacy_validate
    from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
    from platform_config import require_python_packages
    require_python_packages(torch,np)
    request = json.loads(Path(request_path).read_text())
    spec, profile = request["manifest"], request["resource_profile"]
    require(spec.get("format") in SOURCE_FORMATS and request["source_hashes"] == source_identity(),
            "Ordered worker implementation")
    runtime = dict(python=list(sys.version_info[:3]), numpy=np.__version__, torch=torch.__version__,
                   cuda=torch.version.cuda, hip=getattr(torch.version,"hip",None))
    require(runtime == request["runtime"], "Pinned ordered capture runtime")
    if spec["format"]in(MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT):
        from pytorch_mha_ordered_reference import validate
    elif spec["format"]==GELU_SOURCE_FORMAT:
        from pytorch_gelu_reference import validate
    else:validate=legacy_validate
    torch.set_num_threads(1)
    files = {name:Path(model_root,name).read_bytes() if packaged_source(spec) else Path(model_root,name).read_text() for name in spec["files"]}
    validate_package(spec, files)
    parameters = factory_parameters(shim(spec))
    module_name, entry = spec["entry"].rsplit(".",1)
    module = getattr(importlib.import_module(module_name),entry)(*parameters["args"],**parameters["kwargs"])
    require(isinstance(module,torch.nn.Module), "Trusted factory must return nn.Module")
    arrays, _ = read(model_root,spec["weights"],resource_profile=profile)
    loaded = load_state(module,arrays,spec["weights"])
    if spec.get("format")in("poseidon-trusted-pytorch-v23","poseidon-trusted-pytorch-v24","poseidon-trusted-pytorch-v25","poseidon-trusted-pytorch-v26","poseidon-trusted-pytorch-v27","poseidon-trusted-pytorch-v28","poseidon-trusted-pytorch-v29","poseidon-trusted-pytorch-v30","poseidon-trusted-pytorch-v31","poseidon-trusted-pytorch-v32"):
        from pytorch_public_expression_capture import capture_loaded_module as selected
        return selected(request,module,arrays,loaded,output_root)
    if spec.get("format")=="poseidon-trusted-pytorch-v22":
        from pytorch_public_arithmetic_capture import capture_loaded_module as selected
        return selected(request,module,arrays,loaded,output_root)
    prepared = prepare_module(module,spec["inputs"],spec["outputs"],approximations=spec["approximations"],
                              resource_profile=profile,dtype_policy=spec["dtype_policy"],
                              approximation_policy="node-module-default-v1",matrix_policy="static-tensor-matmul-v1",
                              functional_policy=ADAPTIVE_FUNCTION_POLICY,
                               common_calls=spec["format"] in (COMMON_SOURCE_FORMAT,FUSED_SOURCE_FORMAT,GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT),
                              fused_matrix_policy=("poseidon-fx-fused-matrix-draft-v1" if spec["format"] in (FUSED_SOURCE_FORMAT,GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else None),
                               gelu_source_policy=("poseidon-gelu-source-variants-v1" if spec["format"] in (GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else "exact-gelu-source-v1"),
                               mha_source_policy=("poseidon-mha-ordered-source-v1" if spec["format"]in(MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else None),
                               **({"partition_source_policy":"poseidon-pytorch-static-partition-v2" if spec["format"] in (SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else "poseidon-pytorch-static-partition-v1"}if spec["format"]in(PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else {}),
                               **({"einsum_source_policy":"poseidon-pytorch-static-einsum-v1"}if spec["format"]==EINSUM_SOURCE_FORMAT else {"einsum_source_policy":"poseidon-pytorch-static-einsum-v2"}if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT) else {}),
                                **({"exact_arithmetic_policy":"poseidon-pytorch-exact-arithmetic-v1"}if spec["format"]==EXACT_ARITHMETIC_SOURCE_FORMAT else {}))
    high, lowered, captured, context = prepared
    metadata = captured["pytorch_capture"]
    metadata["weight_binding"] = loaded
    lowered, bound = capture_binding(high,spec,resource_profile=profile,capture_metadata=metadata,
                                    expected_weight_binding=loaded)
    finite = validate((high,lowered,bound,context),spec["inputs"],spec["outputs"],
                      resource_profile=profile,count=16)
    alias = metadata["ordered_alias_capture"]
    adapter = alias["capture_adapter"]
    Path(output_root,"original.json").write_text(json.dumps(high,allow_nan=False))
    Path(output_root,"fx-source.txt").write_text(adapter["source_fx_text"])
    Path(output_root,"fx-normalized.txt").write_text(adapter["normalized_fx_text"])
    report = dict(format=reference_version(spec),mode="isolated_trusted_pytorch_reference",
                  original_sha256=digest(high),model_sha256=digest(lowered),manifest_sha256=digest(spec),
                  source_hashes=source_identity(),runtime=runtime,probes=16,resource_profile=identity(profile),
                  reference_status="passed",reference_dtype="float64",finite_reference=finite,
                  capture_metadata=metadata,input_mutation_policy="ordered-nonoverlapping-registered-effects",
                  source_precision_execution_compared=False,approximation_accuracy_certified=False,
                  runtime_layout_validated=False,encrypted_execution=False,agent_generated=False,paid_calls=0)
    check_reference(report,high,lowered,spec,resource_profile=profile,expected_weight_binding=loaded)
    Path(output_root,"capture-reference.json").write_text(json.dumps(report,allow_nan=False))

def worker():
    from pytorch_intake_diagnostics import ApproximationRequired, FILENAME, EXIT_CODE, write
    try:
        capture_request("/request.json","/model","/out")
    except ApproximationRequired as error:
        from strict_json_files import strict_file
        request = strict_file(Path("/request.json"), 1310720)
        write(Path("/out") / FILENAME, error.public_failure,
              request["manifest"], request["source_hashes"])
        raise SystemExit(EXIT_CODE) from None

def run(manifest_path, output, *, resource_profile="legacy-v1", trusted=False):
    require(trusted is True, "Explicit --trust-local-code required")
    import trusted_pytorch_import as legacy
    from workspace_paths import require_linux_backend, work_root
    from platform_config import configuration
    require_linux_backend()
    from hecate_python_env import VENV, WORK
    from process_resource_limits import limits
    from pytorch_weight_package import read, binding
    spec, paths = prepare(manifest_path)
    cap = resolve(resource_profile)
    frozen = source_identity()
    arrays, weight_files = read(Path(manifest_path).parent,spec["weights"],resource_profile=resource_profile)
    expected_weights = binding(arrays,spec["weights"])
    output = Path(output)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh capture directory under results required")
    output.mkdir(mode=0o700)
    request = output/"capture-input.json"
    request.write_text(json.dumps(dict(manifest=spec,resource_profile=resource_profile,
                                      runtime=legacy.runtime_identity(),source_hashes=frozen)))
    if weight_files:
        (output/"weight-snapshot").mkdir()
        for name,raw in weight_files.items(): (output/"weight-snapshot"/name).write_bytes(raw)
    library = os.environ["HECATE_PYTHON_LIBRARY_PATH"]
    require(all(Path(p).is_relative_to("/nix/store") for p in library.split(":")), "Pinned capture library path")
    cmd = ["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
           "--ro-bind","/nix/store","/nix/store","--ro-bind",str(VENV),str(VENV),
           "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
           "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/model",
           "--ro-bind",str(request),"/request.json","--bind",str(output),"/out"]
    for name,path in implementation_paths().items(): cmd += ["--ro-bind",str(path),"/app/"+name]
    if packaged_source(spec):
        from pytorch_source_package_policy import mount_directories
        cmd += mount_directories(spec)
    for name,path in paths.items(): cmd += ["--ro-bind",str(path),"/model/"+name]
    for name in weight_files: cmd += ["--ro-bind",str(output/"weight-snapshot"/name),"/model/"+name]
    for key,value in dict(PYTHONPATH="/app:/model",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
                          LD_LIBRARY_PATH=library,HOME="/nonexistent",
                          POSEIDON_PLATFORM=configuration()["id"],POSEIDON_WORK_ROOT=str(work_root()),
                          OMP_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8").items():
        cmd += ["--setenv",key,value]
    cmd += ["--chdir","/out","--",str(VENV/"bin/python"),"-B","/app/pytorch_ordered_intake.py","--worker"]
    with (output/"capture.log").open("x") as log:
        proc = subprocess.Popen(cmd,env={"PATH":"/usr/bin:/bin"},stdout=log,stderr=subprocess.STDOUT,
                                preexec_fn=limits,start_new_session=True)
        try: proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            raise ValueError("Trusted ordered capture timeout")
    require(source_identity() == frozen, "Ordered capture implementation changed")
    if proc.returncode != 0:
        from pytorch_intake_diagnostics import ApproximationRequired, FILENAME, EXIT_CODE, read
        if proc.returncode == EXIT_CODE:
            raise ApproximationRequired(read(output / FILENAME, spec, frozen)) from None
        raise ValueError("Trusted ordered capture failed; inspect capture.log")
    original = json.loads(regular_bytes(output/"original.json",cap.json_bytes))
    public_options={"expected_weight_arrays":arrays} if spec["format"]in("poseidon-trusted-pytorch-v22","poseidon-trusted-pytorch-v23","poseidon-trusted-pytorch-v24","poseidon-trusted-pytorch-v25","poseidon-trusted-pytorch-v26","poseidon-trusted-pytorch-v27","poseidon-trusted-pytorch-v28","poseidon-trusted-pytorch-v29","poseidon-trusted-pytorch-v30","poseidon-trusted-pytorch-v31","poseidon-trusted-pytorch-v32") else {}
    report = json.loads(regular_bytes(output/"capture-reference.json",cap.json_bytes))
    lowered, bound = capture_binding(original,spec,resource_profile=resource_profile,
                                    capture_metadata=report.get("capture_metadata"),
                                    expected_weight_binding=expected_weights,**public_options)
    check_reference(report,original,lowered,spec,resource_profile=resource_profile,
                    expected_weight_binding=expected_weights,**public_options)
    (output/"trusted-python").mkdir()
    (output/"trusted-python/manifest.json").write_text(json.dumps(spec))
    packaged_files = None
    if packaged_source(spec):
        from pytorch_ordered_contract import source_package_files
        packaged_files = source_package_files(spec,Path(manifest_path).parent)
    for name,path in paths.items():
        raw = packaged_files[name] if packaged_files is not None else regular_bytes(path,65536)
        require(hashlib.sha256(raw).hexdigest() == spec["files"][name], "Source changed during capture")
        target = output/"trusted-python"/name
        target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with target.open("xb") as stream:stream.write(raw)
    for name,raw in weight_files.items():
        require(regular_bytes(Path(manifest_path).parent/name,len(raw)) == raw and
                regular_bytes(output/"weight-snapshot"/name,len(raw)) == raw, "Weight package changed during capture")
        (output/"trusted-python"/name).write_bytes(raw)
    for name,value in (("model.json",lowered),("decomposition.json",bound),("reference-report.json",report)):
        with (output/name).open("x") as stream: json.dump(value,stream,allow_nan=False)
    return dict(status="captured",resource_profile=resource_profile,model_sha256=digest(lowered),
                reference_verified=True,encrypted_execution=False,agent_generated=False,
                reference_scope="16 finite independent CPU float64 inputs; ordered effects",
                normalization_policy=spec["format"])

if __name__ == "__main__":
    require(sys.argv[1:] == ["--worker"], "Internal worker only")
    worker()
