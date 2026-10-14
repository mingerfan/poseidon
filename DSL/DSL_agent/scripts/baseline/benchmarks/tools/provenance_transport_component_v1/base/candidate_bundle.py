from pytorch_logical_view_source import SOURCE_FORMAT as VIEW_SOURCE_FORMAT, PACKAGE_FORMAT as VIEW_PACKAGE_FORMAT, REFERENCE_FORMAT as VIEW_REFERENCE_FORMAT
from pytorch_public_join_source import SOURCE_FORMAT as JOIN_SOURCE_FORMAT, PACKAGE_FORMAT as JOIN_PACKAGE_FORMAT, REFERENCE_FORMAT as JOIN_REFERENCE_FORMAT
from pytorch_public_mask_source import SOURCE_FORMAT as MASK_SOURCE_FORMAT, PACKAGE_FORMAT as MASK_PACKAGE_FORMAT, REFERENCE_FORMAT as MASK_REFERENCE_FORMAT
"""Portable source manifests; exact SDK replay is separate from source integrity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from pytorch_public_constant_source import SOURCE_FORMAT as CONSTANT_SOURCE_FORMAT, PACKAGE_FORMAT as CONSTANT_PACKAGE_FORMAT, REFERENCE_FORMAT as CONSTANT_REFERENCE_FORMAT
from pytorch_public_tensor_transform_source import SOURCE_FORMAT as STATE_TRANSFORM_SOURCE_FORMAT, PACKAGE_FORMAT as STATE_TRANSFORM_PACKAGE_FORMAT, REFERENCE_FORMAT as STATE_TRANSFORM_REFERENCE_FORMAT
import shlex
import sys
from benchmark_graph import digest, canonical, require
from strict_json_files import strict_file, dump
from compiler_configuration import configuration
from candidate_contract import request_file_bytes
from component_contract import reconstruct_request, runner_options

FILES = {"model.json", "candidate.py"}  # v1 compatibility
V2_FILES = FILES | {"request.json"}
PROVENANCE = "model-provenance.json"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check_provenance(value, model):
    from operator_arithmetic_graph import decompose
    from benchmark_graph import graph_resource_profile,validate
    validate(model)
    resource_profile=graph_resource_profile(model)
    require(type(value) is dict and {"original", "decomposition", "reference_report"} <= set(value) <=
            {"original", "decomposition", "reference_report", "python", "trusted_pytorch"}, "Model provenance fields")
    require(not ("python" in value and "trusted_pytorch" in value), "Ambiguous Python provenance")
    if "trusted_pytorch" in value:
        from pytorch_model_intake import package_format,validate_package,capture_binding,check_reference
        package=value["trusted_pytorch"]
        from pytorch_ordered_contract import SOURCE_FORMATS as ORDERED,PUBLIC_TENSOR_SOURCE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,weight_binding_from_payload,weight_arrays_from_payload
        ordered=type(package) is dict and package.get("manifest",{}).get("format") in ORDERED
        require(type(package) is dict and set(package)==({"format","manifest","files","weight_files"} if ordered
                else {"format","manifest","files"}) and
                package["format"]==package_format(package["manifest"]),"Trusted Python provenance package")
        validate_package(package["manifest"],package["files"])
        options={}
        if ordered:
            options["expected_weight_binding"]=weight_binding_from_payload(
                package["manifest"],package["weight_files"],resource_profile=resource_profile)
            if package["manifest"]["format"]in(PUBLIC_TENSOR_SOURCE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,MASK_SOURCE_FORMAT,JOIN_SOURCE_FORMAT,VIEW_SOURCE_FORMAT):
                options["expected_weight_arrays"]=weight_arrays_from_payload(package["manifest"],package["weight_files"],resource_profile=resource_profile)
        lowered,binding=capture_binding(value["original"],package["manifest"],resource_profile=resource_profile,
                                       capture_metadata=value["reference_report"].get("capture_metadata"),**options)
        check_reference(value["reference_report"],value["original"],lowered,package["manifest"],
                        resource_profile=resource_profile,**options)
    elif "python" in value:
        require(resource_profile=="legacy-v1","Restricted Python resource extension pending")
        from restricted_model_python import parse
        package = value["python"]
        require(type(package) is dict and set(package) == {"manifest", "files"}, "Python provenance package")
        original, lowered, binding = parse(package["manifest"], package["files"])
        require(canonical(original) == canonical(value["original"]), "Python original graph provenance")
    else:
        lowered, binding = decompose(value["original"],resource_profile=resource_profile)
    weight=value["decomposition"].get("weight_package")
    if weight is not None:
        require("python" not in value and "trusted_pytorch" not in value,"Weight provenance source format")
        from model_weight_package import check_recorded_binding
        weight=check_recorded_binding(weight,value["original"],resource_profile)
        binding["weight_package"]=weight
        binding.pop("binding_sha256",None);binding["binding_sha256"]=digest(binding)
    require(canonical(lowered) == canonical(model) and canonical(binding) == canonical(value["decomposition"]),
            "Model decomposition provenance mismatch")
    report = value["reference_report"]
    require(report.get("original_sha256") == digest(value["original"]) and
            report.get("model_sha256") == digest(model) and report.get("reference_status") == "passed" and
            report.get("approximation_accuracy_certified") is False, "Model reference provenance")
    return binding

def check_bundle(folder):
    folder = Path(folder)
    manifest = strict_file(folder/"manifest.json", 131072)
    require(type(manifest) is dict, "Bundle manifest")
    common = {"format", "files", "compiler_configuration", "origin", "agent_generated", "binding"}
    require(type(manifest.get("agent_generated")) is bool, "Bundle provenance flag")
    version = manifest.get("format")
    require(version in ("poseidon-dsl-bundle-v1", "poseidon-dsl-bundle-v2", "poseidon-dsl-bundle-v3", "poseidon-dsl-bundle-v4", "poseidon-dsl-bundle-v5", "poseidon-dsl-bundle-v6"), "Bundle manifest")
    require(set(manifest) == common | ({"dependencies", "model_provenance"} if not version.endswith("v1") else set()) | ({"validation"} if version.endswith(("v3","v4","v5","v6")) else set()), "Bundle manifest")
    require(digest({k:v for k,v in manifest.items() if k != "binding"}) == manifest["binding"], "Bundle binding")
    allowed = FILES if version.endswith("v1") else V2_FILES | ({PROVENANCE} if manifest["model_provenance"] == "included" else set())
    if version.endswith(("v4","v5","v6")):
        extra=set(manifest["files"])-allowed
        require(extra and all(n.startswith("artifacts/") and ".." not in Path(n).parts and
                not Path(n).is_absolute() for n in extra),"Runtime artifact paths")
        allowed |= extra
    require(set(manifest["files"]) == allowed, "Unexpected executable bundle files")
    data_limit=4*1024**2 if version.endswith("v6") else 131072
    for name, expected in manifest["files"].items():
        p = folder/name
        require(p.resolve().is_relative_to(folder.resolve()) and not p.is_symlink() and p.is_file() and p.stat().st_size <= (16*1024**2 if name.startswith("artifacts/") else data_limit if name in ("model.json","request.json",PROVENANCE) else 131072) and sha(p) == expected,
                "Bundle file integrity")
    require(canonical(manifest["compiler_configuration"]) == canonical(configuration(manifest["compiler_configuration"]["name"])),
            "Compiler profile changed")
    if version.endswith("v3"):
        require(manifest["validation"] == dict(level="compiled", encrypted_execution=False,
                numerically_validated=False), "Compiled bundle grade")
    if version.endswith(("v4","v5","v6")):
        validation=manifest["validation"]
        require(validation in (dict(level="compiled",encrypted_execution=False,numerically_validated=False),
                               dict(level="numerical",encrypted_execution=True,numerically_validated=True)),
                "Poseidon bundle validation grade")
    if not version.endswith("v1"):
        require(manifest["model_provenance"] in ("included", "not_supplied"), "Model provenance status")
        request = strict_file(folder/"request.json", data_limit)
        reconstruct_request(request)
        require(("public_weight_binding" in request)==version.endswith("v6"),"Package weight policy/version")
        require(canonical(strict_file(folder/"model.json", data_limit)) == canonical(request["model"]), "Bundle model/request")
        require(request["request_id"] == manifest["origin"]["request_id"], "Bundle original request identity")
        expected = request.get("compiler_configuration") or configuration("seal-cpu-eva-w40-v1")
        require(canonical(expected) == canonical(manifest["compiler_configuration"]), "Bundle request configuration")
        deps = manifest["dependencies"]
        require(set(deps) == {"compiler_sha256", "helper_environment", "implementation", "validation_sources"} | ({"backend_identity"} if version.endswith(("v4","v5","v6")) else set()) | ({"application_identities"} if version.endswith(("v5","v6")) else set()), "Bundle dependency fields")
        if version.endswith(("v5","v6")):
            package_identities(folder, manifest)
        require(type(deps["compiler_sha256"]) is str and len(deps["compiler_sha256"]) == 64, "Compiler binary identity")
        expected_helper = "upstream_helpers" in request
        require((deps["helper_environment"] is not None) == expected_helper, "Helper dependency binding")
        if PROVENANCE in allowed:
            check_provenance(strict_file(folder/PROVENANCE, data_limit), request["model"])
    generation=manifest["origin"].get("generation_provenance")
    if generation is not None:
        from generation_receipt import FILE,MAX_BYTES,verify
        require(version in ("poseidon-dsl-bundle-v5","poseidon-dsl-bundle-v6") and manifest["agent_generated"] is True and
                type(generation) is dict and set(generation)=={"file","sha256"} and
                generation["file"]==FILE and manifest["files"].get(FILE)==generation["sha256"],
                "Package generation descriptor")
        identities=package_identities(folder,manifest)
        verify(strict_file(folder/FILE,MAX_BYTES),request,
               (folder/"candidate.py").read_bytes().decode("utf-8"),
               generation_identity_sha256=identities["generation"]["sha256"])
    else:
        from generation_receipt import FILE
        require(FILE not in manifest["files"],"Unreferenced generation evidence")
    from model_source_provenance import FILE as SOURCE_FILE,check_package as check_model_source
    if not version.endswith("v1"):
        source_provenance=strict_file(folder/PROVENANCE,data_limit) if PROVENANCE in allowed else None
        check_model_source(folder,manifest,request,source_provenance)
    else:
        require("model_source" not in manifest["origin"] and SOURCE_FILE not in manifest["files"],
                "Historical bundle cannot claim model source intake")
    return manifest

def package_identities(folder, manifest):
    from application_identity import check_manifest
    value = manifest["dependencies"]["application_identities"]
    require(type(value) is dict and set(value) == {"file", "sha256"} and
            value["file"] == "artifacts/application-identities.json" and
            manifest["files"].get(value["file"]) == value["sha256"], "Package identity descriptor")
    path = Path(folder)/value["file"]
    require(not path.is_symlink() and sha(path) == value["sha256"], "Package identity integrity")
    return check_manifest(strict_file(path, 16*1024**2))

def implementation():
    root = Path(__file__).resolve().parent
    names = ("numerical_results.py", "poseidon_sandbox_policy.py", "process_resource_limits.py", "candidate_bundle.py", "component_contract.py", "component_backend.py",
             "component_worker.py", "validation_adapter.py", "capability_combinations.py", "construction_evidence.py", "unified_graph_contract.py", "unified_poseidon_guidance.py", "unified_binding_guidance.py", "unified_output_binding_guidance.py", "candidate_check_diagnostic.py", "unified_public_contract.py",
             "model_decomposition.py", "component_control.py", "audit_unified_candidate.py",
             "agent_response_provenance.py", "restricted_model_python.py", "trusted_pytorch_import.py", "pytorch_model_capture.py", "poseidon_artifact_gate.py",
             "poseidon_validation.py", "poseidon_runtime_compile.py", "compiler_configuration.py", "poseidon_backend_identity.py", "poseidon_candidate_sandbox.py", "poseidon_package.py")
    return {name:sha(root/name) for name in names}

def audit_provenance(value, model):
    check_provenance(value, model)
    import numpy as np
    from benchmark_graph import samples,graph_resource_profile
    resource_profile=graph_resource_profile(model)
    from operator_arithmetic_graph import evaluate_reference as high
    from benchmark_math import evaluate as core
    from benchmark_torch import evaluate as torch_ref
    for inputs in samples(model, 16):
        expected, actual, independent = high(value["original"], inputs), core(model, inputs,resource_profile=resource_profile), torch_ref(model, inputs,resource_profile=resource_profile)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(independent[name], expected[name], atol=1e-12, rtol=1e-12)

def load_model_provenance(model_import, model, *, max_bytes=131072):
    require(max_bytes in (131072,4*1024**2),"Provenance size policy")
    model_import=Path(model_import)
    value = {k:strict_file(model_import/n, max_bytes) for k,n in (
        ("original","original.json"),("decomposition","decomposition.json"),("reference_report","reference-report.json"))}
    python_manifest = model_import/"python/manifest.json"
    if python_manifest.exists():
        package = strict_file(python_manifest, 131072)
        files = {}
        for name in package.get("files", {}):
            rel = Path(name)
            require(not rel.is_absolute() and len(rel.parts) == 1 and rel.suffix == ".py" and rel.stem.isidentifier(),
                    "Python provenance path")
            file = model_import/"python"/rel
            require(file.is_file() and not file.is_symlink() and file.stat().st_size <= 65536, "Python provenance file")
            files[name] = file.read_bytes().decode("utf-8")
        value["python"] = dict(manifest=package, files=files)
    trusted_manifest=model_import/"trusted-python/manifest.json"
    if trusted_manifest.exists():
        from pytorch_model_intake import package_format,validate_package
        package=strict_file(trusted_manifest,131072);files={}
        from pytorch_source_package_policy import selected as packaged_source
        if packaged_source(package):
            from pytorch_ordered_contract import source_package_files
            files = {name:raw.decode("utf-8") for name,raw in
                     source_package_files(package,model_import/"trusted-python").items()}
        else:
            for name in package.get("files",{}):
                rel=Path(name)
                require(not rel.is_absolute() and len(rel.parts)==1 and rel.suffix==".py" and rel.stem.isidentifier(),
                        "Trusted Python provenance path")
                file=model_import/"trusted-python"/rel
                require(file.is_file() and not file.is_symlink() and file.stat().st_size<=65536,"Trusted Python provenance file")
                files[name]=file.read_bytes().decode("utf-8")
        validate_package(package,files)
        from pytorch_source_contract import STATE_FORMATS
        from pytorch_model_intake import SOURCE_FORMATS
        if package["format"] in (*STATE_FORMATS,*SOURCE_FORMATS):
            from pytorch_weight_package import read,binding
            from benchmark_graph import graph_resource_profile
            arrays,raw_files=read(model_import/"trusted-python",package["weights"],
                          resource_profile=graph_resource_profile(model))
            require(binding(arrays,package["weights"])==value["reference_report"]["capture_metadata"]["weight_binding"],
                    "Retained source weights differ from capture")
        value["trusted_pytorch"]=dict(format=package_format(package),manifest=package,files=files)
        from pytorch_ordered_contract import SOURCE_FORMATS as ORDERED,encode_weight_files
        if package["format"] in ORDERED:
            value["trusted_pytorch"]["weight_files"]=encode_weight_files(
                package,raw_files,resource_profile=graph_resource_profile(model))
    check_provenance(value,model)
    require(len(canonical(value))<=max_bytes,"Model provenance size budget")
    return value

def serialize_model_provenance(value, limit, *, compact_fallback=False):
    """Keep source declaration order; the manifest binds these exact bytes."""
    import json
    require(type(value) is dict and type(limit) is int and limit > 0 and type(compact_fallback) is bool,
            "Model provenance serialization boundary")
    # Canonical identity remains independent of representation. Ordered source
    # reconstruction also needs the original constant declaration sequence.
    require(len(canonical(value)) <= limit, "Model provenance size budget")
    raw = json.dumps(value, indent=2, sort_keys=False, allow_nan=False).encode()+b"\n"
    if len(raw)>limit and compact_fallback:
        raw=json.dumps(value,sort_keys=False,separators=(",",":"),allow_nan=False).encode()
        if len(raw)<limit:raw+=b"\n"
    require(len(raw) <= limit, "Serialized model provenance size budget")
    return raw


def export_bundle(evidence, output, *, model_import=None, model_provenance=None, live_approval=None, validation_level="numerical", application_identities=None, generation_provenance=None, model_source=None):
    from workspace_paths import RESULTS
    from audit_unified_candidate import verify_candidate
    from poseidon_runtime_paths import BUILD
    require(not output.exists(), "Preserve existing export")
    require(evidence.resolve().is_relative_to(RESULTS.resolve()), "Platform evidence required")
    audit = verify_candidate(evidence, live_approval=live_approval, validation_level=validation_level)
    report = strict_file(evidence/"report.json", 8*1024**2)
    request = strict_file(evidence/"request.json", 4*1024**2)
    reconstruct_request(request)
    model = strict_file(evidence/"model.json", 4*1024**2)
    from public_weight_context import storage_limit
    data_limit=storage_limit(request)
    require(digest(model) == audit["model_sha256"], "Model identity")
    attempt = next(a for a in report["attempts"] if a.get("status") == "passed")
    source = evidence/("attempt-%02d" % attempt["index"])/"candidate.py"
    if generation_provenance is not None:
        require(live_approval is None and application_identities is not None and report.get("backend")=="poseidon_cpu",
                "Application generation evidence requires a staged Poseidon package")
        from generation_receipt import verify
        verify(generation_provenance,request,source.read_bytes().decode("utf-8"),
               generation_identity_sha256=application_identities["generation"]["sha256"])
    require(report.get("compiler_sha256") == sha(BUILD/"bin/hecate-opt"), "Export needs recorded compiler identity; requalify old evidence")
    require(model_import is None or model_provenance is None,"Ambiguous model provenance source")
    value = model_provenance
    if model_import is not None:
        value=load_model_provenance(model_import,model,max_bytes=data_limit)
    provenance_bytes = None
    if value is not None:
        audit_provenance(value, model)
        from public_weight_context import COMPACT_POLICY
        compact=request.get("public_weight_binding",{}).get("policy")==COMPACT_POLICY
        provenance_bytes = serialize_model_provenance(value, data_limit,compact_fallback=compact)
    output.mkdir(parents=True)
    dump(output/"model.json", model)
    (output/"request.json").write_bytes(request_file_bytes(request))
    (output/"candidate.py").write_bytes(source.read_bytes())
    files = set(V2_FILES)
    if report.get("backend")=="poseidon_cpu":
        import shutil
        out=evidence/("attempt-%02d"%attempt["index"])/"output"
        artifacts=output/"artifacts";artifacts.mkdir()
        trace_files=("candidate_trace.mlir","lowered.earth.mlir","lowered.ckks.mlir","lowered._hecate_golden.runtime-plan.json")
        if request["compiler_configuration"]["schema"] in (3,4):
            trace_files+=("candidate_trace.entry.mlir","entry-selection.json")
        if request["compiler_configuration"]["schema"]==4:
            trace_files+=("candidate_trace.compact.mlir",)
        for name in trace_files:
            shutil.copyfile(out/name,artifacts/name)
        for name in ("bound","lowered._hecate_golden.bundle"):
            if (out/name).exists():shutil.copytree(out/name,artifacts/name)
        from poseidon_runtime_paths import SPEC,PROFILE
        shutil.copyfile(SPEC,artifacts/"operator-spec.json")
        shutil.copyfile(PROFILE,artifacts/"compiler-profile.json")
        entry = dict(runtime_plan_entry=attempt["runtime_plan_entry"],compile_command=attempt["compile_command"]) if request["compiler_configuration"]["schema"] in (3,4) else {}
        dump(artifacts/"validation.json",dict(level=validation_level,scope="four fixed inputs",gate=attempt["artifact_gate"],**entry))
        dump(artifacts/"dependencies.json",report["backend_identity"])
        files |= {str(p.relative_to(output)) for p in artifacts.rglob("*") if p.is_file()}
    if value is not None:
        (output/PROVENANCE).write_bytes(provenance_bytes); files.add(PROVENANCE)
    manifest = dict(format="poseidon-dsl-bundle-v2", files={n:sha(output/n) for n in sorted(files)},
        compiler_configuration=request.get("compiler_configuration") or configuration("seal-cpu-eva-w40-v1"),
        origin=dict(report_sha256=sha(evidence/"report.json"), request_id=request["request_id"],
                    frontend_sha256=report["frontend_sha256"], runtime_sha256=report["runtime_sha256"],
                    platform=report["platform_identity"]),
        dependencies=dict(compiler_sha256=report["compiler_sha256"],
                          helper_environment=report.get("upstream_helper_environment"), implementation=implementation(),
                          validation_sources=report["source_hashes"]),
        model_provenance="included" if value is not None else "not_supplied",
        agent_generated=live_approval is not None)
    if validation_level == "compiled":
        manifest["format"] = "poseidon-dsl-bundle-v3"
        manifest["validation"] = dict(level="compiled", encrypted_execution=False, numerically_validated=False)
    if report.get("backend")=="poseidon_cpu":
        manifest["format"]="poseidon-dsl-bundle-v4"
        manifest["dependencies"]["backend_identity"]=report["backend_identity"]
        manifest["validation"]=dict(level=validation_level,encrypted_execution=validation_level=="numerical",
                                    numerically_validated=validation_level=="numerical")
    if application_identities is not None:
        from application_identity import check_manifest, current_identity_manifest
        require(manifest["format"] == "poseidon-dsl-bundle-v4", "Staged identity requires Poseidon artifacts")
        check_manifest(application_identities)
        require(application_identities == current_identity_manifest(), "Application identity changed before export")
        manifest["format"] = "poseidon-dsl-bundle-v5"
        name = "artifacts/application-identities.json"
        (output/name).write_bytes(canonical(application_identities))
        manifest["files"][name] = sha(output/name)
        manifest["dependencies"]["application_identities"] = dict(file=name, sha256=sha(output/name))
    if "public_weight_binding" in request:
        require(manifest["format"]=="poseidon-dsl-bundle-v5","Descriptor weights require staged package identity")
        manifest["format"]="poseidon-dsl-bundle-v6"
    if generation_provenance is not None:
        from generation_receipt import FILE
        (output/FILE).write_bytes(canonical(generation_provenance))
        manifest["files"][FILE]=sha(output/FILE)
        manifest["origin"]["generation_provenance"]=dict(file=FILE,sha256=sha(output/FILE))
        manifest["agent_generated"]=True
    if model_source is not None:
        from model_source_provenance import FILE as SOURCE_FILE,FORMAT as SOURCE_FORMAT,verify as verify_source
        require(manifest["format"] in ("poseidon-dsl-bundle-v5","poseidon-dsl-bundle-v6"),
                "Model source requires staged Poseidon package identity")
        verify_source(model_source,request,value,
                      controller_sha256=application_identities["generation"]["sources"].get("application_model_source.py"))
        (output/SOURCE_FILE).write_bytes(canonical(model_source))
        manifest["files"][SOURCE_FILE]=sha(output/SOURCE_FILE)
        manifest["origin"]["model_source"]=dict(format=SOURCE_FORMAT,file=SOURCE_FILE,sha256=sha(output/SOURCE_FILE))
    manifest["binding"] = digest(manifest)
    dump(output/"manifest.json", manifest)
    (output/"README.md").write_text(
        ("Poseidon RuntimePlan DSL bundle: compilation acceptance.\n"
         if validation_level == "compiled" else "Poseidon RuntimePlan DSL bundle: numerical acceptance on four fixed inputs.\n") +
        "request.json preserves the complete public contract, helper and layout bindings.\n"
        "No credentials, secret keys, private test inputs or reference arrays are included.\n"
        "Run scripts/dsl_bundle.py replay --bundle <directory> --execute in a matching SDK.\n"
        "Default replay only checks integrity; it does not prove execution or provenance.\n"
        "Exact replay refuses different SDKs; cross-platform requalification is not implicit.\n")
    check_bundle(output)
    return manifest

def check_execution_dependencies(folder):
    folder = Path(folder)
    manifest = check_bundle(folder)
    require(manifest["format"] in ("poseidon-dsl-bundle-v4","poseidon-dsl-bundle-v5", "poseidon-dsl-bundle-v6") and
            manifest["compiler_configuration"]["backend"]=="poseidon_cpu",
            "Historical SEAL packages are read-only; explicit Poseidon requalification is required")
    from poseidon_runtime_paths import BUILD
    from poseidon_backend_identity import identity
    require(identity()==manifest["dependencies"]["backend_identity"],"Poseidon package dependency mismatch")
    deps = manifest["dependencies"]
    if manifest["format"] in ("poseidon-dsl-bundle-v5","poseidon-dsl-bundle-v6"):
        from application_identity import compatible, current_identity_manifest
        require(compatible(package_identities(folder, manifest), current_identity_manifest()),
                "Bundle stage identity changed; needs explicit requalification")
    else:
        require(deps["implementation"] == implementation(), "Bundle implementation changed; needs explicit requalification")
        from audit_unified_candidate import verify_files
        verify_files(Path(__file__).resolve().parent, deps["validation_sources"])
    require(sha(BUILD/"bin/hecate-opt") == deps["compiler_sha256"], "Bundle compiler binary mismatch")
    raw_request = (folder/"request.json").read_bytes()
    raw_source = (folder/"candidate.py").read_bytes()
    require(hashlib.sha256(raw_request).hexdigest() == manifest["files"]["request.json"] and
            hashlib.sha256(raw_source).hexdigest() == manifest["files"]["candidate.py"], "Bundle changed before execution")
    from candidate_contract import strict_json
    # Storage codecs are reversible representations, not the execution envelope.
    # Reconstruct and verify the original hash before any artifact audit or runtime call.
    from public_weight_context import read_request
    request = reconstruct_request(read_request(raw_request.decode()))
    if deps["helper_environment"] is not None:
        from upstream_candidate_helpers import verify_sources
        from poly_dependencies import verify
        require(dict(sources=verify_sources(backend="poseidon_cpu"), dependency=verify()) == deps["helper_environment"], "Bundle helper environment mismatch")
    if manifest["model_provenance"] == "included":
        from public_weight_context import storage_limit
        audit_provenance(strict_file(folder/PROVENANCE, storage_limit(request)), request["model"])
    return manifest, request

def replay_bundle(folder):
    manifest, request = check_execution_dependencies(folder)
    from poseidon_package import replay
    return replay(Path(folder),manifest)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("export")
    e.add_argument("--evidence", type=Path, required=True); e.add_argument("--output", type=Path, required=True)
    e.add_argument("--model-import", type=Path)
    e.add_argument("--identity-policy", choices=("legacy-v1","staged-v1"), default="legacy-v1")
    e.add_argument("--live-approval", type=Path, help="Existing exact paid configuration for provenance audit; never makes API calls")
    r = sub.add_parser("replay")
    r.add_argument("--bundle", type=Path, required=True); r.add_argument("--execute", action="store_true")
    x = sub.add_parser("execute")
    x.add_argument("--bundle", type=Path, required=True)
    x.add_argument("--inputs", type=Path, required=True,
                   help="poseidon-local-inputs-v1 JSON with 1..16 named samples")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.mode == "replay":
        manifest = check_bundle(a.bundle)
        if not a.execute:
            print(json.dumps(dict(mode="replay_plan", binding=manifest["binding"], paid_calls=0, encrypted_execution=False)))
            return 0
    from hecate_python_env import enter_nix, VENV, ROOT
    pinned = os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV
    if not pinned:
        require(not a.inside, "Pinned pure environment")
        # Resolve host paths before entering the shell; no cwd dependence in worker use.
        forwarded = []
        for value in sys.argv[1:]:
            forwarded.append(value)
        cmd = [str(VENV/"bin/python"), "-B", str(ROOT/"scripts/dsl_bundle.py"), "--inside", *forwarded]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(cmd), seconds=930, environment="poseidon-cpu")
    if a.mode == "export":
        approval = strict_file(a.live_approval, 131072) if a.live_approval is not None else None
        identities = None
        if a.identity_policy == "staged-v1":
            from application_identity import current_identity_manifest
            identities = current_identity_manifest()
        print(json.dumps(export_bundle(a.evidence, a.output, model_import=a.model_import,
                                     live_approval=approval, application_identities=identities), sort_keys=True))
        return 0
    if a.mode == "execute":
        inputs = strict_file(a.inputs, 4*1024**2)
        require(type(inputs) is dict and set(inputs) == {"format","samples"} and
                inputs["format"] == "poseidon-local-inputs-v1", "Local execution input format")
        from program_execution import execute_program
        from execution_input_domain import ApproximationDomainRejected
        try:
            result = execute_program(a.bundle, inputs["samples"])
        except ApproximationDomainRejected as exc:
            print(json.dumps(dict(status="validation_failed",failure=exc.public_failure,
                                  encrypted_execution=False,agent_calls=0,compiler_calls=0),sort_keys=True,allow_nan=False))
            return 1
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    return replay_bundle(a.bundle)
