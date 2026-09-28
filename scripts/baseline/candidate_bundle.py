"""Portable source manifests; exact SDK replay is separate from source integrity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from benchmark_graph import digest, canonical, require
from benchmark_runner import strict_file, dump
from compiler_configuration import configuration
from component_contract import reconstruct_request, runner_options

FILES = {"model.json", "candidate.py"}  # v1 compatibility
V2_FILES = FILES | {"request.json"}
PROVENANCE = "model-provenance.json"

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check_provenance(value, model):
    from model_decomposition import decompose
    require(type(value) is dict and {"original", "decomposition", "reference_report"} <= set(value) <=
            {"original", "decomposition", "reference_report", "python"}, "Model provenance fields")
    if "python" in value:
        from restricted_model_python import parse
        package = value["python"]
        require(type(package) is dict and set(package) == {"manifest", "files"}, "Python provenance package")
        original, lowered, binding = parse(package["manifest"], package["files"])
        require(canonical(original) == canonical(value["original"]), "Python original graph provenance")
    else:
        lowered, binding = decompose(value["original"])
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
    require(version in ("poseidon-dsl-bundle-v1", "poseidon-dsl-bundle-v2", "poseidon-dsl-bundle-v3"), "Bundle manifest")
    require(set(manifest) == common | ({"dependencies", "model_provenance"} if not version.endswith("v1") else set()) | ({"validation"} if version.endswith("v3") else set()), "Bundle manifest")
    require(digest({k:v for k,v in manifest.items() if k != "binding"}) == manifest["binding"], "Bundle binding")
    allowed = FILES if version.endswith("v1") else V2_FILES | ({PROVENANCE} if manifest["model_provenance"] == "included" else set())
    require(set(manifest["files"]) == allowed, "Unexpected executable bundle files")
    for name, expected in manifest["files"].items():
        p = folder/name
        require(not p.is_symlink() and p.is_file() and p.stat().st_size <= 131072 and sha(p) == expected,
                "Bundle file integrity")
    require(canonical(manifest["compiler_configuration"]) == canonical(configuration(manifest["compiler_configuration"]["name"])),
            "Compiler profile changed")
    if version.endswith("v3"):
        require(manifest["validation"] == dict(level="compiled", encrypted_execution=False,
                numerically_validated=False), "Compiled bundle grade")
    if not version.endswith("v1"):
        require(manifest["model_provenance"] in ("included", "not_supplied"), "Model provenance status")
        request = strict_file(folder/"request.json", 131072)
        reconstruct_request(request)
        require(canonical(strict_file(folder/"model.json", 131072)) == canonical(request["model"]), "Bundle model/request")
        require(request["request_id"] == manifest["origin"]["request_id"], "Bundle original request identity")
        expected = request.get("compiler_configuration") or configuration("seal-cpu-eva-w40-v1")
        require(canonical(expected) == canonical(manifest["compiler_configuration"]), "Bundle request configuration")
        deps = manifest["dependencies"]
        require(set(deps) == {"compiler_sha256", "helper_environment", "implementation", "validation_sources"}, "Bundle dependency fields")
        require(type(deps["compiler_sha256"]) is str and len(deps["compiler_sha256"]) == 64, "Compiler binary identity")
        expected_helper = "upstream_helpers" in request
        require((deps["helper_environment"] is not None) == expected_helper, "Helper dependency binding")
        if PROVENANCE in allowed:
            check_provenance(strict_file(folder/PROVENANCE, 131072), request["model"])
    return manifest

def implementation():
    root = Path(__file__).resolve().parent
    names = ("candidate_bundle.py", "component_contract.py", "component_backend.py",
             "component_worker.py", "validation_adapter.py", "capability_combinations.py", "construction_evidence.py", "unified_graph_contract.py", "unified_public_contract.py",
             "model_decomposition.py", "component_control.py", "audit_unified_candidate.py",
             "agent_response_provenance.py", "restricted_model_python.py")
    return {name:sha(root/name) for name in names}

def audit_provenance(value, model):
    check_provenance(value, model)
    import numpy as np
    from benchmark_graph import samples
    from operator_reference import evaluate as high
    from benchmark_math import evaluate as core
    from benchmark_torch import evaluate as torch_ref
    for inputs in samples(model, 16):
        expected, actual, independent = high(value["original"], inputs), core(model, inputs), torch_ref(model, inputs)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(independent[name], expected[name], atol=1e-12, rtol=1e-12)

def export_bundle(evidence, output, *, model_import=None, live_approval=None, validation_level="numerical"):
    from workspace_paths import RESULTS
    from audit_unified_candidate import verify_candidate
    from python_compiler_smoke import BUILD
    require(not output.exists(), "Preserve existing export")
    require(evidence.resolve().is_relative_to(RESULTS.resolve()), "Platform evidence required")
    audit = verify_candidate(evidence, live_approval=live_approval, validation_level=validation_level)
    report = strict_file(evidence/"report.json", 8*1024**2)
    request = strict_file(evidence/"request.json", 131072)
    reconstruct_request(request)
    model = strict_file(evidence/"model.json", 131072)
    require(digest(model) == audit["model_sha256"], "Model identity")
    attempt = next(a for a in report["attempts"] if a.get("status") == "passed")
    source = evidence/("attempt-%02d" % attempt["index"])/"candidate.py"
    require(report.get("compiler_sha256") == sha(BUILD/"bin/hecate-opt"), "Export needs recorded compiler identity; requalify old evidence")
    value = None
    if model_import is not None:
        value = {k:strict_file(model_import/n, 131072) for k,n in (
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
                files[name] = file.read_text()
            value["python"] = dict(manifest=package, files=files)
        audit_provenance(value, model)
        require(len(canonical(value)) <= 131072, "Model provenance size budget")
    output.mkdir(parents=True)
    dump(output/"model.json", model)
    dump(output/"request.json", request)
    (output/"candidate.py").write_bytes(source.read_bytes())
    files = set(V2_FILES)
    if value is not None:
        dump(output/PROVENANCE, value); files.add(PROVENANCE)
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
    manifest["binding"] = digest(manifest)
    dump(output/"manifest.json", manifest)
    (output/"README.md").write_text(
        ("Poseidon single-source DSL bundle v3: compilation acceptance ONLY; no numerical claim.\n"
         if validation_level == "compiled" else "Poseidon single-source DSL bundle v2: numerical acceptance.\n") +
        "request.json preserves the complete public contract, helper and layout bindings.\n"
        "No credentials, secret keys, private test inputs or reference arrays are included.\n"
        "Run scripts/dsl_bundle.py replay --bundle <directory> --execute in a matching SDK.\n"
        "Default replay only checks integrity; it does not prove execution or provenance.\n"
        "Exact replay refuses different SDKs; cross-platform requalification is not implicit.\n")
    check_bundle(output)
    return manifest

def replay_bundle(folder):
    manifest = check_bundle(folder)
    from run_candidate import parse_args, inside
    from python_compiler_smoke import BUILD
    for key, name in (("frontend_sha256","libHecateFrontend.so"),("runtime_sha256","libSEAL_HEVM.so")):
        require(sha(BUILD/"lib"/name) == manifest["origin"][key], "Bundle SDK binary identity mismatch")
    if manifest["format"].endswith("v1"):
        args = parse_args(["--inside", "--case", str(folder/"model.json"), "--golden-file", str(folder/"candidate.py"),
                          "--max-repairs", "0", "--compiler-configuration", manifest["compiler_configuration"]["name"]])
        return inside(args)
    deps = manifest["dependencies"]
    require(deps["implementation"] == implementation(), "Bundle implementation changed; needs explicit requalification")
    from audit_unified_candidate import verify_files
    verify_files(Path(__file__).resolve().parent, deps["validation_sources"])
    require(sha(BUILD/"bin/hecate-opt") == deps["compiler_sha256"], "Bundle compiler binary mismatch")
    raw_request = (folder/"request.json").read_bytes()
    raw_source = (folder/"candidate.py").read_bytes()
    require(hashlib.sha256(raw_request).hexdigest() == manifest["files"]["request.json"] and
            hashlib.sha256(raw_source).hexdigest() == manifest["files"]["candidate.py"], "Bundle changed before execution")
    from candidate_contract import strict_json
    request = strict_json(raw_request.decode())
    if deps["helper_environment"] is not None:
        from upstream_candidate_helpers import verify_sources
        from poly_dependencies import verify
        require(dict(sources=verify_sources(), dependency=verify()) == deps["helper_environment"], "Bundle helper environment mismatch")
    if manifest["model_provenance"] == "included":
        audit_provenance(strict_file(folder/PROVENANCE, 131072), request["model"])
    from component_backend import qualify
    result = qualify(request, dict(schema=1, request_id=request["request_id"], hecate_source=raw_source.decode()),
                     validation_level=manifest.get("validation", {}).get("level", "numerical"))
    result["bundle_binding"] = manifest["binding"]
    result["origin_agent_generated"] = manifest["agent_generated"]
    # A replay is never a new model generation, regardless of original provenance.
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("export")
    e.add_argument("--evidence", type=Path, required=True); e.add_argument("--output", type=Path, required=True)
    e.add_argument("--model-import", type=Path)
    e.add_argument("--live-approval", type=Path, help="Existing exact paid configuration for provenance audit; never makes API calls")
    r = sub.add_parser("replay")
    r.add_argument("--bundle", type=Path, required=True); r.add_argument("--execute", action="store_true")
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
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(cmd), seconds=930)
    if a.mode == "export":
        approval = strict_file(a.live_approval, 131072) if a.live_approval is not None else None
        print(json.dumps(export_bundle(a.evidence, a.output, model_import=a.model_import, live_approval=approval), sort_keys=True))
        return 0
    return replay_bundle(a.bundle)
