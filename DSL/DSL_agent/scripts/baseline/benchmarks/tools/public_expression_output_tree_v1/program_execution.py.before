"""Local host execution of a checked program package on caller-supplied inputs.

This API is for trusted host code inside the pinned worker. It never generates,
compiles or treats previously unseen inputs as numerically certified.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from benchmark_graph import require, validate, canonical
from cipher_abi import has_zero_argument
from poseidon_validation import input_payload

FORMAT = "poseidon-local-execution-v1"

def normalize_samples(request, samples):
    """Validate and copy the caller inputs into logical float64 snapshots."""
    model = request["model"]
    checked = validate(model)
    specs = model["inputs"]
    require(type(samples) is list and 1 <= len(samples) <= 16, "Execution requires 1..16 samples")
    names = {s["name"] for s in specs}
    normalized = []
    for sample in samples:
        require(type(sample) is dict and set(sample) == names, "Named input signature mismatch")
        logical = {}
        for spec in specs:
            value = sample[spec["name"]]
            require(type(value) in (list, np.ndarray), "Input must be a numeric tensor")
            if type(value) is np.ndarray:
                require(value.dtype.kind in "fiu" and value.size == math.prod(spec["shape"]),
                        "Input dtype or element count")
            if type(value) is list:
                def check_list(item, shape):
                    if shape:
                        require(type(item) is list and len(item) == shape[0], "Input list shape")
                        for child in item:
                            check_list(child, shape[1:])
                    else:
                        require(type(item) in (int, float), "Input list numeric leaf")
                        if type(item) is int:
                            require(-(2**53) <= item <= 2**53, "Integer input precision")
                        require(math.isfinite(item), "Input contains nonfinite values")
                check_list(value, spec["shape"])
            array = np.asarray(value)
            require(array.dtype.kind in "fiu" and list(array.shape) == spec["shape"], "Input shape or dtype")
            require(np.isfinite(array).all(), "Input contains nonfinite values")
            # Preserve integer values exactly in the float64 computation domain.
            if array.dtype.kind in "iu":
                require(all(-(2**53) <= int(v) <= 2**53 for v in array.flat), "Integer input precision")
            array = array.astype(np.float64)
            require(np.isfinite(array).all(), "Input exceeds float64 domain")
            logical[spec["name"]] = array
        normalized.append(logical)
    return normalized

def _pack_logical_samples(request, logical_samples):
    """Pack already validated snapshots using the fixed physical binding."""
    layout = request["layout"]
    period = layout["input_slot_period"]
    packed = []
    for logical in logical_samples:
        vectors = []
        for binding in layout["inputs"]:
            flat = logical[binding["name"]].reshape(-1)
            start = binding.get("offset", 0)
            count = binding.get("elements", len(flat))
            require(0 <= start < len(flat) and 1 <= count <= period and start + count <= len(flat),
                    "Physical input slice")
            vectors.append(np.pad(flat[start:start+count], (0, period-count)))
        if has_zero_argument(layout):
            vectors.append(np.zeros(period, dtype=np.float64))
        packed.append(vectors)
    return np.asarray(packed, dtype=np.float64)


def pack_samples(request, samples):
    """Keep the existing packing interface and validation semantics."""
    return _pack_logical_samples(request, normalize_samples(request, samples))

def named_outputs(request, flat, sample_count):
    checked = validate(request["model"])
    specs = request["model"]["outputs"]
    total = sum(math.prod(checked["shapes"][s["value"]]) for s in specs)
    flat = np.asarray(flat)
    require(flat.shape == (sample_count, total) and np.isfinite(flat).all(), "Output signature mismatch")
    result = []
    for row in flat:
        cursor = 0
        values = {}
        for spec in specs:
            shape = checked["shapes"][spec["value"]]
            count = math.prod(shape)
            values[spec["name"]] = row[cursor:cursor+count].reshape(shape).tolist()
            cursor += count
        result.append(values)
    return result

def package_execution_metadata(folder,manifest,model):
    """Read the bound original graph and output structure from one snapshot."""
    if manifest.get("model_provenance")!="included":
        return dict(original_model=model,output_structure=None)
    from candidate_bundle import PROVENANCE,check_provenance
    from model_weight_package import regular_bytes
    from candidate_contract import strict_json
    from pytorch_source_contract import EXTENDED
    import hashlib
    max_bytes=4*1024**2 if manifest.get("format")=="poseidon-dsl-bundle-v6" else 131072
    raw=regular_bytes(Path(folder)/PROVENANCE,max_bytes)
    require(hashlib.sha256(raw).hexdigest()==manifest["files"][PROVENANCE],
            "Package output provenance changed")
    from public_weight_context import parse_data
    provenance=parse_data(raw.decode())
    check_provenance(provenance,model)
    package=provenance.get("trusted_pytorch")
    structure = None
    from pytorch_model_intake import ordered
    if package is not None and (package["manifest"]["format"] in EXTENDED
                                or ordered(package["manifest"])):
        structure = copy.deepcopy(provenance["reference_report"]["capture_metadata"]["output_structure"])
    result=dict(original_model=provenance["original"],output_structure=structure)
    if package is not None and package["manifest"]["format"]=="poseidon-trusted-pytorch-v16":
        result["output_structure_policy"]="poseidon-mha-ordered-source-v1"
    return result


def package_output_structure(folder,manifest,model):
    """Compatibility interface for callers reading the output tree."""
    return package_execution_metadata(folder,manifest,model)["output_structure"]

def execute_program(folder, samples):
    """Execute 1..16 named samples; package verification remains mandatory."""
    from candidate_bundle import check_execution_dependencies
    from hecate_python_env import WORK, VENV, digest
    from poseidon_backend_identity import identity
    from poseidon_package import copy_checked_artifacts
    from poseidon_runtime_compile import audit_entrypoint
    from poseidon_validation import audit_gate, execute
    from poseidon_candidate_sandbox import run
    from native_execution_slots import native_slot
    from candidate_contract import trace_payload_json
    import tempfile
    import os,sys
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV, "Pinned execution worker required")
    folder = Path(folder)
    manifest, request = check_execution_dependencies(folder)
    metadata = package_execution_metadata(folder, manifest, request["model"])
    structure = metadata["output_structure"]
    logical = normalize_samples(request, samples)
    from execution_input_domain import check_approximation_domains
    domain_check = check_approximation_domains(metadata["original_model"], logical)
    packed = _pack_logical_samples(request, logical)
    start_identity = identity()
    evidence = Path(tempfile.mkdtemp(prefix="poseidon-program-execution-", dir=WORK/"results"))
    output = evidence/"output"
    output.mkdir()
    copy_checked_artifacts(folder, manifest, output)
    record = json.loads((output/"validation.json").read_text())
    audit_entrypoint(output, request, record.get("runtime_plan_entry"), record.get("compile_command"))
    audit_gate(output, request, record["gate"])
    from application_planning.compiled_resources import execution_storage_precheck
    resource_precheck = execution_storage_precheck(output, request)
    candidate = dict(schema=1, request_id=request["request_id"],
                     hecate_source=(folder/"candidate.py").read_text())
    payload = evidence/"payload.json"
    payload.write_text(trace_payload_json(request, candidate))
    (output/"inputs.json").write_bytes(canonical(input_payload(packed, request["layout"])))
    frozen = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    payload_hash = digest(payload)
    with native_slot(WORK/"cache/agent-native-slots"):
        execution = execute(payload, output, request, run, evidence/"execute.log")
    require(all(digest(output/k) == v for k,v in frozen.items()) and digest(payload) == payload_hash,
            "Execution inputs or artifacts changed")
    final_manifest, _ = check_execution_dependencies(folder)
    require(final_manifest == manifest and identity() == start_identity, "Execution dependency drift")
    outputs = named_outputs(request, np.load(output/"decrypted.npy", allow_pickle=False), len(logical))
    report = dict(format=FORMAT, status="executed", backend="poseidon_cpu",
                  bundle_binding=manifest["binding"], request_id=request["request_id"],
                  package_validation=copy.deepcopy(manifest["validation"]),
                  encrypted_execution=True, executor_secret_key=False,
                  numerically_validated_on_supplied_inputs=False,
                  application_accuracy_certified=False,
                  validation_scope="Supplied inputs executed; no reference comparison requested",
                  samples=len(logical), outputs=outputs, execution=execution,
                  resource_precheck=resource_precheck,
                   input_domain_check=copy.deepcopy(domain_check),
                  agent_calls=0, compiler_calls=0, evidence=str(evidence))
    if structure is not None:
        from pytorch_source_contract import restore_outputs
        report.update(output_structure=structure,
                      structured_outputs=[restore_outputs(value,structure,**({"allow_none":True} if
                           metadata.get("output_structure_policy")=="poseidon-mha-ordered-source-v1" else {})) for value in outputs])
    (evidence/"report.json").write_bytes(canonical(report))
    return report
