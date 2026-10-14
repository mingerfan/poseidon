"""Data-only audit of one joint-706 offline case; never imports the SDK.

This binds persisted compilation/execution receipts and recomputes their numerical
comparisons. Reference semantics remain those of the frozen, reviewed runner;
this is not a second mathematical/PyTorch evaluation or proof of application accuracy.
No private numerical values are returned or printed.
"""
import ast
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import runtime_scope as scope

class AuditProfile(NamedTuple):
    format: str
    cases: tuple
    phases: tuple
    origin: str
    strategy: str
    frozen_candidates: bool


# These are audit contracts, not runtime scopes or user-selectable overrides.
LEGACY_PROFILE = AuditProfile(
    "poseidon-common-fused-offline-plan-v1",
    ("common_alpha", "common_dropout", "common_boolean",
     "addmm_cipher", "addmm_public_right", "addmm_public_bias",
     "baddbmm_cipher", "baddbmm_public_right", "baddbmm_public_bias",
     "addbmm_cipher", "addbmm_public_right", "addbmm_public_bias"),
    ("qualification", "new_inputs", "cache"), "manual_selected_generic_balanced_scalar",
    "lower-balanced-scalar-v1", False)
CAPACITY_PROFILE = AuditProfile(
    "poseidon-common-fused-capacity-offline-plan-v1",
    ("common_alpha", "common_dropout",
     "addmm_cipher", "addmm_public_right", "addmm_public_bias",
     "baddbmm_cipher", "baddbmm_public_right", "baddbmm_public_bias",
     "addbmm_cipher", "addbmm_public_right", "addbmm_public_bias"),
    ("qualification", "new_inputs", "cache"), "rule_baseline_layout_aware_capacity",
    "poseidon-generic-candidate-dispatch-v2", True)
AUDIT_PROFILES = MappingProxyType({p.format: p for p in (LEGACY_PROFILE, CAPACITY_PROFILE)})


def audit_profile(plan):
    require(type(plan) is dict and type(plan.get("format")) is str
            and plan["format"] in AUDIT_PROFILES, "recognized_audit_profile")
    profile = AUDIT_PROFILES[plan["format"]]
    require(plan.get("cases") == list(profile.cases) and plan.get("phases") == list(profile.phases)
            and plan.get("candidate_origin") == profile.origin
            and plan.get("candidate_strategy") == profile.strategy, "fixed_audit_profile_contract")
    if profile.frozen_candidates:
        require(type(plan.get("bindings")) is dict, "capacity_binding_inventory")
        hashes = plan["bindings"].get("candidate_hashes")
        require(type(hashes) is dict and set(hashes) == set(profile.cases)
                and all(type(h) is str and re.fullmatch("[0-9a-f]{64}", h) for h in hashes.values()),
                "capacity_frozen_candidate_inventory")
        require(all(type(plan["bindings"].get(field)) is dict and set(plan["bindings"][field]) == set(profile.cases)
                    for field in ("requests", "source_reference")), "capacity_fixed_request_inventory")
    return profile


MAX_FILE = 16 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_FILES = 4096
STAGES = ("generation", "compilation", "execution", "validation")
PRIMES = (1152921504581419009, 1152921504581877761, 1152921504583647233,
    1152921504585547777, 1152921504586530817, 1152921504589938689,
    1152921504592429057, 1152921504592822273, 1152921504593412097,
    1152921504595640321, 1152921504595968001, 1152921504597016577,
    1152921504598720513, 1152921504606584833)


def integer(value, expected):
    return type(value) is int and value == expected
COMPARISON_FIELDS = {"passed", "atol", "rtol", "compared_values", "mae",
    "max_absolute_error", "max_nonzero_reference_relative_error", "nonzero_reference_count",
    "cosine_similarity", "actual", "reference", "absolute_error", "relative_error", "elementwise_pass"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def self_hash(value, field="binding"):
    require(type(value) is dict and value.get(field) == digest({k: v for k, v in value.items() if k != field}),
            "self_hash_" + field)


def relative(value):
    require(type(value) is str and value and len(value) <= 1024, "relative_path")
    p = Path(value)
    require(not p.is_absolute() and all(n not in (".", "..") for n in p.parts)
            and str(p) == value, "relative_path")
    return p


class Reader:
    def __init__(self):
        self.files = {}
        self.total = 0

    def bytes(self, path, limit=MAX_FILE):
        p = Path(path)
        require(not any(n.startswith(".") or n.lower() in ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")
                or re.search(r"(?:private|secret)[-_]?key", n, flags=re.I)
                for n in p.parts) and p.suffix.lower() not in (".pem", ".key", ".p12", ".pfx", ".ppk"),
                "credential_or_secret_path")
        require(p.is_absolute() and p == p.resolve() and p.is_file()
                and not any(x.is_symlink() for x in (p, *p.parents)), "regular_evidence_path")
        n = p.stat().st_size
        require(n <= limit, "evidence_file_budget")
        require(str(p) in self.files or (len(self.files) < MAX_FILES and self.total + n <= MAX_TOTAL),
                "evidence_aggregate_budget")
        with p.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(len(raw) == n and len(raw) <= limit, "evidence_file_changed")
        h = hashlib.sha256(raw).hexdigest()
        if str(p) in self.files:
            require(self.files[str(p)] == h, "evidence_changed_during_audit")
        else:
            self.files[str(p)] = h
            self.total += n
        return raw

    def json(self, path):
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, "duplicate_json_field")
                result[key] = value
            return result
        def finite(token):
            value = float(token)
            require(math.isfinite(value), "nonfinite_json")
            return value
        try:
            return json.loads(self.bytes(path), object_pairs_hook=pairs, parse_float=finite,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        except (RecursionError, MemoryError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("bounded_json") from error

    def hashes(self, folder, entries):
        require(type(entries) is dict and 0 < len(entries) <= MAX_FILES, "evidence_hash_inventory")
        for name, expected in entries.items():
            p = folder / relative(name)
            require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected), "evidence_hash")
            require(hashlib.sha256(self.bytes(p)).hexdigest() == expected, "evidence_hash_mismatch")

    def finish(self):
        for path, h in tuple(self.files.items()):
            require(hashlib.sha256(self.bytes(Path(path))).hexdigest() == h, "evidence_changed_during_audit")
        return dict(sorted(self.files.items()))


def npy(raw, expected_shape):
    """Read only bounded little/native-endian float64 C-order arrays."""
    require(type(raw) is bytes and len(raw) <= MAX_FILE and raw[:6] == b"\x93NUMPY", "npy_format_budget")
    version = raw[6:8]
    require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "npy_version")
    offset, width = (10, 2) if version == b"\x01\x00" else (12, 4)
    require(len(raw) >= offset, "npy_header")
    length = int.from_bytes(raw[8:offset], "little")
    require(0 < length <= 4096 and offset + length <= len(raw), "npy_header_budget")
    try:
        header = ast.literal_eval(raw[offset:offset+length].decode("utf-8" if version == b"\x03\x00" else "latin1"))
    except (ValueError, SyntaxError, RecursionError) as error:
        raise ValueError("npy_header") from error
    require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}
            and header["descr"] in ("<f8", "=f8") and header["fortran_order"] is False
            and type(header["shape"]) is tuple and header["shape"] == tuple(expected_shape)
            and all(type(x) is int and 0 < x <= MAX_FILE // 8 for x in header["shape"]), "npy_dtype_shape_order")
    count = math.prod(header["shape"])
    require(count <= MAX_FILE // 8 and len(raw) == offset + length + count * 8, "npy_payload_size")
    values = tuple(x[0] for x in struct.iter_unpack("<d", raw[offset+length:]))
    require(all(math.isfinite(x) for x in values), "npy_nonfinite")
    return values


def npz_reference(raw, shape):
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            require(len(entries) == 2 and {i.filename for i in entries} == {"inputs.npy", "reference.npy"}, "npz_inventory")
            require(all(0 < i.file_size <= MAX_FILE and i.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and not i.flag_bits & 1 for i in entries), "npz_member_budget")
            with archive.open("reference.npy") as stream:
                reference = stream.read(MAX_FILE + 1)
            require(len(reference) == archive.getinfo("reference.npy").file_size, "npz_reference_size")
            return npy(reference, shape)
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("npz_format") from error


def flatten(value):
    if type(value) is list:
        require(len(value) <= MAX_FILE // 8, "numeric_shape_budget")
        return tuple(x for part in value for x in flatten(part))
    require(type(value) in (int, float) and math.isfinite(value), "finite_numeric_value")
    return (float(value),)


def matrix(value, groups, width, boolean=False, nullable=False):
    require(type(value) is list and len(value) == groups and
            all(type(row) is list and len(row) == width for row in value), "comparison_shape")
    if boolean:
        require(all(type(x) is bool for row in value for x in row), "comparison_boolean")
        return tuple(x for row in value for x in row)
    require(all((nullable and x is None) or (type(x) in (int, float) and math.isfinite(x))
                for row in value for x in row), "comparison_finite")
    return tuple(x for row in value for x in row)


def near(actual, expected):
    # This tolerance covers only different summation orders in summary statistics.
    # Acceptance uses exact persisted arrays and the frozen per-element atol/rtol;
    # near() never relaxes that decision or the elementwise_pass comparison.
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), "comparison_statistic")


def comparison(value, groups, width, actual=None, reference=None):
    require(type(value) is dict and set(value) == COMPARISON_FIELDS, "comparison_fields")
    require(value["passed"] is True and value["atol"] == 1e-5 and value["rtol"] == 1e-4
            and type(value["compared_values"]) is int and value["compared_values"] == groups * width, "comparison_tolerance_count")
    a = matrix(value["actual"], groups, width)
    r = matrix(value["reference"], groups, width)
    require((actual is None or a == actual) and (reference is None or r == reference), "comparison_array_binding")
    errors = tuple(abs(x-y) for x, y in zip(a, r))
    require(matrix(value["absolute_error"], groups, width) == errors, "comparison_absolute_error")
    flags = tuple(e <= 1e-5 + 1e-4 * abs(y) for e, y in zip(errors, r))
    require(all(flags) and matrix(value["elementwise_pass"], groups, width, boolean=True) == flags,
            "comparison_elementwise_result")
    relative = matrix(value["relative_error"], groups, width, nullable=True)
    for observed, e, y in zip(relative, errors, r):
        if y == 0:
            require(observed is None, "comparison_zero_reference")
        else:
            near(observed, e / abs(y))
    nz = sum(y != 0 for y in r)
    require(type(value["nonzero_reference_count"]) is int and value["nonzero_reference_count"] == nz, "comparison_nonzero_count")
    near(value["mae"], math.fsum(errors) / len(errors))
    near(value["max_absolute_error"], max(errors))
    if nz:
        near(value["max_nonzero_reference_relative_error"], max(e / abs(y) for e, y in zip(errors, r) if y))
    else:
        require(value["max_nonzero_reference_relative_error"] is None, "comparison_relative_statistic")
    norm = math.sqrt(math.fsum(x*x for x in a)) * math.sqrt(math.fsum(y*y for y in r))
    if norm:
        near(value["cosine_similarity"], math.fsum(x*y for x, y in zip(a, r)) / norm)
    else:
        require(value["cosine_similarity"] is None, "comparison_cosine_statistic")
    return {k: value[k] for k in ("compared_values", "max_absolute_error", "atol", "rtol")}


def bound_source(name, h, plan, reader):
    p = relative(name)
    require(p.suffix in (".py", ".json", ".patch", ".sh", ".nix")
            and not any(n.startswith(".") for n in p.parts), "public_source_path")
    inside = plan["bindings"]["inside_sources"]
    if name in inside:
        require(inside[name] == h, "staged_source_hash")
    else:
        require(hashlib.sha256(reader.bytes(scope.ROOT / p)).hexdigest() == h, "additional_closure_source_hash")


def implementation_projection(inside):
    """Exact PoseidonBackend.identity() glob projection, using loaded bytes."""
    entries = {}
    for name, h in inside.items():
        if not name.startswith("scripts/baseline/"):
            continue
        relative_name = Path(name).relative_to("scripts/baseline")
        parts = relative_name.parts
        if ((len(parts) == 1 and relative_name.suffix in (".py", ".json"))
                or (len(parts) == 2 and parts[0] == "patches" and relative_name.suffix == ".patch")
                or (len(parts) == 2 and parts[0] == "upstream_adapters" and relative_name.suffix == ".py")):
            entries[str(relative_name)] = h
    require(entries, "worker_implementation_projection")
    return entries


def worker_identity(worker, directory, plan, reader):
    """WorkerContext stores the backend wrapper, not a bare SDK identity."""
    backend = worker["backend_identity"]
    require(worker.get("format") == "poseidon-worker-context-v1" and worker.get("backend") == "poseidon_cpu"
            and type(backend) is dict and set(backend) == {"backend", "tools", "implementation_digest"}
            and backend["backend"] == "poseidon_cpu" and backend["tools"] == plan["bindings"]["backend_identity"]
            and backend["implementation_digest"] == digest(implementation_projection(plan["bindings"]["inside_sources"]))
            and worker["paths"]["source_root"] == str(scope.ROOT)
            and worker["paths"]["task_store"] == str(directory / "tasks"), "worker_identity_binding")
    authority = worker["source_authority"]
    require(type(authority) is dict and set(authority) == {"policy", "sources", "sha256"}
            and authority["policy"] == "conservative-local-module-closure-v1"
            and type(authority["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", authority["sha256"])
            and type(authority["sources"]) is dict and authority["sources"], "worker_source_authority")
    for name, h in authority["sources"].items():
        bound_source("scripts/baseline/" + str(relative(name)), h, plan, reader)


def identity(value, plan, reader=None):
    reader = Reader() if reader is None else reader
    require(type(value) is dict and set(value) == {"format", "backend", "policy", *STAGES, "manifest_sha256"}
            and value["format"] == "poseidon-application-identities-v1"
            and value["policy"] == "conservative-local-module-closure-v1", "application_identity_format")
    self_hash(value, "manifest_sha256")
    require(value["backend"] == plan["bindings"]["backend_identity"], "application_backend_identity")
    inside = plan["bindings"]["inside_sources"]
    for stage in STAGES:
        item = value[stage]
        require(type(item) is dict and set(item) == {"format", "stage", "policy", "roots", "sources", "import_edges",
                "dynamic_dependencies", "external_imports", "configuration_files", "backend", "sha256"}, "application_stage_fields")
        require(type(item["sources"]) is dict and item["sources"] and type(item["configuration_files"]) is dict
                and type(item["roots"]) is list and item["roots"] and item["roots"] == sorted(set(item["roots"]))
                and all(n in item["sources"] for n in item["roots"]), "application_stage_roots")
        require(type(item["import_edges"]) is dict and all(n in item["sources"] and type(children) is list
                and children == sorted(set(children)) and all(c in item["sources"] for c in children)
                for n, children in item["import_edges"].items()), "application_stage_closure")
        self_hash(item, "sha256")
        require(item["format"] == "poseidon-application-stage-identity-v1" and item["stage"] == stage
                and item["policy"] == value["policy"], "application_stage_identity")
        for name, h in item["sources"].items():
            relative(name)
            bound_source("scripts/baseline/" + name, h, plan, reader)
        for name, h in item["configuration_files"].items():
            relative(name)
            bound_source(name, h, plan, reader)
    return value["manifest_sha256"]


def execution_metadata(value):
    require(value.get("backend") == "poseidon_cpu" and value.get("encrypted_execution") is True
            and value.get("executor_secret_key") is False and value.get("bootstrap") is False
            and value.get("degree") == 32768 and value.get("data_moduli") == 13
            and value.get("special_moduli") == 1 and value.get("security") == "tc128"
            and value.get("q") == list(PRIMES[:13]) and value.get("p") == list(PRIMES[13:]), "native_execution_metadata")


def decoded(reader, output, request, groups, width):
    native = reader.json(output / "execution.json")
    metadata = reader.json(output / "execution-metadata.json")
    execution_metadata(native)
    execution_metadata(metadata)
    require({k: v for k, v in native.items() if k != "outputs"} == metadata, "execution_metadata_binding")
    require(type(native["outputs"]) is list and len(native["outputs"]) == groups, "execution_output_groups")
    selectors = request["layout"]["output_selectors"]
    require(type(selectors) is list and len(selectors) == width, "execution_selector_width")
    selected = []
    for sample in native["outputs"]:
        require(type(sample) is list, "execution_cipher_outputs")
        for selector in selectors:
            require(type(selector) is list and len(selector) == 2 and all(type(x) is int and x >= 0 for x in selector), "execution_selector")
            c, slot = selector
            require(c < len(sample) and type(sample[c]) is list and slot < len(sample[c]), "execution_selector_bounds")
            selected.extend(flatten(sample[c][slot]))
    values = npy(reader.bytes(output / "decrypted.npy"), (groups, width))
    require(tuple(selected) == values, "execution_decoding_binding")
    return values, metadata


def request_matches(stored, logical, plan, reader):
    """Compare exact logical requests or their producer's lossless wire codec."""
    if stored == logical:
        return True
    path = scope.BASE / "unified_request_storage.py"
    raw = reader.bytes(path)
    require(hashlib.sha256(raw).hexdigest() == plan["bindings"]["inside_sources"].get(
        "scripts/baseline/unified_request_storage.py"), "request_codec_source_binding")
    # This frozen codec imports only json/math. Loading it creates no pycache;
    # compact_request is data-only, while SDK-dependent restoration is not called.
    tree = ast.parse(raw)
    require(all(all(alias.name in ("json", "math") for alias in node.names)
            for node in tree.body if isinstance(node, ast.Import))
            and not any(isinstance(node, ast.ImportFrom) for node in tree.body), "request_codec_standard_library")
    namespace = {"__name__": "_frozen_request_wire_codec"}
    exec(compile(tree, str(path), "exec"), namespace)
    return stored == namespace["compact_request"](logical)


def receipt_context(value, worker, request):
    self_hash(value)
    require(value["format"] == "poseidon-request-context-v1" and value["worker_binding"] == worker["binding"]
            and value["request_id"] == request["request_id"] and value["request_sha256"] == digest(request)
            and value["compiler_configuration_sha256"] == digest(request["compiler_configuration"])
            and value["layout_sha256"] == digest(request["layout"]) and value["validation_level"] == "numerical", "request_context_binding")


def validate_case(directory, name, runtime_plan):
    """Return only public counts and a complete hash inventory, or fail closed."""
    profile = audit_profile(runtime_plan)
    d = Path(directory)
    require(d.is_absolute() and d == d.resolve() and d.is_dir() and name in profile.cases and d.name == name, "case_directory")
    self_hash(runtime_plan)
    require(runtime_plan["new_agent_generation"] is False and runtime_plan["production_applied"] is False, "joint_runtime_plan")
    inside = runtime_plan["bindings"]["inside_sources"]
    require(type(inside) is dict and len(inside) == 706, "joint_706_inventory")
    reader = Reader()
    require(reader.json(d.parent / "plan.json") == runtime_plan
            and reader.json(d.parent / "staged-sources.json") == inside, "full_staged_706_binding")
    aggregate = reader.json(d.parent / "report.json")
    require(aggregate.get("format") == "poseidon-ordered-common-offline-report-v1"
            and aggregate.get("binding") == runtime_plan["binding"]
            and aggregate.get("status") in ("running", "passed", "failed", "stopped"),
            "aggregate_runtime_binding")
    aggregate_rows = aggregate.get("rows")
    expected_phases = {(case, phase) for case in profile.cases for phase in profile.phases}
    require(type(aggregate_rows) is list and 0 < len(aggregate_rows) <= len(expected_phases)
            and all(type(row) is dict and row.get("case") in profile.cases and row.get("phase") in profile.phases
                    for row in aggregate_rows)
            and len({(row["case"], row["phase"]) for row in aggregate_rows}) == len(aggregate_rows),
            "aggregate_phase_inventory")
    aggregate_by_phase = {(row["case"], row["phase"]): row for row in aggregate_rows}
    require(all((name, phase) in aggregate_by_phase for phase in profile.phases), "selected_case_complete_phases")
    rows = {}
    for phase in profile.phases:
        row = reader.json(d / (phase + ".json"))
        require(row["case"] == name and row["phase"] == phase and row["status"] == "passed"
                and integer(row.get("provider_calls"), 0)
                and row.get("new_agent_generation") is False
                and row.get("candidate_origin") == profile.origin, "phase_metadata")
        # one() persists the phase receipt before inside() adds the owned child's
        # exit code to the aggregate. Both records must describe the same phase.
        completed = aggregate_by_phase[(name, phase)]
        require(integer(completed.get("exit_code"), 0), "aggregate_phase_exit")
        require({k: v for k, v in completed.items() if k != "exit_code"} == row,
                "aggregate_phase_receipt_binding")
        start = reader.json(d / (phase + "-started.json"))
        require(start["format"] == "poseidon-offline-phase-start-v1" and start["binding"] == runtime_plan["binding"]
                and start["case"] == name and start["phase"] == phase, "phase_start_binding")
        rows[phase] = row
    q, new, cache = (rows[p] for p in profile.phases)
    require(q["compiled"] is True and q["encrypted_execution"] is True
            and q["source_capture_supervisor_executed"] is True, "qualification_grade")
    result = q["result"]
    require(result["status"] == "succeeded" and result["validation_level"] == "numerical"
            and integer(result["generations"], 1) and integer(result["http_attempts"], 0)
            and result["generation"]["agent_generated"] is False, "qualification_call_boundary")
    task_id = reader.json(d / "task-id.json")["task_id"]
    require(type(task_id) is str and re.fullmatch("[0-9a-f]{32}", task_id) and result["task_id"] == task_id, "qualification_task_id")
    task = d / "tasks" / task_id
    package = task / "program"
    require(q["package"] == str(package), "qualification_package_path")
    binding = reader.json(d / "qualification-binding.json")
    if profile.frozen_candidates:
        # The remaining checks bind this exact SHA to the qualification row,
        # delivered package, native candidate and reload payload independently.
        require(binding.get("candidate_sha256") == runtime_plan["bindings"]["candidate_hashes"][name],
                "capacity_frozen_candidate_binding")
    request = binding["request"]
    expected = runtime_plan["bindings"]["requests"][name]
    source_ref = runtime_plan["bindings"]["source_reference"][name]
    require(digest(request) == expected["request_canonical_sha256"] == binding["request_sha256"]
            and request["request_id"] == expected["request_id"] == q["request_id"]
            and digest(request["model"]) == expected["model_sha256"] == source_ref["model_sha256"]
            and digest(request["public_constants"]) == expected["public_constants_sha256"]
            and binding["model_source_sha256"] == source_ref["source_package_sha256"]
            and binding["candidate_origin"] == profile.origin and binding["agent_generated"] is False, "request_source_binding")
    meta = reader.json(task / "task.json")
    self_hash(meta)
    require(meta["request"] == request and meta["validation_level"] == "numerical" and meta["identity_policy"] == "staged-v1", "task_request_binding")
    worker = reader.json(d / "tasks" / "worker-context.json")
    self_hash(worker)
    worker_identity(worker, d, runtime_plan, reader)
    stage_hash = identity(meta["identity"], runtime_plan, reader)
    require(worker["stage_identity_sha256"] == stage_hash and binding["identity"] == meta["identity"], "qualification_identity_binding")
    manifest = reader.json(package / "manifest.json")
    self_hash(manifest)
    require(manifest["format"] == "poseidon-dsl-bundle-v6" and manifest["agent_generated"] is False
            and manifest["binding"] == q["package_binding"] == result["package"]["binding"]
            and result["package"]["reused"] is False and result["package"]["task_id"] == task_id
            and manifest["validation"] == dict(level="numerical", encrypted_execution=True, numerically_validated=True)
            and manifest["compiler_configuration"] == request["compiler_configuration"]
            and manifest["model_provenance"] == "included"
            and manifest["origin"]["request_id"] == request["request_id"], "package_grade_binding")
    require({"candidate.py", "model.json", "request.json", "model-provenance.json"} <= set(manifest["files"])
            and all(n in {"candidate.py", "model.json", "request.json", "model-provenance.json"}
                    or n.startswith("artifacts/") for n in manifest["files"]), "package_file_inventory")
    reader.hashes(package, manifest["files"])
    require(request_matches(reader.json(package / "request.json"), request, runtime_plan, reader)
            and reader.json(package / "model.json") == request["model"], "package_request_binding")
    require(reader.files[str(package / "candidate.py")] == binding["candidate_sha256"] == q["candidate_sha256"], "candidate_byte_binding")
    require(manifest["dependencies"]["backend_identity"] == runtime_plan["bindings"]["backend_identity"], "package_backend_binding")
    packaged_identity = reader.json(package / "artifacts/application-identities.json")
    descriptor = manifest["dependencies"]["application_identities"]
    require(descriptor == dict(file="artifacts/application-identities.json", sha256=manifest["files"]["artifacts/application-identities.json"])
            and packaged_identity == meta["identity"], "package_stage_binding")
    provenance = reader.json(package / "model-provenance.json")
    require(digest(provenance["original"]) == source_ref["original_sha256"]
            and digest(provenance["trusted_pytorch"]["manifest"]) == source_ref["source_package_sha256"], "original_model_source_binding")
    origin = reader.json(package / "artifacts/model-source.json")
    self_hash(origin)
    source = reader.json(task / "source.json")
    self_hash(source)
    require(origin["format"] == "poseidon-model-source-origin-v1" and origin["kind"] == source["kind"] == "trusted_pytorch"
            and origin["snapshot_binding"] == source["binding"] and origin["files"] == source["files"]
            and origin["implementation"] == source["implementation"]
            and source["identity"] == meta["identity"] and source["identity_policy"] == "staged-v1"
            and origin["model_sha256"] == digest(request["model"]) and origin["provenance_sha256"] == digest(provenance)
            and origin["resource_profile"] == source["resource_profile"] == request["model"]["resource_profile"], "source_origin_binding")
    require(origin["implementation"] == dict(format="poseidon-source-intake-controller-v1",
            controller_sha256=meta["identity"]["generation"]["sources"]["application_model_source.py"]), "source_controller_binding")
    source_package = provenance["trusted_pytorch"]
    source_manifest = source_package["manifest"]
    source_bytes = canonical(source_manifest)
    files = {"manifest.json": dict(bytes=len(source_bytes), sha256=hashlib.sha256(source_bytes).hexdigest())}
    require(set(source_package["files"]) == set(source_manifest["files"]), "source_text_inventory")
    for filename, text in source_package["files"].items():
        relative(filename)
        require(type(text) is str and len(text.encode()) <= MAX_FILE, "source_text_budget")
        raw = text.encode()
        require(hashlib.sha256(raw).hexdigest() == source_manifest["files"][filename], "source_text_hash")
        files[filename] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for declaration in source_manifest["weights"].values():
        relative(declaration["file"])
        record = dict(bytes=declaration["bytes"], sha256=declaration["sha256"])
        require(type(record["bytes"]) is int and 0 < record["bytes"] <= MAX_FILE
                and (declaration["file"] not in files or files[declaration["file"]] == record), "source_weight_record")
        files[declaration["file"]] = record
    require(origin["files"] == files and manifest["origin"]["model_source"] == dict(format=origin["format"],
            file="artifacts/model-source.json", sha256=manifest["files"]["artifacts/model-source.json"]), "package_source_inventory")
    validation = reader.json(task / "validation-0.json")
    require(validation["format"] == "poseidon-agent-component-v1" and validation["status"] == "passed"
            and integer(validation["exit_code"], 0) and validation["validation_level"] == "numerical"
            and validation["compiled_validated"] is True and validation["encrypted_execution"] is True
            and validation["numerically_validated"] is True and integer(validation["paid_calls"], 0)
            and validation["new_agent_generation"] is False and validation["request_id"] == request["request_id"], "task_validation_grade")
    require(all(validation["observed_stages"].get(k) is True for k in
            ("compiled", "encrypted_execution", "evidence_integrity_not_rejected", "numerically_compared", "traced")), "observed_validation_stages")
    receipt_context(validation["run_context"], worker, request)
    evidence = Path(validation["evidence"])
    require(evidence.is_absolute() and (evidence.is_relative_to(scope.RESULTS) or evidence.is_relative_to(d.parent)), "native_evidence_scope")
    report = reader.json(evidence / "report.json")
    require(reader.files[str(evidence / "report.json")] == manifest["origin"]["report_sha256"]
            and report["status"] == "passed" and report["backend"] == "poseidon_cpu"
            and integer(report["agent_calls"], 0) and report["llm_generation_validated"] is False
            and report["backend_identity"] == runtime_plan["bindings"]["backend_identity"]
            and report["request_id"] == request["request_id"] and report["validation_level"] == "numerical", "native_report_binding")
    require(request_matches(reader.json(evidence / "request.json"), request, runtime_plan, reader)
            and reader.json(evidence / "model.json") == request["model"], "native_request_binding")
    for path, h in report["source_hashes"].items():
        bound_source("scripts/baseline/" + str(relative(path)), h, runtime_plan, reader)
    reader.hashes(evidence, report["frozen_hashes"])
    require(len(report["attempts"]) == 1, "single_compiler_attempt")
    attempt = report["attempts"][0]
    require(attempt["index"] == 0 and attempt["status"] == "passed" and attempt["compiled"] is True
            and attempt["executed"] is True and attempt["numerically_correct"] is True
            and attempt["trace"]["frontend"] == "real_Hecate" and attempt["trace"]["candidate_python_executed"] is False
            and attempt["trace"]["construction"] == "validated_native_AST_to_Hecate_functions"
            and attempt["trace"]["request_id"] == request["request_id"], "real_compilation_execution_receipt")
    output = evidence / "attempt-00/output"
    reader.hashes(output, attempt["artifact_hashes"])
    require("trace-evidence.json" in attempt["artifact_hashes"]
            and reader.json(output / "trace-evidence.json") == attempt["trace"], "native_trace_receipt_binding")
    require(hashlib.sha256(reader.bytes(evidence / "attempt-00/candidate.py")).hexdigest() == binding["candidate_sha256"], "native_candidate_binding")
    require(manifest["dependencies"]["compiler_sha256"] == report["compiler_sha256"]
            == runtime_plan["bindings"]["backend_identity"]["compiler_sha256"]
            and manifest["origin"]["frontend_sha256"] == report["frontend_sha256"]
            == runtime_plan["bindings"]["backend_identity"]["frontend_sha256"]
            and manifest["origin"]["runtime_sha256"] == report["runtime_sha256"]
            == runtime_plan["bindings"]["backend_identity"]["executor_sha256"]
            and report["compiler_configuration"] == request["compiler_configuration"], "compiler_sdk_binding")
    require(attempt["runtime_plan_entry"]["format"] == "hecate-private-symbol-dce-v1"
            and attempt["runtime_plan_entry"]["function_bodies_unchanged"] is True
            and type(attempt["compile_command"]) is list and attempt["compile_command"]
            and Path(attempt["compile_command"][0]).name == "hecate-opt", "compiled_entry_receipt")
    artifact_validation = reader.json(package / "artifacts/validation.json")
    require(artifact_validation["level"] == "numerical" and artifact_validation["scope"] == "four fixed inputs", "artifact_validation_scope")
    require(artifact_validation["gate"] == attempt["artifact_gate"] and artifact_validation["runtime_plan_entry"] == attempt["runtime_plan_entry"]
            and artifact_validation["compile_command"] == attempt["compile_command"], "compiled_artifact_record_binding")
    for filename, h in manifest["files"].items():
        if filename.startswith("artifacts/") and filename[10:] in attempt["artifact_hashes"]:
            require(h == attempt["artifact_hashes"][filename[10:]], "compiled_package_artifact_binding")
    layout_outputs = request["layout"]["outputs"]
    require([o["name"] for o in layout_outputs] == [o["name"] for o in request["model"]["outputs"]]
            and all(type(o["shape"]) is list and all(type(n) is int and n > 0 for n in o["shape"]) for o in layout_outputs), "output_layout_shapes")
    width = sum(math.prod(o["shape"]) for o in layout_outputs)
    require(len(request["layout"]["output_selectors"]) == width, "output_layout_width")
    require(0 < width <= MAX_FILE // 32, "output_width_budget")
    actual, metadata = decoded(reader, output, request, 4, width)
    require(metadata == attempt["execution"], "qualification_execution_binding")
    reference = npz_reference(reader.bytes(evidence / "arrays.npz"), (4, width))
    qualification_summary = comparison(attempt["comparison"], 4, width, actual, reference)
    acceptance = result["acceptance"]
    require(acceptance["level"] == "numerical" and all(acceptance[k] is True for k in
            ("artifact_checked", "compiled", "encrypted_execution", "numerically_validated"))
            and acceptance["input_groups"] == 4 and acceptance["compared_values"] == 4 * width
            and acceptance["maximum_absolute_error"] == qualification_summary["max_absolute_error"], "qualification_acceptance_binding")
    require(q["input_groups"] == 4 and q["compared_values"] == 4 * width
            and q["maximum_absolute_error"] == qualification_summary["max_absolute_error"]
            and validation["compared_values"] == 4 * width and validation["max_absolute_error"] == q["maximum_absolute_error"], "qualification_summary_binding")
    require(new["compiled"] is False and new["encrypted_execution"] is True and new["package_reloaded"] is True
            and new["input_groups"] == 3, "new_input_phase_grade")
    execution = new["execution"]
    require(execution["format"] == "poseidon-local-execution-v1" and execution["status"] == "executed"
            and execution["bundle_binding"] == manifest["binding"] and execution["request_id"] == request["request_id"]
            and integer(execution["agent_calls"], 0) and integer(execution["compiler_calls"], 0) and integer(execution["samples"], 3)
            and execution["encrypted_execution"] is True and execution["executor_secret_key"] is False
            and execution["package_validation"] == manifest["validation"]
            and execution["numerically_validated_on_supplied_inputs"] is False, "new_input_execution_boundary")
    receipt_context(execution["run_context"], worker, request)
    new_evidence = Path(execution["evidence"])
    require(new_evidence.is_absolute() and new_evidence != evidence
            and (new_evidence.is_relative_to(scope.RESULTS) or new_evidence.is_relative_to(d.parent)), "new_execution_evidence_scope")
    require(reader.json(new_evidence / "report.json") == {k: v for k, v in execution.items() if k != "run_context"}, "new_execution_report_binding")
    new_output = new_evidence / "output"
    payload = reader.json(new_evidence / "payload.json")
    require(request_matches(payload["request"], request, runtime_plan, reader)
            and payload["candidate"]["request_id"] == request["request_id"]
            and hashlib.sha256(payload["candidate"]["hecate_source"].encode()).hexdigest() == binding["candidate_sha256"], "new_execution_payload_binding")
    reader.bytes(new_output / "inputs.json")
    for filename, h in manifest["files"].items():
        if filename.startswith("artifacts/"):
            require(hashlib.sha256(reader.bytes(new_output / relative(filename[10:]))).hexdigest() == h, "reload_artifact_binding")
    new_actual, new_metadata = decoded(reader, new_output, request, 3, width)
    require(new_metadata == execution["execution"], "new_execution_metadata_binding")
    require(type(execution["outputs"]) is list and len(execution["outputs"]) == 3, "new_named_outputs")
    named = tuple(x for sample in execution["outputs"] for o in request["model"]["outputs"] for x in flatten(sample[o["name"]]))
    require(named == new_actual, "new_named_decoded_binding")
    new_summary = comparison(new["comparison"], 3, width, new_actual)
    require(cache["compiled"] is False and cache["encrypted_execution"] is False and cache["package_reused"] is True, "cache_phase_boundary")
    cached = cache["result"]
    require(cached["status"] == "succeeded" and integer(cached["generations"], 0) and integer(cached["http_attempts"], 0)
            and cached["package"]["reused"] is True and cached["package"]["binding"] == manifest["binding"]
            and cached["package"]["task_id"] == task_id and cached["task_id"] != task_id, "cache_reuse_boundary")
    require(type(cached["task_id"]) is str and re.fullmatch("[0-9a-f]{32}", cached["task_id"]), "cache_task_id")
    state = reader.json(d / "tasks" / cached["task_id"] / "state.json")
    require(all(state[k] == cached[k] for k in ("status", "generations", "http_attempts", "package")), "cache_state_binding")
    return dict(case=name, status="passed", candidate_origin=profile.origin,
        runtime_binding=runtime_plan["binding"], source_count=706, request_id=request["request_id"],
        candidate_sha256=binding["candidate_sha256"], package_binding=manifest["binding"],
        application_identity_sha256=stage_hash, backend_identity_sha256=digest(runtime_plan["bindings"]["backend_identity"]),
        qualification=qualification_summary, new_inputs=new_summary, compilations=1,
        qualification_encrypted_phases=1, new_input_encrypted_phases=1, cache_generations=0,
        provider_calls=0, http_attempts=0, reference_semantics="frozen_reviewed_runner",
        evidence_files=reader.finish())
