"""Staged order-preserving transport for retained rule-only model provenance.

Normalization and the complete production source/package checks stay unchanged.
Only JSON representation is repaired in a new, separately bound package.
"""
import copy
import hashlib
import json
from pathlib import Path
from benchmark_graph import canonical, digest, require

PROVENANCE = "model-provenance.json"
MAX_PROVENANCE = 4*1024**2
MAX_MANIFEST = 131072
MAX_ARTIFACT = 16*1024**2
FORMAT = "poseidon-ordered-provenance-transport-v1"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _parse(value, limit):
    require(type(value) is bytes and len(value) <= limit, "Transport JSON size")
    def pairs(items):
        result = {}
        for name, item in items:
            require(name not in result, "Transport duplicate JSON key")
            result[name] = item
        return result
    def nonfinite(value):
        raise ValueError("Transport nonfinite JSON")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)


def declaration_orders(value):
    """Bind declared constant names at their actual nested JSON locations."""
    result = []
    def visit(item, path):
        if type(item) is dict:
            for key, child in item.items():
                location = path+[key]
                if key in ("constants", "public_constants") and type(child) is dict:
                    result.append(dict(path=location, names=list(child)))
                visit(child, location)
        elif type(item) is list:
            for index, child in enumerate(item):
                visit(child, path+[index])
    visit(value, [])
    return result


def serialize(provenance, model):
    """Fully validate the original and its order-preserving JSON roundtrip."""
    from candidate_bundle import check_provenance
    before = canonical(provenance)
    orders = declaration_orders(provenance)
    check_provenance(provenance, model)
    require(canonical(provenance) == before and declaration_orders(provenance) == orders,
            "Transport original provenance mutated during validation")
    blob = json.dumps(provenance, indent=2, sort_keys=False, allow_nan=False).encode()+b"\n"
    require(len(blob) <= MAX_PROVENANCE, "Transport provenance size")
    parsed = _parse(blob, MAX_PROVENANCE)
    require(canonical(parsed) == before and declaration_orders(parsed) == orders,
            "Transport provenance value or declaration order changed")
    check_provenance(parsed, model)
    require(canonical(parsed) == before and declaration_orders(parsed) == orders,
            "Transport reloaded provenance mutated during validation")
    implementation = Path(__file__).resolve()
    record = dict(format=FORMAT, canonical_sha256=sha_bytes(before),
        serialized_sha256=sha_bytes(blob), serialized_bytes=len(blob),
        constant_declaration_orders=orders, declaration_order_sha256=digest(orders),
        implementation=dict(path=str(implementation), sha256=sha_bytes(implementation.read_bytes())),
        full_provenance_checks=2, canonical_value_unchanged=True,
        normalization_changed=False, production_export_changed=False,
        compiler_run=False, encrypted_execution=False, api_calls=0)
    record["binding"] = digest(record)
    return blob, record


def _source_file(folder, name, limit):
    require(type(name) is str and name and "\\" not in name, "Transport source artifact path")
    relative = Path(name)
    parts = name.split("/")
    require(not relative.is_absolute() and all(parts) and not any(p in (".", "..") for p in parts),
            "Transport source artifact path")
    path = folder/relative
    require(path.resolve().is_relative_to(folder.resolve()), "Transport source artifact escape")
    current = folder
    for part in relative.parts:
        current = current/part
        require(not current.is_symlink(), "Transport source artifact symlink")
    require(path.is_file() and path.stat().st_size <= limit, "Transport source artifact size or type")
    blob = path.read_bytes()
    require(len(blob) <= limit, "Transport source artifact size changed")
    return blob


def _write_new(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)


def _file_limit(name, data_limit):
    if name.startswith("artifacts/"): return MAX_ARTIFACT
    if name in ("model.json", "request.json", PROVENANCE): return data_limit
    return MAX_MANIFEST


def repack(source_folder, destination, provenance):
    """Retain failed evidence and rebind only the checked JSON representation."""
    from candidate_bundle import check_bundle
    source = Path(source_folder)
    output = Path(destination)
    require(source.is_dir() and not source.is_symlink(), "Transport source package directory")
    require(not output.exists() and not output.is_symlink(), "Preserve destination package")
    manifest_bytes = _source_file(source, "manifest.json", MAX_MANIFEST)
    manifest = _parse(manifest_bytes, MAX_MANIFEST)
    require(type(manifest) is dict and manifest.get("format") in
            ("poseidon-dsl-bundle-v4", "poseidon-dsl-bundle-v5", "poseidon-dsl-bundle-v6"),
            "Transport Poseidon package format")
    require(manifest.get("agent_generated") is False and
            "generation_provenance" not in manifest.get("origin", {}),
            "Transport rule-only package authority")
    require(manifest.get("model_provenance") == "included", "Transport retained model provenance")
    require(digest({k:v for k,v in manifest.items() if k != "binding"}) == manifest.get("binding"),
            "Transport source manifest binding")
    file_hashes = manifest.get("files")
    require(type(file_hashes) is dict and {"model.json", "request.json", "candidate.py", PROVENANCE}
            <= set(file_hashes), "Transport source package files")
    limit = MAX_PROVENANCE if manifest["format"].endswith("v6") else MAX_MANIFEST
    original_files = {}
    for name, expected in file_hashes.items():
        require(type(name) is str and (name in ("model.json", "request.json", "candidate.py", PROVENANCE)
                or name.startswith("artifacts/")), "Transport source package file class")
        blob = _source_file(source, name, _file_limit(name, limit))
        require(type(expected) is str and len(expected) == 64 and sha_bytes(blob) == expected,
                "Transport source file integrity")
        original_files[name] = blob
    stored = _parse(original_files[PROVENANCE], limit)
    require(canonical(provenance) == canonical(stored), "Transport original provenance value mismatch")
    model = _parse(original_files["model.json"], limit)
    serialized, record = serialize(provenance, model)
    require(len(serialized) <= limit, "Transport package provenance size")
    output.mkdir(parents=True, exist_ok=False)
    for name, blob in original_files.items():
        _write_new(output/name, serialized if name == PROVENANCE else blob)
    repaired = copy.deepcopy(manifest)
    repaired["files"][PROVENANCE] = sha_bytes(serialized)
    repaired.pop("binding")
    repaired["binding"] = digest(repaired)
    manifest_blob = json.dumps(repaired, indent=2, sort_keys=True, allow_nan=False).encode()+b"\n"
    _write_new(output/"manifest.json", manifest_blob)
    require(check_bundle(output) == repaired, "Transport full package reload mismatch")
    require(_source_file(source, "manifest.json", MAX_MANIFEST) == manifest_bytes,
            "Transport source manifest changed")
    for name, blob in original_files.items():
        require(_source_file(source, name, _file_limit(name, limit)) == blob,
                "Transport source file changed")
        if name != PROVENANCE:
            require((output/name).read_bytes() == blob, "Transport unrelated package file changed")
    updated = dict(record)
    updated.pop("binding")
    updated.update(source_kind="deterministic_rule_baseline", source_package=str(source),
        repaired_package=str(output), old_package_binding=manifest["binding"],
        new_package_binding=repaired["binding"],
        old_manifest_sha256=sha_bytes(manifest_bytes), new_manifest_sha256=sha_bytes(manifest_blob),
        old_provenance_sha256=sha_bytes(original_files[PROVENANCE]),
        new_provenance_sha256=sha_bytes(serialized),
        source_declared_order_sha256=digest(declaration_orders(stored)),
        retained_declaration_order_sha256=record["declaration_order_sha256"],
        unchanged_file_sha256={n:s for n,s in file_hashes.items() if n != PROVENANCE},
        request_sha256=file_hashes["request.json"], candidate_sha256=file_hashes["candidate.py"],
        model_sha256=file_hashes["model.json"], compiler_sha256=manifest["dependencies"]["compiler_sha256"],
        compiler_configuration_sha256=digest(manifest["compiler_configuration"]),
        provenance_value_unchanged=True, request_unchanged=True, candidate_unchanged=True,
        model_unchanged=True, compiler_unchanged=True, original_package_unchanged=True,
        full_package_check_passed=True, new_agent_generation=False)
    updated["binding"] = digest(updated)
    return repaired, updated
