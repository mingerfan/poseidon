"""Inactive analysis of complete retained v34 source provenance.

Hash and wire roundtrip checks only; does not replace trusted NPY or capture
validation. The 8 MiB part read bound applies to known historical evidence
analysis, not production intake. No generated code is executed.
"""
import base64, hashlib, json, re
from pathlib import Path
from codec import encode, decode, raw, sha, require

PART_NAMES = {
    "original": "original.json",
    "decomposition": "decomposition.json",
    "reference_report": "reference-report.json",
}
PART_READ_BYTES = 8 * 1024**2
SOURCE_BYTES = 65536
MANIFEST_BYTES = 131072
WEIGHT_FILE_BYTES = 8 * 1024**2

def regular(path, limit, expected_sha=None, exact_size=None):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "retained_regular_file")
    size = path.stat().st_size
    require(type(limit) is int and 0 <= size <= limit, "retained_file_budget")
    require(exact_size is None or type(exact_size) is int and size == exact_size,
            "retained_file_size")
    data = path.read_bytes()
    require(len(data) == size, "retained_file_changed")
    if expected_sha is not None:
        require(type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
                "retained_hash_format")
        require(hashlib.sha256(data).hexdigest() == expected_sha, "retained_file_hash")
    return data

def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_retained_key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite_retained_json")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)

def assemble(root, part_sha256):
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), "retained_root")
    require(type(part_sha256) is dict and set(part_sha256) == set(PART_NAMES),
            "retained_part_inventory")
    value = {key: parse(regular(root / name, PART_READ_BYTES, part_sha256[key]))
             for key, name in PART_NAMES.items()}
    source = root / "trusted-python"
    require(not source.is_symlink() and source.is_dir(), "retained_source_root")
    spec = parse(regular(source / "manifest.json", MANIFEST_BYTES))
    require(type(spec) is dict and spec.get("format") == "poseidon-trusted-pytorch-v34",
            "draft_v34_source_scope")
    require(value["reference_report"].get("manifest_sha256") == sha(spec, sorted_keys=True)
            and value["decomposition"].get("manifest_sha256") == sha(spec, sorted_keys=True),
            "retained_manifest_binding")
    inventory = spec.get("files")
    require(type(inventory) is dict and 1 <= len(inventory) <= 8, "retained_source_inventory")
    files, spent = {}, 0
    for name, expected in inventory.items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py", name),
                "retained_source_filename")
        data = regular(source / name, SOURCE_BYTES - spent, expected)
        spent += len(data)
        files[name] = data.decode("utf-8")
    weights = spec.get("weights")
    require(type(weights) is dict and len(weights) <= 256, "retained_weight_inventory")
    weight_files, seen = {}, {}
    for record in weights.values():
        require(type(record) is dict and set(record) ==
                {"file", "sha256", "bytes", "shape", "dtype"}, "retained_weight_descriptor")
        name = record["file"]
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9_-]+\.npy", name),
                "retained_weight_filename")
        require(name not in seen or seen[name] == record, "retained_weight_file_conflict")
        if name in seen:
            continue
        seen[name] = record
        data = regular(source / name, WEIGHT_FILE_BYTES, record["sha256"], record["bytes"])
        weight_files[name] = base64.b64encode(data).decode("ascii")
    value["trusted_pytorch"] = {
        "format": "poseidon-trusted-pytorch-provenance-v34",
        "manifest": spec,
        "files": files,
        "weight_files": weight_files,
    }
    return value

def analyze(value):
    packet = encode(value)
    restored = decode(packet)
    require(raw(restored) == raw(value), "retained_ordered_roundtrip")
    require(sha(restored, sorted_keys=True) == sha(value, sorted_keys=True),
            "retained_canonical_roundtrip")
    return packet, dict(
        full_provenance_fields=list(value),
        logical_bytes=packet["logical_bytes"],
        wire_bytes=len(raw(packet)),
        ordered_sha256=packet["ordered_sha256"],
        canonical_sha256=packet["canonical_sha256"],
        packet_binding=packet["binding"],
        exact_roundtrip=True,
        manifest_source_weight_hashes_checked=True,
        trusted_npy_decoding_checked=False,
        trusted_capture_semantics_checked=False,
        production_active=False,
        production_source_status="blocked by unchanged provenance intake gate",
        provider_calls=0, compiler_calls=0, encrypted_calls=0,
    )
