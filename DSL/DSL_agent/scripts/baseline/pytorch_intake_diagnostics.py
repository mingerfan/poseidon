"""Bounded public diagnostics for the trusted-local PyTorch intake boundary."""
import json
import re
from pathlib import Path
from benchmark_graph import digest, require
from model_weight_package import regular_bytes
from pytorch_approximation import (
    DIAGNOSTIC_FORMAT as FORMAT, ApproximationRequired, approximation_required,
    checked_approximation_failure as checked,
)

WIRE_FORMAT = "poseidon-pytorch-input-failure-v1"
FILENAME = "capture-failure.json"
EXIT_CODE = 78
MAX_BYTES = 16384

def _missing(value, manifest):
    from pytorch_approximation import Resolver
    require(type(manifest) is dict and "approximations" in manifest, "Capture diagnostic manifest")
    context, location = value["context"], value["location"]
    try:
        Resolver(manifest["approximations"]).select(
            location["name"], context["operator"], context["function"], context["module_paths"])
    except ApproximationRequired as error:
        require(error.public_failure == value, "Capture approximation witness")
        return
    raise ValueError("Capture diagnostic does not witness missing approximation")

def record(diagnostic, manifest, source_hashes):
    value = checked(diagnostic)
    _missing(value, manifest)
    require(type(source_hashes) is dict and source_hashes and
            all(type(k) is str and type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
                for k, v in source_hashes.items()), "Capture diagnostic implementation")
    result = dict(format=WIRE_FORMAT, manifest_sha256=digest(manifest),
                  source_identity_sha256=digest(source_hashes), diagnostic=value)
    result["binding_sha256"] = digest(result)
    return result

def write(path, diagnostic, manifest, source_hashes):
    value = record(diagnostic, manifest, source_hashes)
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    require(len(raw) <= MAX_BYTES, "Capture diagnostic byte limit")
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return value

def read(path, manifest, source_hashes):
    from public_weight_context import parse_data
    raw = regular_bytes(Path(path), MAX_BYTES)
    value = parse_data(raw.decode("utf-8"))
    require(type(value) is dict and set(value) == {
            "format", "manifest_sha256", "source_identity_sha256", "diagnostic", "binding_sha256"},
            "Capture failure envelope")
    rebuilt = record(value["diagnostic"], manifest, source_hashes)
    require(value == rebuilt, "Capture failure binding")
    return checked(value["diagnostic"])
