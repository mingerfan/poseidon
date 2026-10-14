"""Versioned, data-only model-source provenance for tasks and DSL packages.

Source provenance describes submission/capture, separately from compilation,
encrypted numerical validation and live-generation evidence.
"""
import copy
import hashlib
import re
from pathlib import Path

from benchmark_graph import canonical, digest, require, graph_resource_profile
from model_resources import resolve
from pytorch_weight_package import declarations

FORMAT = "poseidon-model-source-origin-v1"
FILE = "artifacts/model-source.json"
LIMIT = 4 * 1024**2
FIELDS = {"format", "kind", "resource_profile", "snapshot_binding", "files",
          "implementation", "model_sha256", "provenance_sha256", "binding"}


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _file(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _implementation(value):
    require(type(value) is dict and set(value) == {"format", "controller_sha256"} and
            value["format"] == "poseidon-source-intake-controller-v1" and
            _hash(value["controller_sha256"]), "Model source controller identity")


def _expected(kind, model, provenance, profile):
    if kind == "graph":
        if provenance is None:
            original = model
        else:
            require("python" not in provenance and "trusted_pytorch" not in provenance,
                    "Graph source/provenance kind")
            original = provenance["original"]
        return {"model.json": _file(canonical(original))}
    require(type(provenance) is dict, "Python source provenance required")
    if kind == "restricted_python":
        require(profile == "legacy-v1" and "python" in provenance and
                "trusted_pytorch" not in provenance, "Restricted source provenance kind")
        package = provenance["python"]
        weights = {}
    else:
        require(kind == "trusted_pytorch" and "trusted_pytorch" in provenance and
                "python" not in provenance, "Trusted source provenance kind")
        package = provenance["trusted_pytorch"]
        weights = declarations(package["manifest"].get("weights", {}), resource_profile=profile)
    manifest = package["manifest"]
    expected = {"manifest.json": _file(canonical(manifest))}
    require(type(package["files"]) is dict and set(package["files"]) == set(manifest["files"]),
            "Source origin file inventory")
    for name, text in package["files"].items():
        require(type(text) is str, "Source origin text")
        record = _file(text.encode("utf-8"))
        require(record["sha256"] == manifest["files"][name], "Source origin source hash")
        expected[name] = record
    for item in weights.values():
        name = item["file"]
        record = dict(bytes=item["bytes"], sha256=item["sha256"])
        require(name not in expected or expected[name] == record, "Conflicting origin file")
        expected[name] = record
    return expected


def verify(value, request, provenance=None, *, controller_sha256=None):
    if provenance is not None:
        from provenance_json_transport import unpack
        provenance=unpack(provenance,purpose="model-provenance")
    require(type(value) is dict and set(value) == FIELDS and value["format"] == FORMAT,
            "Model source origin fields/version")
    require(_hash(value["binding"]) and digest({k: v for k, v in value.items() if k != "binding"})
            == value["binding"], "Model source origin binding")
    require(value["kind"] in ("graph", "restricted_python", "trusted_pytorch") and
            type(value["resource_profile"]) is str and _hash(value["snapshot_binding"]),
            "Model source origin kind/profile/snapshot")
    resolve(value["resource_profile"])
    _implementation(value["implementation"])
    if controller_sha256 is not None:
        require(_hash(controller_sha256) and
                value["implementation"]["controller_sha256"] == controller_sha256,
                "Model source origin controller/dependency mismatch")
    model = request["model"]
    require(value["resource_profile"] == graph_resource_profile(model) and
            value["model_sha256"] == digest(model) and
            value["provenance_sha256"] == (digest(provenance) if provenance is not None else None),
            "Model source origin model/provenance binding")
    expected = _expected(value["kind"], model, provenance, value["resource_profile"])
    require(value["files"] == expected and len(canonical(value)) <= LIMIT,
            "Model source origin inventory/size")
    return value


def reuse_identity(value):
    """Semantic source identity, independent of a task's operational budget.

    Full snapshot bindings remain in each task and delivered source record.
    Model/request, profile, provenance, file contents and controller still bind
    reuse; changing a deadline or call allowance cannot change the program.
    """
    require(type(value) is dict and set(value) == FIELDS and value["format"] == FORMAT and
            digest({k:v for k,v in value.items() if k != "binding"}) == value["binding"],
            "Model source reuse record")
    return dict(format="poseidon-model-source-reuse-v1",origin_format=value["format"],
                **{k:copy.deepcopy(value[k]) for k in
                   ("kind","resource_profile","files","implementation","model_sha256","provenance_sha256")})


def build(snapshot, request, provenance=None):
    if provenance is not None:
        from provenance_json_transport import unpack
        provenance=unpack(provenance,purpose="model-provenance")
    value = dict(format=FORMAT, kind=snapshot["kind"],
                 resource_profile=snapshot["resource_profile"],
                 snapshot_binding=snapshot["binding"], files=copy.deepcopy(snapshot["files"]),
                 implementation=copy.deepcopy(snapshot["implementation"]),
                 model_sha256=digest(request["model"]),
                 provenance_sha256=digest(provenance) if provenance is not None else None)
    value["binding"] = digest(value)
    return verify(value, request, provenance)


def check_package(folder, manifest, request, provenance=None):
    origin = manifest["origin"].get("model_source")
    if origin is None:
        require(FILE not in manifest["files"], "Unreferenced model source origin")
        return None
    require(manifest["format"] in ("poseidon-dsl-bundle-v5", "poseidon-dsl-bundle-v6") and
            type(origin) is dict and set(origin) == {"format", "file", "sha256"} and
            origin["format"] == FORMAT and origin["file"] == FILE and
            manifest["files"].get(FILE) == origin["sha256"] and _hash(origin["sha256"]),
            "Package model source descriptor")
    from candidate_bundle import package_identities
    from strict_json_files import strict_file
    identities = package_identities(folder, manifest)
    controller = identities["generation"]["sources"].get("application_model_source.py")
    require(_hash(controller), "Model source controller absent from generation identity")
    return verify(strict_file(Path(folder)/FILE, LIMIT), request, provenance,
                  controller_sha256=controller)
