"""V12 trusted-source provenance and finite-reference contract.

This module validates data returned by the isolated local-model capture worker.
It has no source import, Provider, compiler, runtime or generated DSL path.
"""
import ast, base64, copy, hashlib, math, re
from pathlib import Path
from benchmark_graph import require, digest, canonical, NAME
from model_resources import resolve, identity
from model_weight_package import regular_bytes
import trusted_pytorch_import as legacy
from pytorch_spatial_policy import (
    SOURCE_FORMAT as SPATIAL_SOURCE_FORMAT, PACKAGE_FORMAT as SPATIAL_PACKAGE_FORMAT,
    REFERENCE_FORMAT as SPATIAL_REFERENCE_FORMAT, PUBLIC_FORMAT as SPATIAL_PUBLIC_SOURCE_FORMAT,
    CORE_FORMAT as SPATIAL_CORE_SOURCE_FORMAT, PUBLIC_PACKAGE_FORMAT as SPATIAL_PUBLIC_PACKAGE_FORMAT,
    PUBLIC_REFERENCE_FORMAT as SPATIAL_PUBLIC_REFERENCE_FORMAT, CORE_PACKAGE_FORMAT as SPATIAL_CORE_PACKAGE_FORMAT,
    CORE_REFERENCE_FORMAT as SPATIAL_CORE_REFERENCE_FORMAT, INTERNAL_FORMATS as SPATIAL_INTERNAL_FORMATS,
    CAPTURE_FORMAT as SPATIAL_CAPTURE_FORMAT, POLICY as SPATIAL_POLICY, supports_input_call,
)

from pytorch_static_broadcast import SOURCE_FORMAT as BROADCAST_SOURCE_FORMAT, PACKAGE_FORMAT as BROADCAST_PACKAGE_FORMAT, REFERENCE_FORMAT as BROADCAST_REFERENCE_FORMAT

from pytorch_source_package_policy import (
    SOURCE_FORMAT as SOURCE_PACKAGE_FORMAT, PACKAGE_FORMAT as SOURCE_PACKAGE_PROVENANCE,
    REFERENCE_FORMAT as SOURCE_PACKAGE_REFERENCE, selected as packaged_source, envelope as source_package_envelope,
    raw_sources as source_package_bytes,
)

from pytorch_static_tensor_transforms import SOURCE_FORMAT as TENSOR_SOURCE_FORMAT, PACKAGE_FORMAT as TENSOR_PACKAGE_FORMAT, REFERENCE_FORMAT as TENSOR_REFERENCE_FORMAT
from pytorch_public_constant_source import SOURCE_FORMAT as CONSTANT_SOURCE_FORMAT, PACKAGE_FORMAT as CONSTANT_PACKAGE_FORMAT, REFERENCE_FORMAT as CONSTANT_REFERENCE_FORMAT
from pytorch_public_tensor_transform_source import SOURCE_FORMAT as STATE_TRANSFORM_SOURCE_FORMAT, PACKAGE_FORMAT as STATE_TRANSFORM_PACKAGE_FORMAT, REFERENCE_FORMAT as STATE_TRANSFORM_REFERENCE_FORMAT

FORMAT = "poseidon-trusted-pytorch-v12"
PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v12"
REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v12"
COMMON_SOURCE_FORMAT = "poseidon-trusted-pytorch-v13"
COMMON_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v13"
COMMON_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v13"
FUSED_SOURCE_FORMAT = "poseidon-trusted-pytorch-v14"
FUSED_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v14"
FUSED_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v14"
GELU_SOURCE_FORMAT = "poseidon-trusted-pytorch-v15"
GELU_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v15"
GELU_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v15"
MHA_SOURCE_FORMAT = "poseidon-trusted-pytorch-v16"
MHA_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v16"
MHA_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v16"
PARTITION_SOURCE_FORMAT = "poseidon-trusted-pytorch-v17"
PARTITION_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v17"
PARTITION_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v17"
EINSUM_SOURCE_FORMAT = "poseidon-trusted-pytorch-v18"
EINSUM_PACKAGE_FORMAT = "poseidon-trusted-pytorch-provenance-v18"
EINSUM_REFERENCE_FORMAT = "poseidon-trusted-pytorch-reference-v18"
EINSUM_VIEW_SOURCE_FORMAT="poseidon-trusted-pytorch-v19"
EINSUM_VIEW_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v19"
EINSUM_VIEW_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v19"
SCALAR_PARTITION_SOURCE_FORMAT="poseidon-trusted-pytorch-v20"
SCALAR_PARTITION_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v20"
SCALAR_PARTITION_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v20"
EXACT_ARITHMETIC_SOURCE_FORMAT="poseidon-trusted-pytorch-v21"
EXACT_ARITHMETIC_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v21"
EXACT_ARITHMETIC_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v21"
PUBLIC_TENSOR_SOURCE_FORMAT="poseidon-trusted-pytorch-v22"
PUBLIC_TENSOR_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v22"
PUBLIC_TENSOR_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v22"
PUBLIC_EXPRESSION_SOURCE_FORMAT="poseidon-trusted-pytorch-v23"
PUBLIC_EXPRESSION_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v23"
PUBLIC_EXPRESSION_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v23"
PUBLIC_DTYPE_SOURCE_FORMAT="poseidon-trusted-pytorch-v24"
PUBLIC_DTYPE_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v24"
PUBLIC_DTYPE_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v24"
PUBLIC_ARITHMETIC_SOURCE_FORMAT="poseidon-trusted-pytorch-v25"
PUBLIC_ARITHMETIC_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v25"
PUBLIC_ARITHMETIC_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v25"
INPUT_BINDING_SOURCE_FORMAT="poseidon-trusted-pytorch-v26"
INPUT_BINDING_PACKAGE_FORMAT="poseidon-trusted-pytorch-provenance-v26"
INPUT_BINDING_REFERENCE_FORMAT="poseidon-trusted-pytorch-reference-v26"
SOURCE_FORMATS = (FORMAT, COMMON_SOURCE_FORMAT, FUSED_SOURCE_FORMAT, GELU_SOURCE_FORMAT, MHA_SOURCE_FORMAT, PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,PUBLIC_TENSOR_SOURCE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT)

def allows_none_output(spec):
    """Only contracts inheriting the registered static MHA output tree allow None."""
    require(type(spec)is dict,"Static output source contract")
    return spec.get("format")in(MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,
        EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,
        PUBLIC_TENSOR_SOURCE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,*SPATIAL_INTERNAL_FORMATS)

def package_version(spec):
    require(type(spec) is dict and spec.get("format") in (*SOURCE_FORMATS,*SPATIAL_INTERNAL_FORMATS), "Trusted ordered source version")
    return {FORMAT:PACKAGE_FORMAT, COMMON_SOURCE_FORMAT:COMMON_PACKAGE_FORMAT,
            FUSED_SOURCE_FORMAT:FUSED_PACKAGE_FORMAT,GELU_SOURCE_FORMAT:GELU_PACKAGE_FORMAT,MHA_SOURCE_FORMAT:MHA_PACKAGE_FORMAT,PARTITION_SOURCE_FORMAT:PARTITION_PACKAGE_FORMAT,EINSUM_SOURCE_FORMAT:EINSUM_PACKAGE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT:EINSUM_VIEW_PACKAGE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT:SCALAR_PARTITION_PACKAGE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT:EXACT_ARITHMETIC_PACKAGE_FORMAT,PUBLIC_TENSOR_SOURCE_FORMAT:PUBLIC_TENSOR_PACKAGE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT:PUBLIC_EXPRESSION_PACKAGE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT:PUBLIC_DTYPE_PACKAGE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT:PUBLIC_ARITHMETIC_PACKAGE_FORMAT,INPUT_BINDING_SOURCE_FORMAT:INPUT_BINDING_PACKAGE_FORMAT,SPATIAL_SOURCE_FORMAT:SPATIAL_PACKAGE_FORMAT,BROADCAST_SOURCE_FORMAT:BROADCAST_PACKAGE_FORMAT,SOURCE_PACKAGE_FORMAT:SOURCE_PACKAGE_PROVENANCE,TENSOR_SOURCE_FORMAT:TENSOR_PACKAGE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT:STATE_TRANSFORM_PACKAGE_FORMAT,CONSTANT_SOURCE_FORMAT:CONSTANT_PACKAGE_FORMAT,SPATIAL_PUBLIC_SOURCE_FORMAT:SPATIAL_PUBLIC_PACKAGE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT:SPATIAL_CORE_PACKAGE_FORMAT}[spec["format"]]

def reference_version(spec):
    require(type(spec) is dict and spec.get("format") in (*SOURCE_FORMATS,*SPATIAL_INTERNAL_FORMATS), "Trusted ordered source version")
    return {FORMAT:REFERENCE_FORMAT, COMMON_SOURCE_FORMAT:COMMON_REFERENCE_FORMAT,
            FUSED_SOURCE_FORMAT:FUSED_REFERENCE_FORMAT,GELU_SOURCE_FORMAT:GELU_REFERENCE_FORMAT,MHA_SOURCE_FORMAT:MHA_REFERENCE_FORMAT,PARTITION_SOURCE_FORMAT:PARTITION_REFERENCE_FORMAT,EINSUM_SOURCE_FORMAT:EINSUM_REFERENCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT:EINSUM_VIEW_REFERENCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT:SCALAR_PARTITION_REFERENCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT:EXACT_ARITHMETIC_REFERENCE_FORMAT,PUBLIC_TENSOR_SOURCE_FORMAT:PUBLIC_TENSOR_REFERENCE_FORMAT,PUBLIC_EXPRESSION_SOURCE_FORMAT:PUBLIC_EXPRESSION_REFERENCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT:PUBLIC_DTYPE_REFERENCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT:PUBLIC_ARITHMETIC_REFERENCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT:INPUT_BINDING_REFERENCE_FORMAT,SPATIAL_SOURCE_FORMAT:SPATIAL_REFERENCE_FORMAT,BROADCAST_SOURCE_FORMAT:BROADCAST_REFERENCE_FORMAT,SOURCE_PACKAGE_FORMAT:SOURCE_PACKAGE_REFERENCE,TENSOR_SOURCE_FORMAT:TENSOR_REFERENCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT:STATE_TRANSFORM_REFERENCE_FORMAT,CONSTANT_SOURCE_FORMAT:CONSTANT_REFERENCE_FORMAT,SPATIAL_PUBLIC_SOURCE_FORMAT:SPATIAL_PUBLIC_REFERENCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT:SPATIAL_CORE_REFERENCE_FORMAT}[spec["format"]]
EXTRA_IMPLEMENTATION = (
    "pytorch_arithmetic_normalization.py", "operator_graph.py",
    "pytorch_alias_composition.py", "pytorch_alias_capture_bridge.py",
    "pytorch_ordered_capture.py", "pytorch_ordered_reference.py",
    "pytorch_ordered_contract.py", "pytorch_ordered_intake.py",
    "strict_json_files.py",
)
IMPLEMENTATION = tuple(dict.fromkeys((*legacy.IMPLEMENTATION, *EXTRA_IMPLEMENTATION)))
RESERVED = {"pytorch_model_intake.py","candidate_bundle.py","application_capabilities.py"}

def shim(spec):
    require(type(spec) is dict and spec.get("format") in (*SOURCE_FORMATS,*SPATIAL_INTERNAL_FORMATS), "Trusted ordered source version")
    require(len(canonical(spec)) <= 131072, "Trusted manifest data/budget")
    result = copy.deepcopy(spec)
    if spec["format"]in SPATIAL_INTERNAL_FORMATS:
        require("call"not in spec,"Internal spatial capture uses flat inputs")
    if supports_input_call(spec):
        from pytorch_input_binding import validate_call
        validate_call(spec.get("call"),spec.get("inputs"))
        result.pop("call")
    result["format"] = legacy.SOURCE_V7
    return result

def validate_package(spec, files):
    require(type(spec)is dict and spec.get("format")in SOURCE_FORMATS,"Public trusted source format")
    if packaged_source(spec):
        from pytorch_source_package import validate
        legacy.validate_manifest(shim(spec))
        validate(source_package_envelope(spec), source_package_bytes(files),
                 reserved_module_roots=source_package_reserved_roots())
        return spec
    legacy.validate_package(shim(spec), files)
    require(not set(files).intersection(set(implementation_paths())|RESERVED), "Source conflicts with trusted implementation")
    return spec


from functools import lru_cache

@lru_cache(maxsize=256)
def _dependency_names(raw):
    """Cache only immutable syntax, keyed by the complete source bytes."""
    candidates = set()
    for node in ast.walk(ast.parse(raw.decode("utf-8"))):
        if isinstance(node, ast.Import):
            candidates.update(alias.name.split(".")[0]+".py" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates.add(node.module.split(".")[0]+".py")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "with_name" and len(node.args) == 1
              and isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str):
            adjacent = node.args[0].value
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*\.(py|json)", adjacent):
                candidates.add(adjacent)
    return frozenset(candidates)

def implementation_paths():
    return _implementation_paths_at(Path(__file__).resolve().parent,
                                    Path(legacy.__file__).resolve().parent, IMPLEMENTATION)

def _implementation_paths_at(local, retained, initial):
    # Include project-local imports anywhere in the trusted functions, and
    # adjacent data files explicitly read with Path.with_name. No model imports
    # or module execution are needed to compute this deterministic closure.
    pending = list(initial)
    result = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = local/name if (local/name).is_file() else retained/name
        require(path.is_file(), "Trusted implementation file missing")
        result[name] = path
        if path.suffix != ".py":
            continue
        candidates = _dependency_names(path.read_bytes())
        pending.extend(sorted(n for n in candidates if n not in result
                              and ((local/n).is_file() or (retained/n).is_file())))
    return dict(sorted(result.items()))

def source_identity():
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in implementation_paths().items()}

def source_package_reserved_roots():
    # JSON inventories remain in the SDK hash; they are not importable modules.
    return tuple(sorted({Path(name).stem for name in set(implementation_paths()) | RESERVED
                         if Path(name).suffix == ".py"}))


def source_package_files(spec, root):
    from pytorch_source_package import prepare
    require(packaged_source(spec), "Explicit source package version")
    legacy.validate_manifest(shim(spec))
    record, files = prepare(root, source_package_envelope(spec),
                            reserved_module_roots=source_package_reserved_roots())
    return files


def prepare(manifest_path):
    from strict_json_files import strict_file
    path = Path(manifest_path)
    spec = strict_file(path, 131072)
    legacy.validate_manifest(shim(spec))
    if packaged_source(spec):
        source_package_files(spec, path.parent)
        return spec, {name:path.absolute().parent/name for name in spec["files"]}
    files, paths, total = {}, {}, 0
    for name, expected in spec["files"].items():
        require(type(name) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py", name)
                and name not in (set(implementation_paths())|RESERVED), "Flat source filename")
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "Source hash")
        raw = regular_bytes(path.parent/name, 65536-total)
        total += len(raw)
        files[name], paths[name] = raw.decode(), (path.parent/name).resolve()
    validate_package(spec, files)
    return spec, paths

def check_digest(record, key="binding"):
    require(type(record) is dict and record.get(key) ==
            digest({k:v for k,v in record.items() if k != key}), "Ordered metadata binding")

def check_steps(ids, steps, nodes, refs, seen):
    require(type(ids) is list and len(ids) == len(steps) and len(set(ids)) == len(ids)
            and not seen.intersection(ids), "Emission step inventory")
    for i, (name, step) in enumerate(zip(ids, steps)):
        require(name in nodes, "Emission step missing")
        node = nodes[name]
        require(node["op"] == step["op"] and node["attrs"] == step["attrs"]
                and node["inputs"] == [refs[v] for v in step["inputs"]]
                and len(node["outputs"]) == 1, "Emission step differs from registered plan")
        refs["step_"+str(i)] = node["outputs"][0]
        seen.add(name)

def check_emission_records(metadata, graph, descriptions, spec, resource_profile, *, source_locations=None):
    """Reuse registered index/matrix planners against the preserved emission stream.

    The stream also contains ordered writes and is not a pure value graph. Do not
    pass it to the legacy value-graph Builder or relabel it as a traced INDEX graph.
    """
    from pytorch_static_index import VERSION, plan as index_plan, decode
    from pytorch_matrix_plan import POLICY, plan as matrix_plan, constant_value
    from pytorch_approximation import check_record as check_approximation
    nodes = {n["id"]: n for n in graph["nodes"]}
    require(len(nodes) == len(graph["nodes"]), "Emission node identifiers")
    limits = resolve(resource_profile)
    for key, version in (("index_capture", VERSION), ("matrix_capture", POLICY)):
        record = metadata[key]
        require(type(record) is dict and set(record) == {"version","graph_sha256","entries"}
                and record["version"] == version and record["graph_sha256"] == digest(graph)
                and type(record["entries"]) is list and len(record["entries"]) <= limits.original_nodes,
                "Registered emission record identity")
    seen = set()
    for entry in metadata["index_capture"]["entries"]:
        require(type(entry) is dict and set(entry) ==
                {"operation","source","source_shape","plan","node_ids"}
                and entry["operation"] in ("getitem","select","narrow","index_select"),
                "Index emission record")
        source, dims, declared = entry["source"], entry["source_shape"], entry["plan"]
        require(source in descriptions and type(dims) is list and
                (dims or [1]) == (descriptions[source]["shape"] or [1]), "Index source shape")
        require(type(declared) is dict and type(declared.get("selectors")) is list,
                "Index selection plan")
        actual = index_plan(dims, tuple(decode(v) for v in declared["selectors"]),
                            resource_profile=resource_profile)
        require(actual == declared, "Index selection binding")
        steps = actual["steps"] + [dict(op="reshape", attrs=dict(shape=actual["result_shape"] or [1]))]
        ids = entry["node_ids"]
        require(type(ids) is list and len(ids) == len(steps) and len(set(ids)) == len(ids)
                and not seen.intersection(ids), "Index emission steps")
        previous = source
        for name, step in zip(ids, steps):
            require(name in nodes, "Index emission node missing")
            node = nodes[name]
            require(node["op"] == step["op"] and node["attrs"] == step["attrs"]
                    and node["inputs"] == [previous] and len(node["outputs"]) == 1,
                    "Index emission differs from registered plan")
            previous = node["outputs"][0]
            seen.add(name)
    seen = set()
    for entry in metadata["matrix_capture"]["entries"]:
        require(type(entry) is dict and set(entry) == {"sources","plan","node_ids","constant"},
                "Matrix emission record")
        sources, declared = entry["sources"], entry["plan"]
        require(type(sources) is list and len(sources) == 2 and all(v in descriptions for v in sources)
                and type(declared) is dict and "operation" in declared and "public_side" in declared,
                "Matrix source inventory")
        plain = [descriptions[v]["kind"] == "plain" for v in sources]
        require(not all(plain), "Matrix encrypted dependency")
        side = "left" if plain[0] else "right" if plain[1] else None
        actual = matrix_plan(descriptions[sources[0]]["shape"], descriptions[sources[1]]["shape"],
                             operation=declared["operation"], public_side=side,
                             resource_profile=resource_profile)
        require(actual == declared, "Matrix plan binding")
        refs = dict(zip(("left","right"), sources))
        constant = entry["constant"]
        if actual["constant"] is None:
            require(constant is None, "Unexpected matrix constant")
        else:
            public = sources[0 if side == "left" else 1]
            require(type(constant) is str and constant in graph["constants"]
                    and public in graph["constants"], "Matrix public constant source")
            expected = constant_value(graph["constants"][public], actual)
            require(canonical(graph["constants"][constant]) == canonical(expected), "Derived matrix constant")
            refs["constant"] = constant
        check_steps(entry["node_ids"], actual["steps"], nodes, refs, seen)
    check_approximation(metadata["approximation_selection"], graph, spec["approximations"], source_locations=source_locations)

def check_capture_metadata(metadata, original, spec, *, resource_profile,
                           expected_weight_binding, expected_weight_arrays=None):
    if spec.get("format")in(PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT):
        from pytorch_public_expression_capture import check_capture_metadata as selected
        return selected(metadata,original,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    if spec.get("format")in(PUBLIC_TENSOR_SOURCE_FORMAT,SPATIAL_PUBLIC_SOURCE_FORMAT):
        from pytorch_public_arithmetic_capture import check_capture_metadata as selected
        return selected(metadata,original,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    require(expected_weight_arrays is None,"Public array payloads belong to explicit V22")
    import numpy as np
    from pytorch_static_trace import policy_record, ALIAS_POLICY, INDEX_POLICY
    from pytorch_weight_package import declarations, check_binding
    from pytorch_source_contract import validate_output_tree
    from pytorch_alias_capture_bridge import normalize_emissions, FORMAT as BRIDGE_FORMAT,SCALAR_FORMAT as SCALAR_BRIDGE_FORMAT
    from pytorch_ordered_capture import FORMAT as CAPTURE_FORMAT
    from pytorch_arithmetic_normalization import check_record as check_arithmetic
    from operator_graph import FORMAT as GRAPH_FORMAT
    legacy.validate_manifest(shim(spec))
    fields = {"torch_version","trusted_code","training","graph_sha256","dtype_policy","input_dtype",
              "source_state_dtypes","constant_dtypes","output_structure","source_precision_execution_compared",
              "state_constant_sources","weight_binding","registered_emission_policy",
              "approximation_selection","index_capture","matrix_capture","ordered_alias_capture"}
    mha_selected=spec["format"]in(MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    partition_selected=spec["format"]in(PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    einsum_selected=spec["format"] in (EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    source_selected=spec["format"] in (GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    if source_selected:fields|={"gelu_source_capture"}
    if mha_selected:fields|={"mha_source_capture"}
    spatial_selected=spec["format"]==SPATIAL_CORE_SOURCE_FORMAT
    if spatial_selected:fields.add("spatial_source_capture")
    require(type(metadata) is dict and set(metadata) == fields, "Ordered capture metadata fields")
    require(original.get("format") == ("poseidon-operator-graph-v4"if spatial_selected else GRAPH_FORMAT) and original.get("inputs") == spec["inputs"]
            and [o["name"] for o in original.get("outputs",[])] == spec["outputs"], "Ordered graph signature")
    require(metadata["torch_version"] == legacy.runtime_identity()["torch"]
            and metadata["trusted_code"] is True and metadata["training"] is False
            and metadata["graph_sha256"] == digest(original), "Ordered capture source/runtime")
    require(metadata["dtype_policy"] == spec["dtype_policy"] and metadata["input_dtype"] == "float64"
            and metadata["source_precision_execution_compared"] is False, "Ordered dtype policy")
    records = declarations(spec["weights"], resource_profile=resource_profile)
    check_binding(metadata["weight_binding"], records)
    check_binding(expected_weight_binding, records)
    require(canonical(metadata["weight_binding"]) == canonical(expected_weight_binding),
            "Loaded weight content differs from declared package")
    dtype_names = {"<f4":"float32","<f8":"float64","<i4":"int32","<i8":"int64","|b1":"bool"}
    require(metadata["source_state_dtypes"] == {n:dtype_names[v["dtype"]] for n,v in records.items()},
            "Original state dtype inventory")
    for key in ("source_state_dtypes","constant_dtypes"):
        value = metadata[key]
        require(type(value) is dict and len(value) <= 1024 and all(
                type(n) is str and 0 < len(n) <= 256 and d in dtype_names.values()
                for n,d in value.items()), "Ordered dtype records")
    if spec["dtype_policy"] == "strict-float64-v1":
        require("float32" not in metadata["source_state_dtypes"].values()
                and "float32" not in metadata["constant_dtypes"].values(), "Strict ordered dtype")
    validate_output_tree(metadata["output_structure"], spec["outputs"], allow_scalars=True, allow_none=mha_selected)
    require(metadata["registered_emission_policy"] ==
            policy_record(spec["inputs"], resource_profile, policy=INDEX_POLICY),
            "Registered emission rule selection")
    alias = metadata["ordered_alias_capture"]
    check_digest(alias)
    require(alias.get("format") == (SCALAR_BRIDGE_FORMAT if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT) else BRIDGE_FORMAT) and alias.get("source_capture_policy") ==
            policy_record(spec["inputs"], resource_profile, policy=ALIAS_POLICY), "Actual ordered trace policy")
    emissions = alias.get("capture_emissions")
    require(type(emissions) is dict and digest(emissions) == alias.get("capture_emissions_sha256"),
            "Preserved emission graph binding")
    adapter = alias.get("capture_adapter")
    check_digest(adapter)
    from pytorch_ordered_capture import COMMON_FORMAT as COMMON_CAPTURE_FORMAT
    common_selected=spec["format"] in (COMMON_SOURCE_FORMAT,FUSED_SOURCE_FORMAT,GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    fused_selected=spec["format"] in (FUSED_SOURCE_FORMAT,GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    from pytorch_ordered_capture import FUSED_FORMAT as FUSED_CAPTURE_FORMAT,GELU_FORMAT as GELU_CAPTURE_FORMAT
    from pytorch_ordered_capture import MHA_FORMAT as MHA_CAPTURE_FORMAT, MHA_POLICY, MHA_IMPLEMENTATION
    from pytorch_ordered_capture import PARTITION_FORMAT as PARTITION_CAPTURE_FORMAT,SCALAR_PARTITION_FORMAT as SCALAR_PARTITION_CAPTURE_FORMAT
    from pytorch_ordered_capture import EINSUM_FORMAT as EINSUM_CAPTURE_FORMAT,EINSUM_VIEW_FORMAT as EINSUM_VIEW_CAPTURE_FORMAT
    from pytorch_ordered_capture import EXACT_ARITHMETIC_FORMAT as EXACT_CAPTURE_FORMAT
    exact_selected=spec["format"]in(EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    require(("exact_arithmetic_normalization" in adapter)==exact_selected,"Explicit exact arithmetic source capture version")
    if exact_selected:
        from pytorch_exact_arithmetic import check_transport as check_exact
        exact=adapter["exact_arithmetic_normalization"]
        check_exact(exact,inputs=spec["inputs"],resource_profile=resource_profile)
        require(exact["source_fx_sha256"]==adapter["source_fx_sha256"],"Exact arithmetic original source identity")
    require(("einsum_normalization" in adapter)==einsum_selected,"Explicit einsum source capture version")
    if einsum_selected:
        from pytorch_einsum_normalization import check_transport as check_einsum
        if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT):
            from pytorch_einsum_extension import check_transport as check_einsum
        einsum=adapter["einsum_normalization"]
        check_einsum(einsum,inputs=spec["inputs"],resource_profile=resource_profile)
        require(einsum["source_fx_sha256"]==(exact["normalized_fx_sha256"]if exact_selected else adapter["source_fx_sha256"]),"Einsum original source identity")
        if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT):
            from pytorch_einsum_extension import check_alias_projection
            check_alias_projection(einsum,emissions,alias["capture_source_origins"],inputs=spec["inputs"])
    require(("partition_normalization" in adapter)==partition_selected,"Explicit partition source capture version")
    if partition_selected:
        from pytorch_partition_normalization import check_transport as check_partition
        partition=adapter["partition_normalization"]
        check_partition(partition,inputs=spec["inputs"],resource_profile=resource_profile,
                        **({"source_policy":"poseidon-pytorch-static-partition-v2"}if spec["format"] in (SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT) else {}))
        require(partition["source_fx_sha256"]==(einsum["normalized_fx_sha256"]if einsum_selected else adapter["source_fx_sha256"]),"Partition original source identity")
    require(("mha_source_policy" in adapter)==mha_selected,"Explicit MHA source capture version")
    if mha_selected:require(adapter["mha_source_policy"]==MHA_POLICY,"MHA source policy binding")
    require(("gelu_source_policy" in adapter)==source_selected,"Explicit GELU source capture version")
    require(("fused_matrix_normalization" in adapter)==fused_selected,"Explicit fused matrix capture version")
    require(("common_normalization" in adapter)==common_selected, "Explicit ordered common capture version")
    require(("spatial_source_policy"in adapter)==spatial_selected,"Explicit spatial capture policy")
    if spatial_selected:require(adapter["spatial_source_policy"]==SPATIAL_POLICY,"Spatial policy binding")
    require(adapter.get("format") == (SPATIAL_CAPTURE_FORMAT if spatial_selected else EXACT_CAPTURE_FORMAT if exact_selected else SCALAR_PARTITION_CAPTURE_FORMAT if spec["format"] in (SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT) else EINSUM_VIEW_CAPTURE_FORMAT if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT) else EINSUM_CAPTURE_FORMAT if einsum_selected else PARTITION_CAPTURE_FORMAT if partition_selected else MHA_CAPTURE_FORMAT if mha_selected else GELU_CAPTURE_FORMAT if source_selected else FUSED_CAPTURE_FORMAT if fused_selected else COMMON_CAPTURE_FORMAT if common_selected else CAPTURE_FORMAT)
            and adapter.get("resource_profile") == resource_profile
            and adapter.get("graph_sha256") == digest(original)
            and adapter.get("source_state_dtypes") == metadata["source_state_dtypes"]
            and adapter.get("reference_inputs") == "separate clones for every execution"
            and adapter.get("independent_reference_required") is True, "Ordered adapter binding")
    implementation = implementation_paths()
    expected_adapter = {name:hashlib.sha256(implementation[name].read_bytes()).hexdigest()
                        for name in adapter.get("implementation",{}) if name in implementation}
    expected_names={"pytorch_ordered_capture.py","pytorch_model_capture.py","pytorch_static_trace.py",
                    "pytorch_alias_capture_bridge.py","pytorch_alias_composition.py","pytorch_arithmetic_normalization.py"}
    if spatial_selected:expected_names.update(("pytorch_spatial_source.py","pytorch_spatial_policy.py","operator_spatial_graph.py","operator_spatial_reference.py"))
    if exact_selected:expected_names.add("pytorch_exact_arithmetic.py")
    if fused_selected:expected_names.add("pytorch_fused_matrix_normalization.py")
    if source_selected:expected_names.add("pytorch_gelu_source.py")
    if mha_selected:expected_names.update(MHA_IMPLEMENTATION)
    if partition_selected:expected_names.add("pytorch_partition_normalization.py")
    if einsum_selected:expected_names.add("pytorch_einsum_normalization.py")
    if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT):expected_names.add("pytorch_einsum_extension.py")
    require(adapter.get("implementation") == expected_adapter and set(expected_adapter) == expected_names,
            "Ordered adapter implementation")
    for text_key, sha_key in (("source_fx_text","source_fx_sha256"),("normalized_fx_text","normalized_fx_sha256")):
        text = adapter.get(text_key)
        require(type(text) is str and len(text.encode()) <= resolve(resource_profile).json_bytes
                and hashlib.sha256(text.encode()).hexdigest() == adapter.get(sha_key),
                "Ordered FX text binding")
    require(("public_view_policy"in adapter.get("arithmetic",{}))==
            (spec["format"] in (SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)),"Explicit public arithmetic source version")
    check_arithmetic(adapter.get("arithmetic"), emissions)
    if common_selected:
        from pytorch_arithmetic_normalization import check_common_record, ORDERED_COMMON_POLICY
        common=adapter["common_normalization"]
        check_common_record(common,emissions,adapter["arithmetic"])
        require(common["format"]==ORDERED_COMMON_POLICY and common["source_fx_sha256"]==
                (adapter["fused_matrix_normalization"]["normalized_fx_sha256"] if fused_selected else adapter["source_fx_sha256"])
                and common["resource_profile"]==resource_profile, "Ordered common source identity")
    if fused_selected:
        from pytorch_fused_matrix_normalization import check_transport
        fused=adapter["fused_matrix_normalization"]
        check_transport(fused,inputs=spec["inputs"],resource_profile=resource_profile)
        require(fused["source_fx_sha256"]==(partition["normalized_fx_sha256"] if partition_selected else adapter["source_fx_sha256"]) and set(fused["public_state"])==set(records),
                "Fused matrix original source identity")
        if exact_selected:
            require(exact["public_state"]==fused["public_state"],"Exact arithmetic declared public state binding")
        for name,item in fused["public_state"].items():
            require(type(item) is dict and set(item)=={"shape","dtype","sha256","normalized_float64_sha256"} and
                    item["shape"]==records[name]["shape"] and item["dtype"]==
                    ("<f8" if records[name]["dtype"] in ("<f4","<f8") else records[name]["dtype"]) and
                    item["normalized_float64_sha256"]==expected_weight_binding["normalized_float64_sha256"][name],
                    "Fused matrix source state differs from declared weights")
    rebuilt, lowered, _, regenerated = normalize_emissions(
        emissions, alias["capture_source_origins"], resource_profile=resource_profile,
        constant_dtypes=alias["capture_emission_constant_dtypes"],
        state_constant_sources=alias["capture_emission_state_constant_sources"],
        public_constant_views=alias["capture_public_constant_views"],
        **({"spatial_source_policy":SPATIAL_POLICY}if spatial_selected else {}),
        **({"scalar_views":True}if spec["format"] in (EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT) else {}))
    projection = {k:v for k,v in alias.items() if k not in ("capture_adapter","source_capture_policy","binding")}
    require(canonical(projection) == canonical({k:v for k,v in regenerated.items() if k != "binding"})
            and canonical(rebuilt) == canonical(original), "Ordered source normalization/provenance")
    require(metadata["constant_dtypes"] == regenerated["constant_dtypes"]
            and metadata["state_constant_sources"] == regenerated["state_constant_sources"],
            "Normalized constant provenance")
    mapping = alias["capture_emission_state_constant_sources"]
    if common_selected:
        from pytorch_common_constant_provenance import check as check_constants
        check_constants(emissions,common,mapping,records,expected_weight_binding,
                        resource_profile=resource_profile)
    else:
        require(type(mapping) is dict and set(mapping) <= set(emissions["constants"])
                and all(type(v) is str and v in records for v in mapping.values()), "Emission state sources")
        for name, state in mapping.items():
            value = np.asarray(emissions["constants"][name], dtype="<f8")
            require(list(value.shape) == records[state]["shape"]
                    and hashlib.sha256(value.tobytes(order="C")).hexdigest() ==
                        expected_weight_binding["normalized_float64_sha256"][state],
                    "Emission constant differs from loaded state")
    source_locations=None
    if mha_selected:
        from pytorch_mha_source import check_record as check_mha
        source_locations=check_mha(metadata["mha_source_capture"],original,emissions=emissions,
            normalized_origins=regenerated["normalization"]["mathematical_node_origins"],
            source_origins=alias["capture_source_origins"],public_constant_views=alias["capture_public_constant_views"],
            state_constant_sources=mapping,declared_records=records,
            expected_weight_binding=expected_weight_binding,resource_profile=resource_profile)
    check_emission_records(metadata, emissions, regenerated["normalization"]["source_value_descriptions"],
                           spec, resource_profile,source_locations=source_locations)
    if source_selected:
        from pytorch_gelu_source import POLICY as GELU_POLICY,check_record as check_gelu
        require(adapter["gelu_source_policy"]==GELU_POLICY,"GELU source policy binding")
        check_gelu(metadata["gelu_source_capture"],original,emissions=emissions,
                   normalized_origins=regenerated["normalization"]["mathematical_node_origins"],
                   resource_profile=resource_profile)
    if spatial_selected:
        from pytorch_spatial_source import check_record as check_spatial
        check_spatial(metadata["spatial_source_capture"],emissions,
            regenerated["normalization"]["source_value_descriptions"],
            source_origins=alias["capture_source_origins"],resource_profile=resource_profile)
    return lowered

def capture_binding(original, spec, *, resource_profile, capture_metadata, expected_weight_binding, expected_weight_arrays=None):
    from operator_graph import decompose
    if original.get("format")=="poseidon-operator-graph-v4":
        require(spec.get("format")in(SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT,*SPATIAL_INTERNAL_FORMATS),"Explicit spatial decomposition contract")
        from operator_spatial_graph import decompose
    checked = check_capture_metadata(capture_metadata, original, spec, resource_profile=resource_profile,
                                     expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    lowered, binding = decompose(original, resource_profile=resource_profile)
    require(canonical(lowered) == canonical(checked), "Ordered decomposition differs")
    binding.update(trusted_source_hashes=spec["files"], manifest_sha256=digest(spec),
                   capture_implementation=source_identity(), capture_torch_version=legacy.runtime_identity()["torch"],
                   pytorch_capture=copy.deepcopy(capture_metadata), capture_policy_version=spec["format"])
    binding.pop("binding_sha256", None)
    binding["binding_sha256"] = digest(binding)
    return lowered, binding

def check_reference(report, original, lowered, spec, *, resource_profile, expected_weight_binding, expected_weight_arrays=None):
    if spec.get("format")in(PUBLIC_EXPRESSION_SOURCE_FORMAT,PUBLIC_DTYPE_SOURCE_FORMAT,PUBLIC_ARITHMETIC_SOURCE_FORMAT,INPUT_BINDING_SOURCE_FORMAT,SPATIAL_SOURCE_FORMAT,BROADCAST_SOURCE_FORMAT,SOURCE_PACKAGE_FORMAT,TENSOR_SOURCE_FORMAT,STATE_TRANSFORM_SOURCE_FORMAT,CONSTANT_SOURCE_FORMAT):
        from pytorch_public_expression_capture import check_reference as selected
        return selected(report,original,lowered,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    if spec.get("format")in(PUBLIC_TENSOR_SOURCE_FORMAT,SPATIAL_PUBLIC_SOURCE_FORMAT):
        from pytorch_public_arithmetic_capture import check_reference as selected
        return selected(report,original,lowered,spec,resource_profile=resource_profile,
                        expected_weight_binding=expected_weight_binding,expected_weight_arrays=expected_weight_arrays)
    require(expected_weight_arrays is None,"Public array payloads belong to explicit V22")
    require(type(report) is dict and report.get("format") == reference_version(spec)
            and report.get("mode") == "isolated_trusted_pytorch_reference"
            and report.get("original_sha256") == digest(original) and report.get("model_sha256") == digest(lowered)
            and report.get("manifest_sha256") == digest(spec) and report.get("source_hashes") == source_identity(),
            "Ordered reference identity")
    require(report.get("runtime") == legacy.runtime_identity() and report.get("probes") == 16
            and report.get("resource_profile") == identity(resource_profile), "Ordered reference environment")
    require(report.get("reference_status") == "passed" and report.get("reference_dtype") == "float64"
            and report.get("source_precision_execution_compared") is False
            and report.get("runtime_layout_validated") is False and report.get("encrypted_execution") is False
            and report.get("agent_generated") is False and report.get("paid_calls") == 0
            and report.get("approximation_accuracy_certified") is False, "Ordered reference scope")
    finite = report.get("finite_reference")
    mha_selected=spec["format"]in(MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    partition_selected=spec["format"]in(PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    einsum_selected=spec["format"] in (EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    source_selected=spec["format"] in (GELU_SOURCE_FORMAT,MHA_SOURCE_FORMAT,PARTITION_SOURCE_FORMAT,EINSUM_SOURCE_FORMAT,EINSUM_VIEW_SOURCE_FORMAT,SCALAR_PARTITION_SOURCE_FORMAT,EXACT_ARITHMETIC_SOURCE_FORMAT,SPATIAL_CORE_SOURCE_FORMAT)
    if mha_selected:
        from pytorch_mha_ordered_reference import FORMAT as FINITE_FORMAT
        reference_file="pytorch_mha_ordered_reference.py"
        require(finite.get("mha_source_binding")==report.get("capture_metadata",{}).get(
            "mha_source_capture",{}).get("binding"),"MHA finite reference source binding")
        require(finite.get("gelu_source_binding")==report.get("capture_metadata",{}).get(
            "gelu_source_capture",{}).get("binding"),"GELU finite reference source binding")
    elif source_selected:
        from pytorch_gelu_reference import FORMAT as FINITE_FORMAT
        reference_file="pytorch_gelu_reference.py"
        require(finite.get("gelu_source_binding")==report.get("capture_metadata",{}).get(
            "gelu_source_capture",{}).get("binding"),"GELU finite reference source binding")
    else:
        from pytorch_ordered_reference import FORMAT as FINITE_FORMAT
        reference_file="pytorch_ordered_reference.py"
    check_digest(finite)
    require(finite.get("format") == FINITE_FORMAT and finite.get("status") == "passed"
            and finite.get("groups") == 16 and finite.get("graph_sha256") == digest(original)
            and finite.get("lowered_sha256") == digest(lowered)
            and finite.get("reference_implementation_sha256") == source_identity()[reference_file]
            and finite.get("comparison_tolerance") == dict(atol=1e-12, rtol=1e-12)
            and finite.get("input_storage_isolated") is True and finite.get("public_state_immutable") is True
            and finite.get("physical_layout_validated") is False and finite.get("compiler_validated") is False
            and finite.get("encrypted_execution_validated") is False and finite.get("paid_calls") == 0,
            "Finite ordered reference receipt")
    from benchmark_graph import validate
    output_shapes = validate(lowered,resource_profile=resource_profile)["output_shapes"]
    require(type(finite.get("compared_values")) is int and finite["compared_values"] ==
            16*sum(math.prod(shape) for shape in output_shapes.values()), "Finite compared value count")
    for name in ("max_original_torch_vs_independent_ideal","max_original_vs_normalized_torch",
                 "max_decomposition_error","max_approximation_error","max_reference_tolerance_ratio",
                 "max_normalization_tolerance_ratio"):
        value = finite.get(name)
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0, "Finite reference metric")
    require(finite["max_reference_tolerance_ratio"] <= 1 and finite["max_normalization_tolerance_ratio"] <= 1,
            "Finite reference comparison failed")
    checked, binding = capture_binding(original, spec, resource_profile=resource_profile,
                                      capture_metadata=report.get("capture_metadata"),
                                      expected_weight_binding=expected_weight_binding)
    require(canonical(checked) == canonical(lowered) and finite.get("capture_binding_sha256") ==
            binding["binding_sha256"], "Finite capture binding")
    return report

def encode_weight_files(spec, raw_files, *, resource_profile):
    """Public NPY contents retained for V12 provenance; never pickle."""
    from pytorch_weight_package import declarations, decode
    records = declarations(spec["weights"], resource_profile=resource_profile)
    inventory = {v["file"] for v in records.values()}
    require(type(raw_files) is dict and set(raw_files) == inventory, "Retained weight file inventory")
    for record in records.values():
        decode(raw_files[record["file"]], record)
    return {name:base64.b64encode(raw).decode("ascii") for name,raw in raw_files.items()}

def weight_arrays_from_payload(spec, payload, *, resource_profile):
    from pytorch_weight_package import declarations, decode, binding
    records = declarations(spec["weights"], resource_profile=resource_profile)
    require(type(payload) is dict and set(payload) == {v["file"] for v in records.values()},
            "Retained weight payload inventory")
    decoded = {}
    for record in records.values():
        name = record["file"]
        encoded = payload[name]
        require(type(encoded) is str and len(encoded) == 4*((record["bytes"]+2)//3),
                "Retained weight payload size")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError,base64.binascii.Error) as error:
            raise ValueError("Retained weight payload encoding") from error
        require(base64.b64encode(raw).decode("ascii") == encoded, "Canonical retained weight encoding")
        decoded[name] = raw
    arrays = {name:decode(decoded[record["file"]],record) for name,record in records.items()}
    return arrays

def weight_binding_from_payload(spec, payload, *, resource_profile):
    from pytorch_weight_package import binding
    return binding(weight_arrays_from_payload(spec,payload,resource_profile=resource_profile),spec["weights"])
