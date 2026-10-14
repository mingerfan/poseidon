"""Independent public-state and generated-constant checks for ordered capture.

No source import, FX execution, DSL generation or backend call occurs here.
"""
import hashlib, math
from benchmark_graph import canonical, require
from pytorch_weight_package import check_binding
from model_resources import resolve

def check(graph, record, mapping, records, expected_weights, *, resource_profile):
    import numpy as np
    from pytorch_arithmetic_normalization import ORDERED_COMMON_POLICY
    require(type(record) is dict and record.get("format")==ORDERED_COMMON_POLICY,
            "Ordered common constant policy")
    check_binding(expected_weights, records)
    limits=resolve(resource_profile)
    require(type(mapping) is dict and set(mapping)<=set(graph["constants"]) and
            all(type(v) is str for v in mapping.values()), "Emission state sources")
    items=record.get("generated_constants")
    require(type(items) is list and len(items)<=limits.constants, "Generated state capacity")
    generated={item["target"]:item for item in items}
    require(len(generated)==len(items) and not set(generated).intersection(records),
            "Generated state name collision")
    entries={e["source_node"]:e for e in record["entries"]}
    require(len(entries)==len(record["entries"]), "Common source node uniqueness")
    public={}
    for name,state in mapping.items():
        require(state in records or state in generated, "Undeclared emission state")
        if state not in records:continue
        value=np.asarray(graph["constants"][name],dtype="<f8")
        require(list(value.shape)==records[state]["shape"] and np.isfinite(value).all() and
                hashlib.sha256(value.tobytes(order="C")).hexdigest()==
                    expected_weights["normalized_float64_sha256"][state],
                "Emission constant differs from loaded state")
        if state in public:
            require(np.array_equal(public[state],value), "Repeated public state differs")
        public[state]=value
    for target,item in generated.items():
        names=[name for name,state in mapping.items() if state==target]
        require(names==[item["captured_constant"]], "Generated state emission binding")
        suffix="_alpha_constant" if item["kind"]=="alpha_constant" else "_mask"
        require(target.startswith("_poseidon_common_") and target.endswith(suffix),
                "Generated state target form")
        node=target[len("_poseidon_common_"):-len(suffix)]
        require(node in entries, "Generated state source operation")
        entry=entries[node];origin=item["derivation"]
        require(type(origin) is dict, "Generated constant derivation fields")
        if item["kind"]=="alpha_constant":
            require(set(origin)=={"rule","source","alpha"} and origin["rule"]=="state_alpha" and
                    entry["rule"]=="static_alpha" and origin["alpha"]==entry["attrs"]["alpha"] and
                    type(origin["alpha"]) in (int,float) and math.isfinite(origin["alpha"]),
                    "Alpha derivation rule")
            state=origin["source"]
            require(state in public and records[state]["dtype"] in ("<f4","<f8","<i4","<i8"),
                    "Alpha derivation source binding")
            expected=public[state]*origin["alpha"]
        elif origin.get("rule")=="state_boolean_mask":
            require(set(origin)=={"rule","source"} and entry["rule"]=="sdpa" and
                    entry["attrs"]["mask_kind"]=="boolean",
                    "Boolean mask derivation rule")
            state=origin["source"]
            require(state in records and records[state]["dtype"]=="|b1" and
                    item["shape"]==records[state]["shape"] and
                    item["float64_sha256"]==expected_weights["normalized_float64_sha256"][state],
                    "Boolean mask original state binding")
            expected=np.asarray(graph["constants"][item["captured_constant"]],dtype="<f8")
        else:
            require(set(origin)=={"rule","query_length","key_length"} and origin["rule"]=="causal_triangle" and
                    entry["rule"]=="sdpa" and entry["attrs"]["mask_kind"]=="causal" and
                    type(origin["query_length"]) is int and type(origin["key_length"]) is int and
                    origin["query_length"]==entry["attrs"]["query_shape"][-2] and
                    origin["key_length"]==entry["attrs"]["key_shape"][-2],
                    "Causal mask derivation rule")
            expected=(np.arange(origin["query_length"])[:,None]>=
                      np.arange(origin["key_length"])[None,:]).astype("<f8")
        actual=np.asarray(graph["constants"][item["captured_constant"]],dtype="<f8")
        require(list(expected.shape)==item["shape"] and np.isfinite(expected).all() and
                np.array_equal(actual,expected) and
                hashlib.sha256(expected.tobytes(order="C")).hexdigest()==item["float64_sha256"],
                "Generated state differs from original derivation")
    return dict(declared_state_reads=len(public),generated_state=len(generated),
                loaded_weights_sha256=expected_weights["declarations_sha256"])
