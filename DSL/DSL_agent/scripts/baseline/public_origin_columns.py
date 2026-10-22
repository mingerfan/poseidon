"""Bounded, lossless public metadata columns; no DSL or numeric execution."""
import copy
import math
from benchmark_graph import canonical, digest, require

POLICY = "descriptors-compact-origins-v3"
WEIGHT_FORMAT = "poseidon-public-weight-descriptors-v3"
FORMAT = "poseidon-public-origins-v1"
ROWS_FORMAT = "poseidon-origin-binding-columns-v1"
MAX_RAW_BYTES = 4 * 1024**2
MAX_ROWS = 32768
MAX_COLUMNS = 32
MAX_DEPTH = 64
MAX_ITEMS = 500000
RULES = """
Public origin columns v1:
The public model/constants descriptors occur once. The model_constants and
public_constants fields of public_weight_binding are fixed namespace references
with content hashes, referring to model.constants and public_constants.
constant_origins.value retains the original metadata. Only
capacity_tensor_packing.bindings uses column groups. Each group records count
and sorted column names. The nth binding combines the nth value of every column;
groups concatenate in order. A constant column repeats value count times;
literal supplies its exact values; rle expands each [value, count] run in order.
typed_request_bindings.constant_bindings refers to every public_constants name
in sorted order. Each binding has type prebound_public_plain, immutable true,
and the corresponding descriptor shape.
All rows, operand/chunk/rotation indices, names and roles remain exact.
content_sha256 binds the canonical expanded original constant_origins.
These are public binding metadata, not a DSL implementation. Use only bound
Plain names and the existing logical model, packing and rotation rules.
"""

def _tree(value):
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        require(visited <= MAX_ITEMS and depth <= MAX_DEPTH, "Public origin tree budget")
        if type(item) is dict:
            require(all(type(k) is str for k in item), "Public origin keys")
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
        else:
            require(type(item) in (str, int, float, bool, type(None)), "Public origin JSON value")
            require(type(item) is not float or math.isfinite(item), "Finite public origin value")

def _column(values):
    runs = []
    prior = None
    for value in values:
        encoded = canonical(value)
        if runs and encoded == prior:
            runs[-1][1] += 1
        else:
            runs.append([copy.deepcopy(value), 1])
            prior = encoded
    candidates = [
        dict(encoding="literal", values=copy.deepcopy(values)),
        dict(encoding="rle", runs=runs),
    ]
    if len(runs) == 1:
        candidates.append(dict(encoding="constant", value=copy.deepcopy(values[0])))
    return min(candidates, key=lambda v: (len(canonical(v)), canonical(v)))

def _pack_rows(rows):
    require(type(rows) is list and len(rows) <= MAX_ROWS, "Public origin row budget")
    groups = []
    begin = 0
    while begin < len(rows):
        row = rows[begin]
        require(type(row) is dict and 1 <= len(row) <= MAX_COLUMNS, "Public origin row fields")
        columns = sorted(row)
        end = begin + 1
        while end < len(rows) and type(rows[end]) is dict and sorted(rows[end]) == columns:
            end += 1
        block = rows[begin:end]
        groups.append(dict(count=end - begin, columns=columns,
            data={key: _column([r[key] for r in block]) for key in columns}))
        begin = end
    return dict(format=ROWS_FORMAT, row_count=len(rows), groups=groups)

def compact(original):
    require(type(original) is dict and original.get("format") != FORMAT, "Original public origin object")
    _tree(original)
    require(len(canonical(original)) <= MAX_RAW_BYTES, "Public origin expanded byte budget")
    value = copy.deepcopy(original)
    packing = value.get("capacity_tensor_packing")
    if packing is not None:
        require(type(packing) is dict and type(packing.get("bindings")) is list, "Public origin packing")
        packing["bindings"] = _pack_rows(packing["bindings"])
    return dict(format=FORMAT, content_sha256=digest(original), value=value)

def _encoded_column(column, count):
    require(type(column) is dict, "Public origin column")
    encoding = column.get("encoding")
    if encoding == "constant":
        require(set(column) == {"encoding", "value"}, "Public origin constant fields")
        runs = [(column["value"], count)]
    elif encoding == "literal":
        require(set(column) == {"encoding", "values"} and type(column["values"]) is list
            and len(column["values"]) == count, "Public origin literal count")
        runs = [(v, 1) for v in column["values"]]
    elif encoding == "rle":
        require(set(column) == {"encoding", "runs"} and type(column["runs"]) is list
            and 1 <= len(column["runs"]) <= count, "Public origin run fields")
        runs = []
        for run in column["runs"]:
            require(type(run) is list and len(run) == 2 and type(run[1]) is int
                and 1 <= run[1] <= count, "Public origin run count")
            runs.append((run[0], run[1]))
    else:
        require(False, "Public origin column encoding")
    require(sum(n for _, n in runs) == count, "Public origin expanded column count")
    byte_count = 0
    for value, repeats in runs:
        _tree(value)
        byte_count += len(canonical(value)) * repeats
        require(byte_count <= MAX_RAW_BYTES, "Public origin expanded byte budget")
    return runs, byte_count

def restore(projected):
    require(type(projected) is dict and set(projected) == {"format", "content_sha256", "value"}
        and projected["format"] == FORMAT and type(projected["value"]) is dict,
        "Public origin format")
    _tree(projected)
    value = copy.deepcopy(projected["value"])
    require(value.get("format") != FORMAT, "Nested public origin projection")
    packing = value.get("capacity_tensor_packing")
    if packing is not None:
        require(type(packing) is dict, "Public origin packing")
        encoded = packing.get("bindings")
        require(type(encoded) is dict and set(encoded) == {"format", "row_count", "groups"}
            and encoded["format"] == ROWS_FORMAT and type(encoded["row_count"]) is int
            and 0 <= encoded["row_count"] <= MAX_ROWS and type(encoded["groups"]) is list
            and len(encoded["groups"]) <= encoded["row_count"], "Public origin row budget")
        blocks = []
        total = 0
        binding_bytes = 2 + max(encoded["row_count"] - 1, 0)
        for group in encoded["groups"]:
            require(type(group) is dict and set(group) == {"count", "columns", "data"}
                and type(group["count"]) is int and 1 <= group["count"] <= MAX_ROWS,
                "Public origin group count")
            n, columns = group["count"], group["columns"]
            require(type(columns) is list and 1 <= len(columns) <= MAX_COLUMNS
                and all(type(k) is str for k in columns) and columns == sorted(set(columns))
                and type(group["data"]) is dict and set(group["data"]) == set(columns),
                "Public origin column fields")
            total += n
            require(total <= encoded["row_count"], "Public origin total rows")
            cells = {}
            binding_bytes += n * (2 + len(columns) - 1 + sum(len(canonical(k)) + 1 for k in columns))
            for key in columns:
                runs, size = _encoded_column(group["data"][key], n)
                binding_bytes += size
                require(binding_bytes <= MAX_RAW_BYTES, "Public origin expanded byte budget")
                cells[key] = runs
            blocks.append((n, columns, cells))
        require(total == encoded["row_count"], "Public origin total rows")
        packing["bindings"] = []
        require(len(canonical(value)) - 2 + binding_bytes <= MAX_RAW_BYTES,
            "Public origin expanded byte budget")
        # Allocation follows the exact serialized byte calculation above.
        rows = []
        for n, columns, cells in blocks:
            expanded = {key: [v for v, count in cells[key] for _ in range(count)] for key in columns}
            for i in range(n):
                rows.append({key: copy.deepcopy(expanded[key][i]) for key in columns})
        packing["bindings"] = rows
    _tree(value)
    require(len(canonical(value)) <= MAX_RAW_BYTES and digest(value) == projected["content_sha256"],
        "Public origin content hash")
    require(canonical(compact(value)) == canonical(projected), "Canonical public origin columns")
    return value

def project(original):
    if type(original) is dict and original.get("format") == FORMAT:
        restore(original)
        return copy.deepcopy(original)
    return compact(original)

def origin_digest(projected, *, compact_policy=False):
    return digest(restore(projected) if compact_policy else projected)


def descriptor_references(model_constants, public_constants):
    return dict(model_constants=dict(source="model.constants", content_sha256=digest(model_constants)),
        public_constants=dict(source="public_constants", content_sha256=digest(public_constants)))

def registries(context):
    """Resolve the two fixed namespace references, with exact content hashes."""
    info=context["public_weight_binding"]
    require(info.get("policy")==POLICY and info.get("format")==WEIGHT_FORMAT,
        "Compact descriptor policy")
    expected=descriptor_references(context["model"]["constants"],context["public_constants"])
    require(all(info.get(k)==v for k,v in expected.items()),"Compact descriptor registry reference")
    return dict(model_constants=context["model"]["constants"],public_constants=context["public_constants"])


TYPED_FORMAT="poseidon-prebound-constant-references-v1"

def typed_bindings(registry):
    return [dict(name=name,type="prebound_public_plain",shape=value["shape"],immutable=True)
        for name,value in sorted(registry.items())]

def typed_reference(registry):
    return dict(format=TYPED_FORMAT,source="public_constants",
        type="prebound_public_plain",immutable=True,count=len(registry))

def compact_guidance(guidance, registry):
    """Share optional example lists with the fitter, change only exact ABI duplication."""
    result=dict(guidance)
    typed=guidance.get("typed_request_bindings")
    if type(typed) is dict and "constant_bindings" in typed:
        bindings=typed["constant_bindings"]
        expected=typed_reference(registry)
        if type(bindings) is dict:
            require(bindings==expected,"Compact typed binding reference")
        else:
            require(canonical(bindings)==canonical(typed_bindings(registry)),
                "Compact typed binding identity")
        result["typed_request_bindings"]=dict(typed,constant_bindings=expected)
    return result

def verify_guidance(context):
    guidance=context.get("generation_guidance")
    if guidance is None:return
    registry=registries(context)["public_constants"]
    require(canonical(compact_guidance(guidance,registry))==canonical(guidance),
        "Canonical compact typed bindings")
