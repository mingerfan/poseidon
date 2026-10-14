"""Versioned static output trees for the v16 MHA source draft; no SDK imports."""
import hashlib
import json

SOURCE_FORMAT = "poseidon-trusted-pytorch-v16"
OLD_FORMAT = "poseidon-trusted-pytorch-v15"
FORMAT = "poseidon-static-output-tree-v16-draft-v1"

def require(value, message):
    if not value:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def checked_names(names):
    require(type(names) is list and 1 <= len(names) <= 4 and
            all(type(n) is str and n.isidentifier() and len(n) <= 128 for n in names) and
            len(set(names)) == len(names), "output_names")
    return names

def flatten(value, is_tensor, names, *, source_format):
    checked_names(names)
    require(source_format in (SOURCE_FORMAT, OLD_FORMAT), "output_source_format")
    leaves = []
    none_count = 0
    def visit(item, depth=0):
        nonlocal none_count
        require(depth <= 8, "output_depth")
        if is_tensor(item):
            require(len(leaves) < len(names), "output_leaf_count")
            name = names[len(leaves)]
            leaves.append(item)
            return dict(kind="tensor", name=name)
        if item is None:
            require(source_format == SOURCE_FORMAT, "static_none_requires_v16")
            none_count += 1
            return dict(kind="none")
        if type(item) in (tuple, list):
            require(1 <= len(item) <= 4, "output_sequence")
            return dict(kind="tuple" if type(item) is tuple else "list",
                        items=[visit(x, depth + 1) for x in item])
        if type(item) is dict:
            require(1 <= len(item) <= 4 and
                    all(type(k) is str and 0 < len(k) <= 128 for k in item), "output_dictionary")
            return dict(kind="dict", items=[dict(key=k, value=visit(x, depth + 1))
                                            for k, x in item.items()])
        raise ValueError("output_static_or_tensor")
    tree = visit(value)
    require(len(leaves) == len(names), "output_leaf_count")
    record = dict(format=FORMAT, source_format=source_format, tensor_names=list(names),
                  tree=tree, static_none_count=none_count)
    record["binding"] = digest(record)
    validate(record)
    return leaves, record

def validate(record):
    require(type(record) is dict and set(record) == {
        "format", "source_format", "tensor_names", "tree", "static_none_count", "binding"
    } and record["format"] == FORMAT and record["source_format"] in (SOURCE_FORMAT, OLD_FORMAT),
            "output_record")
    require(record["binding"] == digest({k: v for k, v in record.items() if k != "binding"}),
            "output_binding")
    checked_names(record["tensor_names"])
    found = []
    none_count = 0
    def visit(item, depth=0):
        nonlocal none_count
        require(type(item) is dict and depth <= 8, "output_node")
        kind = item.get("kind")
        if kind == "tensor":
            require(set(item) == {"kind", "name"} and type(item["name"]) is str, "output_tensor")
            found.append(item["name"])
            return
        if kind == "none":
            require(set(item) == {"kind"} and record["source_format"] == SOURCE_FORMAT,
                    "static_none_requires_v16")
            none_count += 1
            return
        require(kind in ("tuple", "list", "dict") and set(item) == {"kind", "items"} and
                type(item["items"]) is list and 1 <= len(item["items"]) <= 4, "output_container")
        keys = set()
        for child in item["items"]:
            if kind == "dict":
                require(type(child) is dict and set(child) == {"key", "value"} and
                        type(child["key"]) is str and 0 < len(child["key"]) <= 128 and
                        child["key"] not in keys, "output_dictionary_key")
                keys.add(child["key"])
                child = child["value"]
            visit(child, depth + 1)
    visit(record["tree"])
    require(found == record["tensor_names"] and type(record["static_none_count"]) is int and
            record["static_none_count"] == none_count, "output_signature")
    return record

def restore(values, record):
    validate(record)
    require(type(values) is dict and list(values) == record["tensor_names"], "output_values")
    def visit(item):
        if item["kind"] == "tensor":
            return values[item["name"]]
        if item["kind"] == "none":
            return None
        if item["kind"] == "dict":
            return {c["key"]: visit(c["value"]) for c in item["items"]}
        result = [visit(c) for c in item["items"]]
        return tuple(result) if item["kind"] == "tuple" else result
    return visit(record["tree"])
