"""Versioned 16-family benchmark; labels are metadata, not model construction IDs.

Original eight families remain exact schema-1 descriptors. Eight added families
are full schema-2/3 user graphs with explicit weights. The translator receives
only each descriptor; family labels never choose its lowering or reference.
"""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).parent
BASE_FAMILIES = ("affine", "polynomial", "linear", "mlp2", "mlp3", "fanout", "residual", "flatten_linear")
NEW_FAMILIES = ("conv1d", "conv2d", "avg_pool1d", "avg_pool2d", "conv_poly_pool",
                "dual_affine", "dual_bilinear", "dual_linear")
FAMILIES = BASE_FAMILIES + NEW_FAMILIES
VERSION = "restricted-model-suite-v1"
FAMILY_DEFINITIONS = {
    "conv1d": "One-dimensional channel-first windowed cross-correlation, then flatten",
    "conv2d": "Two-dimensional spatial or multichannel cross-correlation, then flatten",
    "avg_pool1d": "One-dimensional averaging with explicit window and border divisor",
    "avg_pool2d": "Two-dimensional/channel-preserving averaging, then flatten",
    "conv_poly_pool": "Convolution, explicit square activation, average pooling",
    "dual_affine": "Two separately encrypted inputs with independently weighted affine fusion",
    "dual_bilinear": "Product of two separately encrypted inputs followed by a weighted reduction",
    "dual_linear": "Difference of independently encrypted inputs, user Linear and square",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def template(name):
    return json.loads((ROOT / f"cases/{name}.json").read_text())


def new_descriptor(family, index):
    rng = random.Random(20260907 + 100*NEW_FAMILIES.index(family) + index)
    choices = (-.7, -.3, -.125, .125, .2, .5, .75)
    def weights(value):
        return [weights(v) for v in value] if type(value) is list else rng.choice(choices)
    names = {
        "conv1d": ("conv1d-valid", "conv1d-stride-pad"),
        "conv2d": ("conv2d-valid", "conv2d-channels"),
        "avg_pool1d": ("avg1d-pad-count", "avg1d-pad-exclude"),
        "avg_pool2d": ("avg2d-valid", "avg2d-wide"),
        "conv_poly_pool": ("conv-square-pool",),
        "dual_linear": ("custom-dual-linear",),
    }
    if family in names:
        data = template(names[family][index % len(names[family])])
        data["constants"] = {name: weights(value) for name, value in data["constants"].items()}
        if family == "avg_pool1d" and index >= 2:
            node = data["nodes"][0]
            kernel, stride, padding = ((2, 1, 0), (2, 2, 0), (4, 4, 0), (1, 1, 0))[index-2]
            node.update(kernel=[kernel], stride=[stride], padding=[padding], count_include_pad=bool(index % 2))
        if family == "avg_pool2d" and index >= 2:
            node = data["nodes"][0]
            if index == 2:
                data["input_shape"] = [1, 2, 2]
                node.update(kernel=[1, 1], stride=[1, 1], padding=[0, 0])
            elif index in (3, 4):
                data["input_shape"] = [1, 1, 4]
                node.update(kernel=[1, 3], stride=[1, 2], padding=[0, 1], count_include_pad=index == 3)
            else:
                data["input_shape"] = [2, 1, 2]
                node.update(kernel=[1, 2], stride=[1, 2], padding=[0, 0])
    else:
        data = dict(schema=3, id="pending", inputs=[{"name": "left", "shape": [4]}, {"name": "right", "shape": [4]}],
                    constants={}, nodes=[], output="out")
        if family == "dual_affine":
            data["constants"] = {name: [rng.choice(choices) for _ in range(4)] for name in ("wa", "wb", "bias")}
            data["nodes"] = [dict(id="a", op="multiply", inputs=["left", "wa"]),
                dict(id="b", op="multiply", inputs=["right", "wb"]),
                dict(id="sum", op="add", inputs=["a", "b"]), dict(id="out", op="add", inputs=["sum", "bias"])]
        else:
            width = 1 + index % 4
            data["constants"] = {"weight": [[rng.choice(choices) for _ in range(4)] for _ in range(width)],
                                 "bias": [rng.choice(choices)/4 for _ in range(width)]}
            data["nodes"] = [dict(id="products", op="multiply", inputs=["left", "right"]),
                dict(id="out", op="linear", inputs=["products"], weight="weight", bias="bias")]
    data["id"] = f"extended-{family}-{index}"
    return data


def entries():
    result = [dict(family=f, configuration=i, descriptor=dict(schema=1, id=f"{f}-{i}", family=f, configuration=i))
              for f in BASE_FAMILIES for i in range(6)]
    result += [dict(family=f, configuration=i, descriptor=new_descriptor(f, i)) for f in NEW_FAMILIES for i in range(6)]
    for item in result:
        item["descriptor_sha256"] = digest(item["descriptor"])
    return result


def manifest():
    items = entries()
    return dict(version=VERSION, families=list(FAMILIES), new_family_definitions=FAMILY_DEFINITIONS,
                entries=items, descriptor_set_sha256=digest([x["descriptor"] for x in items]),
                all_encrypted_validated=False, live_agent_validated=False, poseidon_gpu_validated=False)
