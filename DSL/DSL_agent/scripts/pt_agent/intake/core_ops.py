"""The R3 core vocabulary and the separately defined kernel coverage set."""

CORE_OPS = {
    "conv2d": {"inputs": 1, "attrs": {"weight": "weight", "bias": "weight?", "stride": "pair", "padding": "pair", "dilation": "pair", "groups": int}},
    "linear": {"inputs": 1, "attrs": {"weight": "weight", "bias": "weight?"}},
    "matmul": {"inputs": 2, "attrs": {}},
    "add": {"inputs": (1, 2), "attrs": {"scalar": float, "const": "weight"}},
    "sub": {"inputs": (1, 2), "attrs": {"scalar": float, "const": "weight", "reverse": bool}},
    "mul": {"inputs": (1, 2), "attrs": {"scalar": float, "const": "weight"}},
    "batchnorm": {"inputs": 1, "attrs": {"weight": "weight", "bias": "weight", "mean": "weight", "var": "weight", "eps": float}},
    "layernorm": {"inputs": 1, "attrs": {"weight": "weight?", "bias": "weight?", "normalized_shape": "tuple", "eps": float}},
    "act": {"inputs": 1, "attrs": {"fn": str}},
    "softmax": {"inputs": 1, "attrs": {"axis": int}},
    "avgpool": {"inputs": 1, "attrs": {"kernel": "pair", "stride": "pair", "padding": "pair"}},
    "adaptive_avgpool": {"inputs": 1, "attrs": {"output_size": "pair"}},
    "flatten": {"inputs": 1, "attrs": {"start_dim": int, "end_dim": int}},
    "reshape": {"inputs": 1, "attrs": {"shape": "tuple"}},
    "permute": {"inputs": 1, "attrs": {"dims": "tuple"}},
    "concat": {"inputs": "2+", "attrs": {"axis": int}},
    "slice": {"inputs": 1, "attrs": {"axis": int, "start": int, "end": int, "step": int}},
    "split": {"inputs": 1, "attrs": {"axis": int, "sizes": "tuple"}},
    "pad": {"inputs": 1, "attrs": {"pads": "tuple"}},
    "mean": {"inputs": 1, "attrs": {"axes": "tuple", "keepdim": bool}},
    "sum": {"inputs": 1, "attrs": {"axes": "tuple", "keepdim": bool}},
    "identity": {"inputs": 1, "attrs": {}},
}

# User's R3 correction: uncomposed adaptive_avgpool needs composition.
KERNEL_OPS = frozenset({"conv2d", "linear", "add", "sub", "mul", "batchnorm", "act", "avgpool",
                        "flatten", "reshape", "slice", "pad", "mean", "sum", "concat"})
