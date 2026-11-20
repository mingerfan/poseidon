"""Versioned public type/shape descriptions for the unchanged v4 lowering.

Enforcement remains in the existing rule and core validators. This catalog is
declarative metadata and never installs callbacks or changes graph semantics.
"""
from types import MappingProxyType

FORMAT = "poseidon-operator-type-shape-contracts-v1"

# Each specification is immutable: operand roles, shape constraints, result shape,
# and a complete set of public static attribute descriptions.
_SPECS = {
    "padding": (("cipher_tensor",), "Rank-r tensor; r pad pairs. Each nonnegative pad is profile bounded. Reflect pads are smaller than their input axis; replicate and reflect require value zero.", "Each axis grows by its left and right padding.", (("pads", "array[r] of two nonnegative integers"), ("mode", "enum: constant, reflect, replicate"), ("value", "finite real with absolute value <=1024"))),
    "broadcast": (("cipher_tensor",), "Trailing axes must be equal or singleton; prepended axes behave as singleton. Requested rank and every axis obey the selected profile.", "Exactly the public target shape.", (("shape", "nonempty array of positive profile-bounded integers"),)),
    "public_gather": (("cipher_tensor",), "Public integer indices select one valid axis; every index is in [0, axis_size). Index count is positive and profile bounded.", "Input shape with selected axis replaced by index count.", (("axis", "integer axis in [-rank, rank)"), ("indices", "nonempty array of nonnegative integer indices"))),
    "adaptive_avg_pool": (("cipher_tensor",), "Rank3 [N,C,L] or rank4 [N,C,H,W]. Each public output spatial size is positive and no larger than its input size.", "[N,C,*output_size]; windows use floor/ceil adaptive boundaries.", (("output_size", "array[rank-2] of positive downsampling integers"),)),
    "dot": (("cipher_tensor", "cipher_tensor"), "Two encrypted rank1 vectors with identical length. Public operands belong to the separately versioned operator-graph-v2 matrix interface.", "One-element tensor containing the sum of pairwise products.", ()),
    "matmul": (("cipher_tensor", "cipher_tensor"), "Both operands have rank2 or3; left last dimension equals right penultimate dimension; batch dimensions broadcast. General-rank and public-operand matrices use operator-graph-v2.", "Broadcast batch shape followed by [left_rows,right_columns].", ()),
    "variance": (("cipher_tensor",), "Public distinct valid reduction axes; the original core reduction rules define axis and keepdims admission.", "Input shape reduced along axes, retaining singleton axes when requested.", (("axes", "array of distinct integer axes in [-rank,rank)"), ("keepdims", "boolean"))),
    "layer_norm": (("cipher_tensor", "public_tensor", "public_tensor"), "normalized_shape is a nonempty trailing input suffix. Gain and bias have exactly that shape. Epsilon is finite and in (0,1].", "Same input shape; normalize the declared trailing axes.", (("normalized_shape", "nonempty array of positive trailing dimensions"), ("eps", "finite real in (0,1]"), ("approximation", "rsqrt Newton profile"))),
    "rms_norm": (("cipher_tensor", "public_tensor"), "Encrypted rank>=1 tensor; public gain is a vector matching the final input axis. Epsilon is finite and in (0,1].", "Same input shape; RMS reduction is on the last axis.", (("eps", "finite real in (0,1]"), ("approximation", "rsqrt Newton profile"))),
    "group_norm": (("cipher_tensor", "public_tensor", "public_tensor"), "Rank3 [N,C,L] or rank4 [N,C,H,W]; gain/bias are length-C vectors; positive integer groups divides C; epsilon is in (0,1].", "Same input shape; per-sample groups include channels and spatial axes.", (("groups", "positive integer divisor of C"), ("eps", "finite real in (0,1]"), ("approximation", "rsqrt Newton profile"))),
    "instance_norm": (("cipher_tensor", "public_tensor", "public_tensor"), "Rank3 [N,C,L] or rank4 [N,C,H,W]; public gain/bias are length-C vectors; groups equals C; epsilon is in (0,1].", "Same input shape; per-sample per-channel spatial normalization.", (("eps", "finite real in (0,1]"), ("approximation", "rsqrt Newton profile"))),
    "global_avg_pool": (("cipher_tensor",), "Rank3 [N,C,L] or rank4 [N,C,H,W]; average every spatial axis.", "[N,C,1] or [N,C,1,1].", ()),
    "dropout": (("cipher_tensor",), "Inference-only identity; the public training attribute must be the boolean false.", "Same input shape and values.", (("training", "boolean false only"),)),
    "rope": (("cipher_tensor",), "Rank1 vector with even positive length; split-half pairing. Position is integer0..65536; theta is finite and greater than1.", "Same vector length after public sine/cosine rotation.", (("position", "integer in [0,65536]"), ("theta", "finite real >1"))),
    "leaky_relu": (("cipher_tensor",), "Shape-preserving activation; finite public negative slope has absolute value <=1024. ReLU approximation is explicitly declared.", "Same input shape.", (("negative_slope", "finite real with absolute value <=1024"), ("approximation", "explicit ReLU polynomial profile"))),
    "prelu": (("cipher_tensor", "public_tensor"), "Public slope has shape[1] or length-C shape[C] for input rank>1, where C is input axis1. ReLU approximation is explicitly declared.", "Same input shape with singleton or channel-wise slope broadcast.", (("approximation", "explicit ReLU polynomial profile"),)),
    "softmax": (("cipher_tensor",), "Public valid normalization axis; explicitly declared exp and reciprocal domains cover their respective consumed values.", "Same input shape; exp divided by the axis-wise sum using declared approximations.", (("axis", "integer axis in [-rank,rank)"), ("exp", "exp Taylor profile"), ("reciprocal", "positive-domain Goldschmidt profile"))),
    "logsoftmax": (("cipher_tensor",), "Public valid normalization axis; explicitly declared exp and positive-domain log profiles cover their consumed values.", "Same input shape; input minus the declared log of the exp sum.", (("axis", "integer axis in [-rank,rank)"), ("exp", "exp Taylor profile"), ("log", "positive-domain explicit polynomial profile"))),
}
for _name in ("maximum", "minimum"):
    _SPECS[_name] = (("cipher_tensor", "cipher_or_public_tensor"), "Two operands obey the existing core trailing-axis broadcast rules. Pairwise differences use an explicit absolute-value polynomial.", "Broadcast shape of the two operands.", (("approximation", "explicit absolute-value polynomial profile"),))
for _name in ("max", "min"):
    _SPECS[_name] = (("cipher_tensor",), "Nonempty public distinct reduction axes; compare C-order leaves using a bounded balanced tree and explicit absolute-value polynomial.", "Input shape reduced along axes, with optional retained singleton axes.", (("axes", "nonempty array of distinct valid integer axes"), ("keepdims", "boolean"), ("approximation", "explicit absolute-value polynomial profile")))
for _name, _rank in (("max_pool1d", 1), ("max_pool2d", 2)):
    _SPECS[_name] = (("cipher_tensor",), "Rank%d [N,C,*spatial]; positive integer kernel, stride and dilation; nonnegative padding with 2*padding<=kernel; every output window contains an in-bounds element." % (_rank + 2), "Spatial axis i: floor((size+2*padding-dilation*(kernel-1)-1)/stride)+1; [N,C] retained.", (("kernel", "array[%d] of positive profile-bounded integers" % _rank), ("stride", "array[%d] of positive profile-bounded integers" % _rank), ("padding", "array[%d] of nonnegative profile-bounded integers" % _rank), ("dilation", "array[%d] of positive profile-bounded integers" % _rank), ("approximation", "explicit absolute-value polynomial profile")))
for _name, _method in (("exp", "exp Taylor profile"), ("reciprocal", "positive-domain Goldschmidt profile"), ("rsqrt", "positive-domain rsqrt Newton profile"), ("sqrt", "positive-domain rsqrt Newton profile")):
    _SPECS[_name] = (("cipher_tensor",), "Shape-preserving %s; explicit finite interval and bounded work are required. Reciprocal, sqrt and rsqrt domains are strictly positive." % _name, "Same input shape.", (("approximation", _method),))
for _name in ("relu", "gelu", "silu", "sigmoid", "tanh", "log"):
    _SPECS[_name] = (("cipher_tensor",), "Shape-preserving %s using an explicit power/Chebyshev polynomial and finite interval.%s" % (_name, " Log interval is strictly positive." if _name == "log" else ""), "Same input shape.", (("approximation", "explicit polynomial profile"),))
SPECS = MappingProxyType(_SPECS)
del _SPECS, _name

def catalog():
    """Return independent JSON data linked to the actual immutable registry."""
    from model_decomposition import FORMAT as GRAPH_FORMAT, REGISTRY, REGISTRY_VERSION
    if set(SPECS) != set(REGISTRY):
        raise ValueError("Operator type/shape catalog coverage differs from registry")
    operators = {}
    for name, spec in sorted(SPECS.items()):
        roles, constraints, output_shape, attrs = spec
        op = REGISTRY[name]
        if {key for key, _ in attrs} != set(op.attributes):
            raise ValueError("Operator attribute catalog differs from registry: " + name)
        operators[name] = dict(
            summary=name + ": " + constraints, graph_format=GRAPH_FORMAT, registry_version=REGISTRY_VERSION,
            value_type="real_tensor", input_dtype="float64", input_roles=list(roles),
            shape_constraints=constraints, output_shape=output_shape,
            public_attributes={key: dict(type_and_constraints=value) for key, value in attrs},
            reference=op.reference, decomposition_rule=op.rule,
            approximation=op.approximation, required_capabilities=list(op.capabilities),
            resource_bounds="Selected profile governs rank, axis lengths, tensor sizes, public constants and expanded graph size.",
        )
    return dict(format=FORMAT, registry_version=REGISTRY_VERSION, graph_format=GRAPH_FORMAT,
                operators=operators,
                approximation_profiles=dict(
                    polynomial="method=polynomial; basis=power|chebyshev; finite domain within[-1024,1024]; 1..128 finite coefficients with abs<=1024",
                    exp="method=taylor; finite domain within[-1024,1024]; integer degree1..20",
                    reciprocal="method=goldschmidt; strictly positive finite domain within(0,1024]; integer iterations1..8",
                    rsqrt="method=newton; strictly positive finite domain within(0,1024]; integer iterations1..8",
                ),
                domain_policy="Declared consumed-value domains are checked by the existing reference/preflight; values are not silently clipped.")
