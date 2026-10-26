"""The complete R1 table: attribute schemas, defaults and value constraints."""

from dataclasses import dataclass
import math
from typing import Callable

from ..kernels.poly import MINIMAX_SETS


ATTR_TYPES = ("int", "float", "str", "bool", "weight", "int_or_pair", "int_tuple",
              "float_tuple", "float_pair", "axis")


@dataclass(frozen=True)
class AttrSpec:
    type: str
    required: bool = False
    default: object = None
    choices: tuple | None = None


@dataclass(frozen=True)
class OpSpec:
    name: str
    inputs: int | str
    outputs: str
    attrs: dict[str, AttrSpec]
    validate: Callable[[dict], str | None]
    layout: str


def _pow2(value, minimum=1):
    return type(value) is int and value >= minimum and value & (value - 1) == 0


def _layout(a):
    if not _pow2(a["gap"]):
        return "gap must be a positive power of two"
    if a["layout"] != "mux" and a["gap"] != 1:
        return "gap must be 1 outside mux layouts"
    if a["period"] is not None and not _pow2(a["period"], 2):
        return "period must be None or a power of two >= 2"
    if a["layout"] == "mux" and a["period"] is not None:
        return "mux requires period=None"
    if a["layout"] == "rep":
        if not _pow2(a["copies"], 2):
            return "rep requires copies to be a power of two >= 2"
    elif a["copies"] is not None:
        return "copies applies only to rep"


def _conv(a):
    if min(a["stride"] + a["dilation"]) < 1 or min(a["pad"]) < 0:
        return "stride/dilation must be positive and pad nonnegative"
    if a["groups"] < 1 or not a["bn_eps"] > 0:
        return "groups and bn_eps must be positive"


def _linear(a):
    if a["n1"] is not None and (not _pow2(a["n1"]) or a["algo"] != "bsgs"):
        return "n1 requires bsgs and a positive power of two"


def _pool(a):
    if min(a["k"] + a["stride"]) < 1 or min(a["pad"]) < 0:
        return "k/stride must be positive and pad nonnegative"


def _maxpool(a):
    return _pool(a) or _act(dict(a, fn="relu"))


def _axes(a):
    if len(a["axis"]) != len(set(a["axis"])):
        return "axis entries must be distinct"


def new_approx_error(a):
    fn, method = a["fn"], a["method"]
    degrees, deg, domain = a["degrees"], a["deg"], a["domain"]
    extras = (a.get('eps'),a.get('deg0'),a.get('iters'))
    if method in ('remez','composite','newton'):
        if domain is None or not domain[0] < domain[1]:
            return f'{method} requires domain=(lo,hi) with lo < hi'
        if method == 'remez':
            if deg is None or deg < 1 or degrees is not None or any(v is not None for v in extras):
                return 'remez requires positive deg, domain, and degrees=eps=deg0=iters=None'
        elif method == 'composite':
            if (fn != 'relu' or not degrees or any(d < 1 or d % 2 == 0 for d in degrees)
                    or deg is not None or a.get('eps') is None or not 0 < a['eps'] < 1
                    or a.get('deg0') is not None or a.get('iters') is not None
                    or domain[0] != -domain[1] or domain[1] <= 0):
                return 'composite requires relu, positive odd degrees, 0 < eps < 1 and domain=(-B,B); other relu methods: cheb, remez, minimax_comp'
        elif (fn not in ('inv','rsqrt') or a.get('deg0') is None or a['deg0'] < 1
              or a.get('iters') is None or a['iters'] < 0 or deg is not None or degrees is not None
              or a.get('eps') is not None or domain[0] <= 0):
            return 'newton requires inv/rsqrt, positive deg0, nonnegative iters and 0 < lo < hi; other functions may use remez or cheb'
        if fn in ('inv','rsqrt') and domain[0] <= 0:
            return 'inv/rsqrt requires a positive domain'
        return None
    if any(v is not None for v in extras):
        return 'eps/deg0/iters apply to composite/newton only'
    return None


def _act(a):
    fn,method = a['fn'],a['method']
    degrees,deg,domain = a['degrees'],a['deg'],a['domain']
    if method in ('remez','composite','newton'):
        return None  # U5 mathematical parameter checks belong to typecheck.
    if method == "minimax_comp":
        if (fn != "relu" or degrees is None or tuple(degrees) not in MINIMAX_SETS
                or deg is not None or domain is None or domain[0] != -domain[1] or domain[1] <= 0):
            return (f'minimax_comp requires fn="relu", degrees in {sorted(MINIMAX_SETS)}, '
                    'deg=None and a symmetric domain (-B, B)')
    elif method == "cheb":
        if fn == "square" or deg is None or not 1 <= deg <= 255 or degrees is not None or domain is None:
            return "cheb requires a nonsquare function, deg in [1,255], domain and degrees=None"
    elif fn != "square" or any(v is not None for v in (degrees, deg, domain)):
        return "exact requires square and degrees=deg=domain=None"
    if fn in ("inv", "rsqrt") and (domain is None or domain[0] <= 0):
        return "inv/rsqrt requires a positive domain"


def _poly(a):
    if not 1 <= len(a["coeffs"]) <= 256 or not all(math.isfinite(x) for x in a["coeffs"]):
        return "coeffs must contain 1..256 finite numbers"
    if (a["basis"] == "cheb") != (a["domain"] is not None):
        return "cheb requires a domain; power requires domain=None"


def _softmax(a):
    if not 1 <= a["exp_deg"] <= 255 or not 1 <= a["inv_deg"] <= 255 or a["inv_domain"][0] <= 0:
        return "softmax requires degrees in [1,255] and a positive inv_domain"


def _layernorm(a):
    if not a["eps"] > 0 or not 1 <= a["rsqrt_deg"] <= 255 or a["rsqrt_domain"][0] <= 0:
        return "layernorm requires eps>0, rsqrt_deg in [1,255] and a positive rsqrt_domain"


def _reshape(a):
    shape = a["shape"]
    if shape.count(-1) > 1 or any(x < 1 and x != -1 for x in shape):
        return "shape requires positive dimensions and at most one -1"


def _pad(a):
    p = a["pads"]
    if len(p) < 2 or len(p) > 8 or len(p) % 2 or min(p) < 0:
        return "pads requires 2..8 nonnegative entries with even length"


def _none(a):
    return None


def _spec(name, inputs, layout, attrs=None, validate=_none, outputs="one"):
    return OpSpec(name, inputs, outputs, attrs or {}, validate, layout)


_LAYOUT = dict(layout=AttrSpec("str", True, choices=("flat", "mux", "rep")),
               gap=AttrSpec("int", default=1), period=AttrSpec("int"), copies=AttrSpec("int"))
_REDUCE = dict(axis=AttrSpec("axis", True), keepdim=AttrSpec("bool", default=False))
OPS = {
    "pack": _spec("pack", 1, "pack", dict(_LAYOUT), _layout),
    "relayout": _spec("relayout", 1, "relayout", dict(_LAYOUT), _layout),
    "conv2d": _spec("conv2d", 1, "same", dict(
        w=AttrSpec("weight", True), b=AttrSpec("weight"), bn=AttrSpec("str"),
        bn_eps=AttrSpec("float", default=1e-5), stride=AttrSpec("int_or_pair", default=1),
        pad=AttrSpec("int_or_pair", default=0), dilation=AttrSpec("int_or_pair", default=1),
        groups=AttrSpec("int", default=1), algo=AttrSpec("str", default="mpcb", choices=("mpcb", "tap"))), _conv),
    "linear": _spec("linear", 1, "same", dict(
        w=AttrSpec("weight", True), b=AttrSpec("weight"),
        algo=AttrSpec("str", default="bsgs", choices=("bsgs", "diag", "rowsum")), n1=AttrSpec("int")), _linear),
    "matmul": _spec("matmul", 2, "binary", dict(
        algo=AttrSpec("str", default="diag", choices=("diag", "rowsum")))),
    "avgpool": _spec("avgpool", 1, "same", dict(k=AttrSpec("int_or_pair", True),
        stride=AttrSpec("int_or_pair"), pad=AttrSpec("int_or_pair", default=0)), _pool),
    "maxpool": _spec("maxpool", 1, "same", dict(k=AttrSpec("int_or_pair", True),
        stride=AttrSpec("int_or_pair"), pad=AttrSpec("int_or_pair", default=0),
        method=AttrSpec("str", True, choices=("cheb", "minimax_comp", "remez", "composite", "newton")),
        degrees=AttrSpec("int_tuple"), deg=AttrSpec("int"), domain=AttrSpec("float_pair"),
        eps=AttrSpec("float"),deg0=AttrSpec("int"),iters=AttrSpec("int")), _maxpool),
    "mean": _spec("mean", 1, "same", dict(_REDUCE), _axes),
    "sum": _spec("sum", 1, "same", dict(_REDUCE), _axes),
    "add": _spec("add", 2, "binary"),
    "sub": _spec("sub", 2, "binary"),
    "mul": _spec("mul", 2, "binary"),
    "scale": _spec("scale", 1, "same", dict(c=AttrSpec("float", True)),
                   lambda a: None if math.isfinite(a["c"]) else "c must be finite"),
    "bias": _spec("bias", 1, "same", dict(b=AttrSpec("weight", True), axis=AttrSpec("int", default=1))),
    "batchnorm": _spec("batchnorm", 1, "same", dict(bn=AttrSpec("str", True), eps=AttrSpec("float", default=1e-5)),
                       lambda a: None if a["eps"] > 0 else "eps must be positive"),
    "act": _spec("act", 1, "same", dict(
        fn=AttrSpec("str", True, choices=("relu", "silu", "gelu", "sigmoid", "tanh", "exp", "inv", "rsqrt", "hardswish", "square")),
        method=AttrSpec("str", True, choices=("minimax_comp", "cheb", "exact", "remez", "composite", "newton")),
        degrees=AttrSpec("int_tuple"), deg=AttrSpec("int"), domain=AttrSpec("float_pair"),
        eps=AttrSpec("float"),deg0=AttrSpec("int"),iters=AttrSpec("int")), _act),
    "poly": _spec("poly", 1, "same", dict(coeffs=AttrSpec("float_tuple", True),
        basis=AttrSpec("str", default="power", choices=("power", "cheb")), domain=AttrSpec("float_pair")), _poly),
    "softmax": _spec("softmax", 1, "same", dict(axis=AttrSpec("int", default=-1),
        exp_deg=AttrSpec("int", True), exp_domain=AttrSpec("float_pair", True),
        inv_deg=AttrSpec("int", True), inv_domain=AttrSpec("float_pair", True)), _softmax),
    "layernorm": _spec("layernorm", 1, "same", dict(w=AttrSpec("weight"), b=AttrSpec("weight"),
        eps=AttrSpec("float", default=1e-5), rsqrt_deg=AttrSpec("int", True),
        rsqrt_domain=AttrSpec("float_pair", True)), _layernorm),
    "flatten": _spec("flatten", 1, "same", dict(start_dim=AttrSpec("int", default=1))),
    "reshape": _spec("reshape", 1, "same", dict(shape=AttrSpec("int_tuple", True)), _reshape),
    "permute": _spec("permute", 1, "same", dict(dims=AttrSpec("int_tuple", True))),
    "concat": _spec("concat", "2+", "binary", dict(axis=AttrSpec("int", True))),
    "slice": _spec("slice", 1, "same", dict(axis=AttrSpec("int", True), start=AttrSpec("int", default=0),
        end=AttrSpec("int"), step=AttrSpec("int", default=1)),
        lambda a: None if a["step"] >= 1 else "step must be positive"),
    "split": _spec("split", 1, "same", dict(axis=AttrSpec("int", True), sizes=AttrSpec("int_tuple", True)),
        lambda a: None if min(a["sizes"]) >= 1 else "sizes must be positive", outputs="split"),
    "pad": _spec("pad", 1, "same", dict(pads=AttrSpec("int_tuple", True)), _pad),
    "refresh": _spec("refresh", 1, "same"),
}
RESERVED_NAMES = frozenset({"W", "fhe", "region", "Cipher", "Plain"}) | frozenset(OPS)


def describe_markdown() -> str:
    lines = ["| op | inputs | attributes |", "|---|---|---|"]
    for name, spec in OPS.items():
        attributes = []
        for key, attr in spec.attrs.items():
            if attr.required:
                item = f"{key}: {attr.type} (required)"
            else:
                item = f"{key}: {attr.type} = {attr.default!r}"
            if attr.choices is not None:
                item += " {" + ", ".join(map(str, attr.choices)) + "}"
            attributes.append(item)
        lines.append(f"| {name} | {spec.inputs} | " + "; ".join(attributes) + " |")
    return "\n".join(lines) + "\n"
