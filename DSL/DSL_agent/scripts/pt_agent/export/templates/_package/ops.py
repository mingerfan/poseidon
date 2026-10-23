"""PT-DSL operations over (value, layout) pairs and copied arithmetic kernels."""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from .core.layout import FlatLayout, KernelError, MuxLayout, mux_layout
from .core.conv import add_mux, conv2d_mux, downsample_pad_mux, fold_bn
from .core.pool import mean_mux
from .core.linear import linear_flat
from .core.cheb import cheb_eval


Tensor = namedtuple("Tensor", "value layout")


@dataclass(frozen=True)
class RawLayout:
    shape: tuple[int, ...]
    slots: int


def pack(b, x, *, layout, gap):
    value, lin = x
    if not isinstance(lin, RawLayout):
        raise KernelError("pack input must have RawLayout")
    shape = lin.shape
    if layout == "mux" and len(shape) == 4 and shape[0] == 1:
        out = mux_layout(b.slots, *shape[1:], gap)
    elif layout == "flat" and len(shape) == 2 and shape[0] == 1:
        out = FlatLayout(b.slots, shape[1])
    else:
        raise KernelError("pack requires mux (1,C,H,W) or flat (1,F)")
    return Tensor(value, out)


def conv2d(bk, x, W, *, w, b, bn, bn_eps, stride, pad, dilation, groups, algo):
    value, layout = x
    if not isinstance(layout, MuxLayout):
        raise KernelError("conv2d input must have MuxLayout")
    weight, bias = W[w], None if b is None else W[b]
    if (algo != "mpcb" or dilation != (1, 1) or groups != 1 or stride[0] != stride[1]
            or pad != ((weight.shape[2]-1)//2,)*2):
        raise KernelError("conv2d requires mpcb, dilation=1, groups=1, equal strides and same padding")
    scale, shift = None, bias
    if bn is not None:
        scale, shift = fold_bn(W[bn+".weight"], W[bn+".bias"], W[bn+".running_mean"],
                               W[bn+".running_var"], bn_eps, bias)
    return Tensor(*conv2d_mux(bk, value, layout, weight, stride=stride[0], scale=scale, shift=shift))


def act(b, x, a):
    value, layout = x
    if a["method"] == "cheb":
        lo, hi = a["domain"]
        scale = 2/(hi-lo)
        c = -(hi+lo)/(hi-lo)
        u = value*scale+c
        value = cheb_eval(b, u, a["coefficients"])
    elif a["method"] == "exact" and a["fn"] == "square":
        value = value*value
    elif a["method"] == "minimax_comp":
        lo, hi = a["domain"]
        if not (hi > 0 and lo == -hi):
            raise KernelError("minimax_comp requires a symmetric domain (-B, B)")
        s = value*(1.0/hi)
        for coefficients in a["coefficients"]:
            s = cheb_eval(b, s, coefficients)
        value = value*(s+0.5)
    else:
        raise KernelError("v1 lowering 不支持 this activation")
    if isinstance(layout, FlatLayout):
        tail = 0.0
        if a["method"] == "cheb":
            tail = cheb_eval(None, np.zeros(1)*scale+c, a["coefficients"])[0]
        elif a["method"] == "minimax_comp":
            zero = np.zeros(1)
            s = zero*(1.0/hi)
            for coefficients in a["coefficients"]:
                s = cheb_eval(None, s, coefficients)
            tail = (zero*(s+0.5))[0]
        if tail != 0:
            correction = np.zeros(layout.n)
            correction[layout.size:] = -tail
            value = value + b.const(correction)
    return Tensor(value, layout)


def add(b, x, y):
    value, layout = x
    other, other_layout = y
    if isinstance(layout, MuxLayout) and isinstance(other_layout, MuxLayout):
        return Tensor(*add_mux(b, value, layout, other, other_layout))
    if isinstance(layout, FlatLayout) and layout == other_layout:
        return Tensor(value+other, layout)
    raise KernelError("add requires compatible mux or identical flat layouts")


def downsample(b, x, *, pads):
    return Tensor(*downsample_pad_mux(b, x.value, x.layout, pads[0], pads[1]))


def mean(b, x, *, axis, keepdim):
    if not isinstance(x.layout, MuxLayout) or axis != (2, 3) or keepdim:
        raise KernelError("mean requires mux, axis=(2,3), keepdim=False")
    return Tensor(*mean_mux(b, x.value, x.layout))


def relayout(b, x, *, layout):
    if layout != "flat" or not isinstance(x.layout, FlatLayout):
        raise KernelError("relayout requires an already flat input and flat target")
    return x


def linear(bk, x, W, *, w, b, algo, n1):
    if not isinstance(x.layout, FlatLayout):
        raise KernelError("linear input must have FlatLayout")
    return Tensor(*linear_flat(bk, x.value, x.layout, W[w], None if b is None else W[b], algo=algo, n1=n1))


def refresh(b, x, *, enabled):
    return Tensor(x.value.bootstrap(), x.layout) if enabled else x
