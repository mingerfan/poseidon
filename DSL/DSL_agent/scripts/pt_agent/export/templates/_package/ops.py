"""PT-DSL operations over (value, layout) pairs and copied arithmetic kernels."""

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from .core.layout import FlatLayout, KernelError, MuxLayout, mux_layout
from .core.conv import add_mux, conv2d_mux, downsample_pad_mux, fold_bn
from .core.pool import mean_mux
from .core.linear import linear_flat
from .core.cheb import cheb_eval
from .core.layout import MultiMuxLayout, MultiFlatLayout, multi_mux_layout, multi_flat_layout, TokenLayout, ScoreLayout
from .core import multi, cnn


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
    if layout=='token' and len(shape)==3 and shape[0]==1:
        out=TokenLayout(b.slots,*shape[1:])
    elif layout=='score' and len(shape)==4 and shape[0]==1:
        out=ScoreLayout(b.slots,shape[2],shape[1],shape[3])
    elif layout == "mux" and len(shape) == 4 and shape[0] == 1:
        out = multi_mux_layout(b.slots, *shape[1:], gap)
    elif layout == "flat" and len(shape) == 2 and shape[0] == 1:
        out = multi_flat_layout(b.slots, shape[1])
    else:
        raise KernelError("pack requires mux (1,C,H,W) or flat (1,F)")
    if isinstance(out,(TokenLayout,ScoreLayout)) and not isinstance(value,list):value=[value]
    return Tensor(value, out)


def token(b,W,A,inputs,*,input_names,kind,attrs,output):
    from types import SimpleNamespace
    from .core.token_dispatch import dispatch
    attrs=dict(attrs)
    for key,value in list(attrs.items()):
        if isinstance(value,dict) and ('weight' in value or 'value' in value):
            attrs[key]=SimpleNamespace(name=value.get('weight',value.get('value')))
    if kind=='act':attrs.update(A[output])
    for role in ('rsqrt','exp','inv','step_inv','final_inv'):
        entry=A.get(output+'::'+role)
        if entry is not None:
            attrs.update({role+'_'+key:value for key,value in entry.items() if key!='fn'})
    op=SimpleNamespace(kind=kind,attrs=attrs,inputs=input_names,outputs=(output,))
    return Tensor(*dispatch(op,inputs,W,b))


def conv2d(bk, x, W, *, w, b, bn, bn_eps, stride, pad, dilation, groups, algo, post_bn=None, post_eps=1e-5, pre_bn=None, pre_eps=1e-5):
    value, layout = x
    if not isinstance(layout, (MuxLayout, MultiMuxLayout)):
        raise KernelError("conv2d input must have MuxLayout")
    weight, bias = W[w], None if b is None else W[b]
    if pre_bn is not None:
        pre_scale, pre_shift = fold_bn(W[pre_bn+'.weight'], W[pre_bn+'.bias'],
            W[pre_bn+'.running_mean'], W[pre_bn+'.running_var'], pre_eps)
        weight, bias = cnn.fold_input_bn(weight, bias, pre_scale, pre_shift, groups=groups)
    if (algo != "mpcb" or dilation != (1, 1) or groups not in (1, layout.C) or stride[0] != stride[1]
            or pad != ((weight.shape[2]-1)//2,)*2):
        raise KernelError("conv2d requires mpcb, dilation=1, groups=1, equal strides and same padding")
    scale, shift = None, bias
    if bn is not None:
        scale, shift = fold_bn(W[bn+".weight"], W[bn+".bias"], W[bn+".running_mean"],
                               W[bn+".running_var"], bn_eps, bias)
    if post_bn is not None:
        more_scale, more_shift = fold_bn(W[post_bn+'.weight'], W[post_bn+'.bias'],
            W[post_bn+'.running_mean'], W[post_bn+'.running_var'], post_eps)
        shift = more_shift if shift is None else more_scale*shift+more_shift
        scale = more_scale if scale is None else more_scale*scale
    conv = cnn.depthwise_blocks if groups > 1 and groups == layout.C and weight.shape[:2] == (layout.C, 1) else multi.conv2d_blocks
    return Tensor(*conv(bk, value, layout, weight, stride=stride[0], scale=scale, shift=shift))


def act(b, x, a):
    value, layout = x
    if isinstance(layout, multi.MULTI):
        return Tensor(*multi.act_blocks(b, value, layout, lambda item, block: act(b, Tensor(item, block), a).value))
    if a["method"] == "cheb":
        lo, hi = a["domain"]
        scale = 2/(hi-lo)
        c = -(hi+lo)/(hi-lo)
        u = value*scale+c
        value = cheb_eval(b, u, a["coefficients"])
    elif a['method'] in ('remez','composite','newton'):
        from .core.approx import evaluate
        value=evaluate(b,value,a['fn'],a['method'],a['domain'],a['coefficients'],iters=a.get('iters',0))
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
        elif a['method'] in ('remez','composite','newton'):
            tail=evaluate(None,np.zeros(1),a['fn'],a['method'],a['domain'],a['coefficients'],
                          iters=a.get('iters',0))[0]
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
    if isinstance(layout, multi.MULTI) or isinstance(other_layout, multi.MULTI):
        return Tensor(*multi.add_blocks(b, value, layout, other, other_layout))
    if isinstance(layout, MuxLayout) and isinstance(other_layout, MuxLayout):
        return Tensor(*add_mux(b, value, layout, other, other_layout))
    if isinstance(layout, FlatLayout) and layout == other_layout:
        return Tensor(value+other, layout)
    raise KernelError("add requires compatible mux or identical flat layouts")


def downsample(b, x, *, pads):
    out = multi_mux_layout(x.layout.n, x.layout.C+sum(pads), x.layout.H//2, x.layout.W//2, x.layout.k*2)
    if isinstance(x.layout, multi.MULTI) or isinstance(out, multi.MULTI):
        return Tensor(*multi.downsample_blocks(b, x.value, x.layout, pads[0], pads[1]))
    return Tensor(*downsample_pad_mux(b, x.value, x.layout, pads[0], pads[1]))


def mean(b, x, *, axis, keepdim):
    if not isinstance(x.layout, (MuxLayout, MultiMuxLayout)) or axis != (2, 3) or keepdim:
        raise KernelError("mean requires mux, axis=(2,3), keepdim=False")
    mean = multi.mean_blocks if isinstance(x.layout, multi.MULTI) or x.layout.C > x.layout.n//2 else mean_mux
    return Tensor(*mean(b, x.value, x.layout))


def relayout(b, x, *, layout):
    if layout == 'flat' and isinstance(x.layout, MultiMuxLayout):
        return Tensor(*multi.relayout_blocks(b, x.value, x.layout))
    if layout != "flat" or not isinstance(x.layout, (FlatLayout, MultiFlatLayout)):
        raise KernelError("relayout requires an already flat input and flat target")
    return x


def linear(bk, x, W, *, w, b, algo, n1, post_bn=None, post_eps=1e-5, pre_bn=None, pre_eps=1e-5):
    if not isinstance(x.layout, (FlatLayout, MultiFlatLayout, MuxLayout, MultiMuxLayout)):
        raise KernelError("linear input must have FlatLayout or a flattened mux input")
    weight, bias = W[w], None if b is None else W[b]
    if pre_bn is not None:
        pre_scale, pre_shift = fold_bn(W[pre_bn+'.weight'], W[pre_bn+'.bias'],
            W[pre_bn+'.running_mean'], W[pre_bn+'.running_var'], pre_eps)
        weight, bias = cnn.fold_input_bn(weight, bias, pre_scale, pre_shift)
    if post_bn is not None:
        scale, bias = fold_bn(W[post_bn+'.weight'], W[post_bn+'.bias'],
            W[post_bn+'.running_mean'], W[post_bn+'.running_var'], post_eps, bias)
        weight = weight*scale[:, None]
    Ip=1<<(weight.shape[1]-1).bit_length()
    if isinstance(x.layout,FlatLayout) and 2*Ip<=bk.slots and weight.shape[0]/Ip>=4:
        from .core.transformer import tall_linear
        return Tensor(*tall_linear(bk,x.value,x.layout,weight,bias))
    if isinstance(x.layout, (MuxLayout, MultiMuxLayout)):
        linear = cnn.flatten_linear
    else:
        linear = multi.linear_blocks if isinstance(x.layout, multi.MULTI) or weight.shape[0] > x.layout.n//2 else linear_flat
    return Tensor(*linear(bk, x.value, x.layout, weight, bias, algo=algo, n1=n1))


def avgpool(b, x, *, k, stride, pad):
    return Tensor(*cnn.avgpool_blocks(b, x.value, x.layout, kernel=k, stride=stride, pad=pad))


def maxpool(b, x, a, *, k, stride, pad):
    return Tensor(*cnn.maxpool_blocks(b, x.value, x.layout, kernel=k, stride=stride, pad=pad,
        apply=lambda value: act(b, Tensor(value, None), a).value))


def concat(b, *inputs, axis):
    if axis != 1:
        raise KernelError('concat supports axis=1')
    return Tensor(*cnn.concat_blocks([x.value for x in inputs], [x.layout for x in inputs]))


def flatten(b, x, *, start_dim):
    if start_dim != 1 or not isinstance(x.layout, (MuxLayout, MultiMuxLayout, FlatLayout, MultiFlatLayout)):
        raise KernelError('flatten requires a packed tensor and start_dim=1')
    if isinstance(x.layout, (MuxLayout, MultiMuxLayout)) and x.layout.H == x.layout.W == 1:
        return Tensor(*multi.mean_blocks(b, x.value, x.layout))
    return x


def batchnorm(b, x, W, *, bn, eps):
    scale, shift = fold_bn(W[bn+'.weight'], W[bn+'.bias'], W[bn+'.running_mean'], W[bn+'.running_var'], eps)
    return Tensor(*cnn.batchnorm_blocks(b, x.value, x.layout, scale, shift))


def refresh(b, x, *, enabled):
    if enabled and isinstance(x.layout, multi.MULTI):
        return Tensor([item.bootstrap() for item in x.value], x.layout)
    return Tensor(x.value.bootstrap(), x.layout) if enabled else x
