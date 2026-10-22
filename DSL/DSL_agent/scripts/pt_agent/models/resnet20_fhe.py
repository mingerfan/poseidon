"""R5 graph assembly using only the specified mux, activation and flat kernels."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..backends import MockBackend
from ..kernels import (EXACT, FlatLayout, KernelError, MuxLayout, act_cheb, add_mux, conv2d_mux,
                       downsample_pad_mux, linear_flat, mean_mux, mux_layout, pack_mux, unpack_flat)
from ..kernels.conv import fold_bn

if TYPE_CHECKING:
    from ..intake import Graph


def build(b, graph: Graph, act_cfg: dict, *, n: int = 16384, x=None, observer=None):
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise KernelError("v1 requires one input and one linear output")
    shape = graph.tensors[graph.inputs[0]].shape
    if len(shape) != 4 or shape[0] != 1:
        raise KernelError("v1 requires a static batch-one NCHW input")
    layout = mux_layout(n, *shape[1:], 1)
    if x is None:
        x = b.input(0)
    values = {graph.inputs[0]: (x, layout)}
    remaining = Counter(name for node in graph.nodes for name in node.inputs)
    remaining.update(graph.outputs)
    skipped = set()
    linear_outputs = set()

    def weight(attrs, key):
        return graph.weights[attrs[key]] if attrs.get(key) is not None else None

    for node in graph.nodes:
        if node.id in skipped:
            continue
        handled = [node]
        attrs = node.attrs
        output_name = node.outputs[0]
        try:
            value, layout = values[node.inputs[0]]
            if node.op == "conv2d":
                w = weight(attrs, "weight")
                if (tuple(attrs["padding"]) != ((w.shape[2]-1)//2,)*2 or tuple(attrs["dilation"]) != (1, 1)
                        or attrs["groups"] != 1 or tuple(attrs["stride"]) not in ((1, 1), (2, 2))):
                    raise KernelError("v1 convolution requires same padding, dilation=1, groups=1 and stride 1 or 2")
                consumers = graph.consumers(node.outputs[0])
                bias = weight(attrs, "bias")
                scale, shift = None, bias
                if len(consumers) == 1 and consumers[0].op == "batchnorm":
                    bn = consumers[0]
                    scale, shift = fold_bn(weight(bn.attrs, "weight"), weight(bn.attrs, "bias"),
                                           weight(bn.attrs, "mean"), weight(bn.attrs, "var"), bn.attrs["eps"], bias)
                    handled.append(bn)
                    skipped.add(bn.id)
                    output_name = bn.outputs[0]
                value, layout = conv2d_mux(b, value, layout, w, stride=attrs["stride"][0], scale=scale, shift=shift)
            elif node.op == "batchnorm":
                raise KernelError("v1 不支持 standalone batchnorm")
            elif node.op == "act":
                fn = attrs["fn"]
                if fn not in EXACT:
                    raise KernelError(f"v1 不支持 activation {fn}")
                cfg = act_cfg.get(node.inputs[0])
                if not isinstance(cfg, (tuple, list)) or not cfg:
                    raise KernelError("missing activation configuration")
                if observer is not None:
                    observer(node.inputs[0], value, layout, cfg)
                if tuple(cfg) == ("exact",):
                    if not isinstance(b, MockBackend):
                        raise KernelError("exact activation is available only in MockBackend")
                    value = b.apply_exact(value, fn)
                elif len(cfg) == 4 and cfg[0] == "cheb":
                    value = act_cheb(b, value, fn, cfg[1], (cfg[2], cfg[3]))
                else:
                    raise KernelError("invalid activation configuration")
            elif node.op == "add":
                if len(node.inputs) != 2 or any(graph.tensors[name].kind == "weight" for name in node.inputs):
                    raise KernelError("v1 add requires two activation inputs")
                other, other_layout = values[node.inputs[1]]
                value, layout = add_mux(b, value, layout, other, other_layout)
            elif node.op == "slice":
                consumers = graph.consumers(node.outputs[0])
                if not isinstance(layout, MuxLayout) or attrs != dict(axis=2, start=0, end=layout.H, step=2) or len(consumers) != 1:
                    raise KernelError("v1 不支持 slice outside the downsample-pad chain")
                second = consumers[0]
                following = graph.consumers(second.outputs[0])
                if (second.op != "slice" or second.attrs != dict(axis=3, start=0, end=layout.W, step=2)
                        or len(following) != 1 or following[0].op != "pad"):
                    raise KernelError("v1 不支持 slice outside the downsample-pad chain")
                pad = following[0]
                pads = tuple(pad.attrs["pads"])
                if len(pads) != 6 or pads[:4] != (0, 0, 0, 0):
                    raise KernelError("v1 downsample pad requires channel-only padding")
                value, layout = downsample_pad_mux(b, value, layout, pads[4], pads[5])
                handled.extend((second, pad))
                skipped.update((second.id, pad.id))
                output_name = pad.outputs[0]
            elif node.op == "pad":
                raise KernelError("v1 不支持 standalone pad")
            elif node.op == "mean":
                if tuple(attrs["axes"]) != (2, 3) or attrs["keepdim"]:
                    raise KernelError("v1 mean requires axes=(2,3), keepdim=False")
                value, layout = mean_mux(b, value, layout)
            elif node.op == "linear":
                if not isinstance(layout, FlatLayout):
                    raise KernelError("linear input must use FlatLayout")
                value, layout = linear_flat(b, value, layout, weight(attrs, "weight"), weight(attrs, "bias"), algo="bsgs")
                linear_outputs.add(output_name)
            else:
                raise KernelError("v1 不支持 this core op")
        except KernelError as exc:
            raise KernelError(f"{node.id} ({node.op}): {exc}") from exc
        values[output_name] = (value, layout)
        # Release consumed activation arrays while retaining residual branches.
        for completed in handled:
            for name in completed.inputs:
                remaining[name] -= 1
                if remaining[name] == 0:
                    values.pop(name, None)
    output = graph.outputs[0]
    if output not in linear_outputs or not isinstance(values[output][1], FlatLayout):
        raise KernelError("graph output must be a flat value produced by linear")
    return values[output]


def pack_input(img: np.ndarray, n=16384) -> np.ndarray:
    return pack_mux(img, mux_layout(n, 3, 32, 32, 1))


def decode_logits(vec, lay: FlatLayout) -> np.ndarray:
    return unpack_flat(vec, lay)
