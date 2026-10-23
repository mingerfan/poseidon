"""Lower parsed PT-DSL into the existing mux and flat kernels."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .backends import MockBackend
from .backends.base import Backend
from .dsl.ir import Program
from .kernels import (FlatLayout, KernelError, MuxLayout, act_cheb, add_mux, conv2d_mux,
                      downsample_pad_mux, linear_flat, mean_mux, mux_layout, pack_flat,
                      pack_mux, unpack_flat, unpack_mux)
from .kernels.conv import fold_bn
from .kernels.poly import act_minimax


HANDLED_OPS = frozenset({
    "pack", "conv2d", "act", "add", "slice", "pad", "mean", "relayout", "linear", "refresh",
})


class LoweringError(Exception):
    def __init__(self, op_id: int | None, lineno: int | None, msg: str):
        self.op_id, self.lineno, self.msg = op_id, lineno, msg
        super().__init__(f"op {op_id} line {lineno}: {msg}")


@dataclass(frozen=True)
class RawLayout:
    shape: tuple[int, ...]


def _pack_layout(op, shape, slots):
    if op.attrs["layout"] == "mux" and len(shape) == 4 and shape[0] == 1:
        return mux_layout(slots, *shape[1:], op.attrs["gap"])
    if op.attrs["layout"] == "flat" and len(shape) == 2 and shape[0] == 1:
        return FlatLayout(slots, shape[1])
    raise KernelError("pack requires mux (1,C,H,W) or flat (1,F)")


def lower(program: Program, weights: Mapping[str, np.ndarray], b: Backend, *,
          act_mode: str = "program", observer=None) -> tuple[list, list]:
    if act_mode not in ("program", "exact") or act_mode == "exact" and not isinstance(b, MockBackend):
        raise LoweringError(None, None, "act_mode must be program, or exact on MockBackend")
    env = {param.name: (b.input(i), RawLayout(param.shape)) for i, param in enumerate(program.params)}
    consumers = defaultdict(list)
    for op in program.ops:
        for vid in op.inputs:
            consumers[vid].append(op)
    remaining = Counter(vid for op in program.ops for vid in op.inputs)
    remaining.update(program.outputs)
    skipped = set()
    for op in program.ops:
        if op.id in skipped:
            continue
        b.op_context = op.id
        handled = [op]
        output = op.outputs[0]
        a = op.attrs

        def weight(name):
            try:
                return weights[name]
            except KeyError as exc:
                raise LoweringError(op.id, op.lineno, f"missing weight {name}") from exc

        def ref(key):
            return None if a.get(key) is None else weight(a[key].name)

        try:
            value, layout = env[op.inputs[0]]
            if op.kind == "pack":
                if not isinstance(layout, RawLayout):
                    raise KernelError("pack input must have RawLayout")
                layout = _pack_layout(op, layout.shape, program.header.slots)
            elif op.kind == "conv2d":
                if not isinstance(layout, MuxLayout):
                    raise KernelError("conv2d input must have MuxLayout")
                w, bias = ref("w"), ref("b")
                if (a["algo"] != "mpcb" or a["dilation"] != (1, 1) or a["groups"] != 1
                        or a["stride"][0] != a["stride"][1]
                        or a["pad"] != ((w.shape[2]-1)//2,)*2):
                    raise KernelError("conv2d requires mpcb, dilation=1, groups=1, equal strides and same padding")
                scale, shift = None, bias
                if a["bn"] is not None:
                    prefix = a["bn"]
                    scale, shift = fold_bn(weight(prefix+".weight"), weight(prefix+".bias"),
                                           weight(prefix+".running_mean"), weight(prefix+".running_var"),
                                           a["bn_eps"], bias)
                value, layout = conv2d_mux(b, value, layout, w, stride=a["stride"][0], scale=scale, shift=shift)
            elif op.kind == "act":
                cfg = (("exact",) if act_mode == "exact" or a["method"] == "exact"
                       else ("minimax_comp", tuple(a["degrees"]), *a["domain"])
                       if a["method"] == "minimax_comp" else ("cheb", a["deg"], *(a["domain"] or ())))
                if observer is not None:
                    observer(op.inputs[0], value, layout, cfg)
                if act_mode == "exact":
                    value = b.apply_exact(value, a["fn"])
                elif a["method"] == "cheb":
                    value = act_cheb(b, value, a["fn"], a["deg"], a["domain"])
                elif a["method"] == "exact" and a["fn"] == "square":
                    value = value*value
                elif a["method"] == "minimax_comp":
                    value = act_minimax(b, value, tuple(a["degrees"]), a["domain"])
                else:
                    raise KernelError("v1 lowering 不支持 this activation")
                if isinstance(layout, FlatLayout):
                    tail = 0.0
                    if act_mode == "exact":
                        from .kernels.poly import EXACT

                        tail = EXACT[a["fn"]](np.zeros(1))[0]
                    elif a["method"] == "cheb":
                        tail = act_cheb(None, np.zeros(1), a["fn"], a["deg"], a["domain"])[0]
                    elif a["method"] == "minimax_comp":
                        tail = act_minimax(None, np.zeros(1), tuple(a["degrees"]), a["domain"])[0]
                    if tail != 0:
                        correction = np.zeros(layout.n)
                        correction[layout.size:] = -tail
                        value = value + b.const(correction)
            elif op.kind == "add":
                other, other_layout = env[op.inputs[1]]
                if isinstance(layout, MuxLayout) and isinstance(other_layout, MuxLayout):
                    value, layout = add_mux(b, value, layout, other, other_layout)
                elif isinstance(layout, FlatLayout) and layout == other_layout:
                    value = value+other
                else:
                    raise KernelError("add requires compatible mux or identical flat layouts")
            elif op.kind == "slice":
                uses = consumers[output]
                if (not isinstance(layout, MuxLayout) or a["axis"] != 2 or a["start"] != 0
                        or a["end"] not in (None, layout.H) or a["step"] != 2 or len(uses) != 1):
                    raise KernelError("slice requires the downsample slice/slice/pad chain")
                second = uses[0]
                following = consumers[second.outputs[0]]
                if (second.kind != "slice" or second.attrs != dict(axis=3, start=0, end=layout.W, step=2)
                        and second.attrs != dict(axis=3, start=0, end=None, step=2)
                        or len(following) != 1 or following[0].kind != "pad"):
                    raise KernelError("slice requires the downsample slice/slice/pad chain")
                pad = following[0]
                pads = pad.attrs["pads"]
                if len(pads) != 6 or pads[:4] != (0, 0, 0, 0):
                    raise KernelError("downsample pad requires channel-only padding")
                value, layout = downsample_pad_mux(b, value, layout, pads[4], pads[5])
                handled.extend((second, pad))
                skipped.update((second.id, pad.id))
                output = pad.outputs[0]
            elif op.kind == "pad":
                raise KernelError("standalone pad is unsupported")
            elif op.kind == "mean":
                if not isinstance(layout, MuxLayout) or a["axis"] != (2, 3) or a["keepdim"]:
                    raise KernelError("mean requires mux, axis=(2,3), keepdim=False")
                value, layout = mean_mux(b, value, layout)
            elif op.kind == "relayout":
                if a["layout"] != "flat" or not isinstance(layout, FlatLayout):
                    raise KernelError("relayout requires an already flat input and flat target")
            elif op.kind == "linear":
                if not isinstance(layout, FlatLayout):
                    raise KernelError("linear input must have FlatLayout")
                value, layout = linear_flat(b, value, layout, ref("w"), ref("b"), algo=a["algo"], n1=a["n1"])
            elif op.kind == "refresh":
                if program.header.refresh == "agent":
                    value = value.bootstrap()
            else:
                raise KernelError(f"v1 lowering 不支持 {op.kind}")
        except KernelError as exc:
            raise LoweringError(op.id, op.lineno, str(exc)) from exc
        env[output] = value, layout
        for completed in handled:
            for vid in completed.inputs:
                remaining[vid] -= 1
                if remaining[vid] == 0:
                    env.pop(vid, None)
    values, layouts = zip(*(env[vid] for vid in program.outputs))
    values, layouts = list(values), list(layouts)
    b.output(values)
    return values, layouts


def pack_inputs(program: Program, arrays: list[np.ndarray]) -> list[np.ndarray]:
    if len(arrays) != len(program.params):
        raise LoweringError(None, None, "input count differs from program parameters")
    packed = []
    for param, array in zip(program.params, arrays):
        ops = [op for op in program.ops if op.kind == "pack" and op.inputs == (param.name,)]
        if len(ops) != 1:
            raise LoweringError(None, param.lineno, "each parameter requires one pack operation")
        op = ops[0]
        try:
            array = np.asarray(array, dtype=np.float64)
            if array.ndim != len(param.shape) or array.shape[1:] != param.shape[1:]:
                raise KernelError("input array shape differs from parameter")
            layout = _pack_layout(op, param.shape, program.header.slots)
            pack = pack_mux if isinstance(layout, MuxLayout) else pack_flat
            packed.append(np.stack([pack(sample, layout) for sample in array]))
        except KernelError as exc:
            raise LoweringError(op.id, op.lineno, str(exc)) from exc
    return packed


def decode_outputs(values_np: list[np.ndarray], layouts: list) -> list[np.ndarray]:
    if len(values_np) != len(layouts):
        raise LoweringError(None, None, "output and layout counts differ")
    decoded = []
    for value, layout in zip(values_np, layouts):
        if isinstance(layout, FlatLayout):
            decoded.append(unpack_flat(value, layout))
        elif isinstance(layout, MuxLayout):
            value = np.asarray(value)
            decoded.append(unpack_mux(value, layout) if value.ndim == 1
                           else np.stack([unpack_mux(sample, layout) for sample in value]))
        else:
            raise LoweringError(None, None, "output has no supported packed layout")
    return decoded
