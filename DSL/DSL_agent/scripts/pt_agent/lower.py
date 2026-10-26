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
from .kernels.layout import (MultiMuxLayout, MultiFlatLayout, layout_blocks, multi_mux_layout,
                             multi_flat_layout, pack_multi, unpack_multi)
from .kernels import multi, cnn, approx


HANDLED_OPS = frozenset({
    "pack", "conv2d", "act", "add", "slice", "pad", "mean", "relayout", "linear", "refresh",
    "avgpool", "maxpool", "concat", "flatten", "batchnorm",
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
        return multi_mux_layout(slots, *shape[1:], op.attrs["gap"])
    if op.attrs["layout"] == "flat" and len(shape) == 2 and shape[0] == 1:
        return multi_flat_layout(slots, shape[1])
    raise KernelError("pack requires mux (1,C,H,W) or flat (1,F)")


def _act_block(b, value, layout, attrs, act_mode):
    if act_mode == 'exact':
        value = (_new_exact_activation(b,value,layout,attrs['fn']) if attrs['method'] in approx.NEW_METHODS
                 else b.apply_exact(value, attrs['fn']))
    elif attrs['method'] in approx.NEW_METHODS:
        value = approx.activation(b,value,attrs)
    elif attrs['method'] == 'cheb':
        value = act_cheb(b, value, attrs['fn'], attrs['deg'], attrs['domain'])
    elif attrs['method'] == 'minimax_comp':
        value = act_minimax(b, value, tuple(attrs['degrees']), attrs['domain'])
    elif attrs['method'] == 'exact' and attrs['fn'] == 'square':
        value = value*value
    else:
        raise KernelError("v1 lowering 不支持 this activation")
    if isinstance(layout, FlatLayout):
        if act_mode == 'exact':
            from .kernels.poly import EXACT
            tail = 0.0 if attrs['method'] in approx.NEW_METHODS else EXACT[attrs['fn']](np.zeros(1))[0]
        elif attrs['method'] in approx.NEW_METHODS:
            tail = approx.activation(None,np.zeros(1),attrs)[0]
        elif attrs['method'] == 'cheb':
            tail = act_cheb(None, np.zeros(1), attrs['fn'], attrs['deg'], attrs['domain'])[0]
        elif attrs['method'] == 'minimax_comp':
            tail = act_minimax(None, np.zeros(1), tuple(attrs['degrees']), attrs['domain'])[0]
        else:
            tail = 0.0
        if tail != 0:
            correction = np.zeros(layout.n)
            correction[layout.size:] = -tail
            value = value+b.const(correction)
    return value


def _new_exact_activation(b,value,layout,fn):
    """Evaluate new mathematical targets only on the layout's valid slots."""
    from .backends.mock import MockExpr
    if isinstance(layout,FlatLayout):
        output = np.zeros_like(value.value)
        output[..., :layout.size] = approx.function(fn,value.value[..., :layout.size])
    elif isinstance(layout,MuxLayout):
        output = np.zeros_like(value.value)
        grid = layout.index_grid()
        output[...,grid] = approx.function(fn,value.value[...,grid])
    else:
        output = approx.function(fn,value.value)
    return MockExpr(b,output)


def _forward_bn_target(op, consumers, outputs):
    uses = consumers[op.outputs[0]]
    if op.outputs[0] in outputs or len(uses) != 1:
        return None
    following = uses[0]
    return following if following.kind == 'linear' or (
        following.kind == 'conv2d' and following.attrs['pad'] == (0, 0)) else None


def lower(program: Program, weights: Mapping[str, np.ndarray], b: Backend, *,
          act_mode: str = "program", observer=None) -> tuple[list, list]:
    if act_mode not in ("program", "exact") or act_mode == "exact" and not isinstance(b, MockBackend):
        raise LoweringError(None, None, "act_mode must be program, or exact on MockBackend")
    env, input_index = {}, 0
    for param in program.params:
        packs = [op for op in program.ops if op.kind == 'pack' and op.inputs == (param.name,)]
        try:
            lay = _pack_layout(packs[0], param.shape, program.header.slots) if len(packs) == 1 else None
        except KernelError as exc:
            raise LoweringError(packs[0].id, packs[0].lineno, str(exc)) from exc
        count = len(layout_blocks(lay)) if lay is not None else 1
        inputs = [b.input(input_index+i) for i in range(count)]
        input_index += count
        env[param.name] = (inputs if count > 1 else inputs[0], RawLayout(param.shape))
    consumers = defaultdict(list)
    for op in program.ops:
        for vid in op.inputs:
            consumers[vid].append(op)
    remaining = Counter(vid for op in program.ops for vid in op.inputs)
    remaining.update(program.outputs)
    skipped, input_bn = set(), {}
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
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)):
                    raise KernelError("conv2d input must have MuxLayout")
                w, bias = ref("w"), ref("b")
                if op.id in input_bn:
                    w, bias = cnn.fold_input_bn(w, bias, *input_bn[op.id], groups=a['groups'])
                if (a["algo"] != "mpcb" or a["dilation"] != (1, 1) or a["groups"] not in (1, layout.C)
                        or a["stride"][0] != a["stride"][1]
                        or a["pad"] != ((w.shape[2]-1)//2,)*2):
                    raise KernelError("conv2d requires mpcb, dilation=1, groups=1, equal strides and same padding")
                scale, shift = None, bias
                if a["bn"] is not None:
                    prefix = a["bn"]
                    scale, shift = fold_bn(weight(prefix+".weight"), weight(prefix+".bias"),
                                           weight(prefix+".running_mean"), weight(prefix+".running_var"),
                                           a["bn_eps"], bias)
                uses = consumers[output]
                if len(uses) == 1 and uses[0].kind == 'batchnorm' and output not in program.outputs:
                    bn = uses[0]
                    prefix = bn.attrs['bn']
                    extra_scale, extra_shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                        weight(prefix+'.running_mean'), weight(prefix+'.running_var'), bn.attrs['eps'])
                    shift = extra_shift if shift is None else extra_scale*shift+extra_shift
                    scale = extra_scale if scale is None else extra_scale*scale
                    output = bn.outputs[0]
                    handled.append(bn); skipped.add(bn.id)
                conv = cnn.depthwise_blocks if a['groups'] > 1 and a['groups'] == layout.C and w.shape[:2] == (layout.C, 1) else multi.conv2d_blocks
                value, layout = conv(b, value, layout, w, stride=a["stride"][0], scale=scale, shift=shift)
            elif op.kind in ('avgpool', 'maxpool'):
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)):
                    raise KernelError('pool input must have MuxLayout')
                if op.kind == 'avgpool':
                    value, layout = cnn.avgpool_blocks(b, value, layout,
                        kernel=a['k'], stride=a['stride'], pad=a['pad'])
                else:
                    stats = cnn.DifferenceStats()
                    def observe(difference, block, valid):
                        data = difference.value[..., block.index_grid()[0]]
                        stats.add(data[..., valid])
                        if act_mode != 'exact':
                            lo, hi = a['domain']
                            max_u = float(np.max(np.abs(stats.value*(2/(hi-lo))-(hi+lo)/(hi-lo))))
                            if max_u > 1:
                                if observer is not None:
                                    observer(op.inputs[0], stats, cnn.DifferenceLayout(), cfg)
                                from .backends.base import BackendError
                                raise BackendError('maxpool comparison input leaves its domain')
                    cfg = (('exact',) if act_mode == 'exact' else
                           approx.configuration(a) if a['method'] in approx.NEW_METHODS else
                           ('minimax_comp', tuple(a['degrees']), *a['domain']) if a['method'] == 'minimax_comp'
                           else ('cheb', a['deg'], *a['domain']))
                    attrs = dict(a, fn='relu')
                    value, layout = cnn.maxpool_blocks(b, value, layout,
                        kernel=a['k'], stride=a['stride'], pad=a['pad'],
                        apply=lambda item: _act_block(b, item, None, attrs, act_mode),
                        observer=observe if observer is not None or isinstance(b, MockBackend) else None)
                    if observer is not None:
                        observer(op.inputs[0], stats, cnn.DifferenceLayout(), cfg)
            elif op.kind == 'concat':
                if a['axis'] != 1:
                    raise KernelError('concat supports axis=1')
                pairs = [env[name] for name in op.inputs]
                value, layout = cnn.concat_blocks([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            elif op.kind == 'flatten':
                if a['start_dim'] != 1 or not isinstance(layout, (MuxLayout, MultiMuxLayout, FlatLayout, MultiFlatLayout)):
                    raise KernelError('flatten requires a packed tensor and start_dim=1')
                if isinstance(layout, (MuxLayout, MultiMuxLayout)) and layout.H == layout.W == 1:
                    value, layout = multi.mean_blocks(b, value, layout)
            elif op.kind == 'batchnorm':
                prefix = a['bn']
                scale, shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                    weight(prefix+'.running_mean'), weight(prefix+'.running_var'), a['eps'])
                following = _forward_bn_target(op, consumers, program.outputs)
                if following is not None:
                    input_bn[following.id] = scale, shift
                else:
                    value, layout = cnn.batchnorm_blocks(b, value, layout, scale, shift)
            elif op.kind == "act":
                cfg = (("exact",) if act_mode == "exact" or a["method"] == "exact"
                       else approx.configuration(a) if a['method'] in approx.NEW_METHODS
                       else ("minimax_comp", tuple(a["degrees"]), *a["domain"])
                       if a["method"] == "minimax_comp" else ("cheb", a["deg"], *(a["domain"] or ())))
                if observer is not None:
                    observer(op.inputs[0], value, layout, cfg)
                if isinstance(layout, multi.MULTI):
                    value, layout = multi.act_blocks(b, value, layout,
                        lambda item, block: _act_block(b, item, block, a, act_mode))
                elif act_mode == "exact":
                    value = (_new_exact_activation(b,value,layout,a['fn']) if a['method'] in approx.NEW_METHODS
                             else b.apply_exact(value,a['fn']))
                elif a['method'] in approx.NEW_METHODS:
                    value = approx.activation(b,value,a)
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
                        tail = 0.0 if a['method'] in approx.NEW_METHODS else EXACT[a["fn"]](np.zeros(1))[0]
                    elif a['method'] in approx.NEW_METHODS:
                        tail = approx.activation(None,np.zeros(1),a)[0]
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
                if isinstance(layout, multi.MULTI) or isinstance(other_layout, multi.MULTI):
                    value, layout = multi.add_blocks(b, value, layout, other, other_layout)
                elif isinstance(layout, MuxLayout) and isinstance(other_layout, MuxLayout):
                    value, layout = add_mux(b, value, layout, other, other_layout)
                elif isinstance(layout, FlatLayout) and layout == other_layout:
                    value = value+other
                else:
                    raise KernelError("add requires compatible mux or identical flat layouts")
            elif op.kind == "slice":
                uses = consumers[output]
                if (not isinstance(layout, (MuxLayout, MultiMuxLayout)) or a["axis"] != 2 or a["start"] != 0
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
                out = multi_mux_layout(layout.n, layout.C+pads[4]+pads[5], layout.H//2, layout.W//2, layout.k*2)
                downsample = (multi.downsample_blocks if isinstance(layout, multi.MULTI) or isinstance(out, multi.MULTI)
                              else downsample_pad_mux)
                value, layout = downsample(b, value, layout, pads[4], pads[5])
                handled.extend((second, pad))
                skipped.update((second.id, pad.id))
                output = pad.outputs[0]
            elif op.kind == "pad":
                raise KernelError("standalone pad is unsupported")
            elif op.kind == "mean":
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)) or a["axis"] != (2, 3) or a["keepdim"]:
                    raise KernelError("mean requires mux, axis=(2,3), keepdim=False")
                mean = multi.mean_blocks if isinstance(layout, multi.MULTI) or layout.C > layout.n//2 else mean_mux
                value, layout = mean(b, value, layout)
            elif op.kind == "relayout":
                if a['layout'] == 'flat' and isinstance(layout, MultiMuxLayout):
                    value, layout = multi.relayout_blocks(b, value, layout)
                elif a["layout"] != "flat" or not isinstance(layout, (FlatLayout, MultiFlatLayout)):
                    raise KernelError("relayout requires an already flat input and flat target")
            elif op.kind == "linear":
                if not isinstance(layout, (FlatLayout, MultiFlatLayout, MuxLayout, MultiMuxLayout)):
                    raise KernelError("linear input must have FlatLayout or a flattened mux input")
                w, bias = ref('w'), ref('b')
                if op.id in input_bn:
                    w, bias = cnn.fold_input_bn(w, bias, *input_bn[op.id])
                uses = consumers[output]
                if len(uses) == 1 and uses[0].kind == 'batchnorm' and output not in program.outputs:
                    bn = uses[0]
                    prefix = bn.attrs['bn']
                    scale, shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                        weight(prefix+'.running_mean'), weight(prefix+'.running_var'), bn.attrs['eps'], bias)
                    w, bias = w*scale[:, None], shift
                    output = bn.outputs[0]
                    handled.append(bn); skipped.add(bn.id)
                if isinstance(layout, (MuxLayout, MultiMuxLayout)):
                    linear = cnn.flatten_linear
                else:
                    linear = multi.linear_blocks if isinstance(layout, multi.MULTI) or w.shape[0] > layout.n//2 else linear_flat
                value, layout = linear(b, value, layout, w, bias, algo=a["algo"], n1=a["n1"])
            elif op.kind == "refresh":
                if program.header.refresh == "agent":
                    value = ([item.bootstrap() for item in value] if isinstance(layout, multi.MULTI)
                             else value.bootstrap())
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
    layouts = list(layouts)
    values = [item for value, lay in zip(values, layouts) for item in multi.values_list(value, lay)]
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
            if isinstance(layout, multi.MULTI):
                blocks = [pack_multi(sample, layout) for sample in array]
                packed.extend(np.stack([sample[i] for sample in blocks]) for i in range(len(layout.blocks)))
            else:
                pack = pack_mux if isinstance(layout, MuxLayout) else pack_flat
                packed.append(np.stack([pack(sample, layout) for sample in array]))
        except KernelError as exc:
            raise LoweringError(op.id, op.lineno, str(exc)) from exc
    return packed


def decode_outputs(values_np: list[np.ndarray], layouts: list) -> list[np.ndarray]:
    if len(values_np) != sum(len(layout_blocks(lay)) for lay in layouts):
        raise LoweringError(None, None, "output and layout counts differ")
    decoded = []
    index = 0
    for layout in layouts:
        count = len(layout_blocks(layout))
        value = values_np[index]
        if isinstance(layout, multi.MULTI):
            decoded.append(unpack_multi(values_np[index:index+count], layout))
        elif isinstance(layout, FlatLayout):
            decoded.append(unpack_flat(value, layout))
        elif isinstance(layout, MuxLayout):
            value = np.asarray(value)
            decoded.append(unpack_mux(value, layout) if value.ndim == 1
                           else np.stack([unpack_mux(sample, layout) for sample in value]))
        else:
            raise LoweringError(None, None, "output has no supported packed layout")
        index += count
    return decoded
