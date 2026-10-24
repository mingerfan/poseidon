"""Float64 tensor validation and process-local, arithmetic-preserving caches.

The tensor interpreter follows the supported lowering operations.  Prefix cache
entries hold an activation's input and observations of its ancestors; a cache
hit cuts that part of the computation DAG without changing the remaining order.
"""

from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import weakref

import numpy as np
import torch
import torch.nn.functional as F

from .backends import BackendError
from .dsl.ir import WeightRef
from .dsl.printer import print_program
from .kernels.conv import fold_bn
from .kernels import cnn
from .kernels.layout import FlatLayout, KernelError, MuxLayout, mux_layout
from .kernels.layout import (MultiMuxLayout, MultiFlatLayout, layout_blocks,
                             multi_mux_layout, multi_flat_layout)
from .kernels.poly import cheb_eval, cheb_fit, minimax_coeffs
from .lower import LoweringError, RawLayout, _pack_layout, _forward_bn_target


DEFAULT_PREFIX_BYTES = 3 * 1024**3 // 2
_identities = {}
_structures = {}
_results = {}
_estimates = {}
_prefix = OrderedDict()
_prefix_bytes = 0
_prefix_limit = DEFAULT_PREFIX_BYTES
_counts = {}


def _count(group, event):
    events = _counts.setdefault(group, {})
    events[event] = events.get(event, 0) + 1


def identity_token(value):
    """Give a live object an identity that remains safe after Python id reuse."""
    index = id(value)
    found = _identities.get(index)
    if found is not None:
        owner, token, weak = found
        if (owner() if weak else owner) is value:
            return token
    token = object()
    try:
        def expired(ref):
            old = _identities.get(index)
            if old is not None and old[0] is ref:
                _identities.pop(index, None)
        owner = weakref.ref(value, expired)
        weak = True
    except TypeError:
        owner, weak = value, False
    _identities[index] = (owner, token, weak)
    return token


def _normalized(value):
    if isinstance(value, WeightRef):
        return ("weight", value.name)
    if isinstance(value, dict):
        return tuple((key, _normalized(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    return value


def _structure(kind, attrs, inputs):
    # Interning these recursive nodes avoids repeatedly traversing residual DAGs.
    signature = kind, _normalized(attrs), tuple(inputs)
    if signature not in _structures:
        _structures[signature] = object()
    return _structures[signature]


def result_cache_get(key):
    if key in _results:
        _count("result", "hits")
        return True, deepcopy(_results[key])
    _count("result", "misses")
    return False, None


def result_cache_put(key, value):
    _results[key] = deepcopy(value)


def cached_estimate(program, weights, profile, estimator, *, cache=True):
    if not cache:
        return estimator(program, weights, profile=profile)
    from .cost import PROFILES
    configuration = PROFILES[profile] if isinstance(profile, str) else profile
    # Hotspot locations are consumed by the tuner's ordering even when the
    # normalized arithmetic text is unchanged (for example after macro expansion).
    locations = tuple((op.lineno, op.call_lines) for op in program.ops)
    key = (print_program(program), identity_token(weights), _normalized(configuration),
           identity_token(estimator), locations)
    if key in _estimates:
        _count("estimate", "hits")
        return deepcopy(_estimates[key])
    _count("estimate", "misses")
    result = estimator(program, weights, profile=profile)
    _estimates[key] = deepcopy(result)
    return result


@lru_cache(maxsize=2048)
def _fit(fn, deg, lo, hi):
    result = cheb_fit(fn, deg, lo, hi)
    result.flags.writeable = False
    return result


def set_prefix_cache_limit(nbytes):
    global _prefix_limit, _prefix_bytes
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError("prefix cache byte limit must be a nonnegative integer")
    _prefix_limit = nbytes
    while _prefix_bytes > nbytes:
        _, entry = _prefix.popitem(last=False)
        _prefix_bytes -= entry.value.nbytes
        _count("prefix", "evictions")


def clear_caches():
    """Clear values and counters, retaining the configured prefix byte limit."""
    global _prefix_bytes
    _identities.clear()
    _structures.clear()
    _results.clear()
    _estimates.clear()
    _prefix.clear()
    _counts.clear()
    _fit.cache_clear()
    _prefix_bytes = 0


def cache_info():
    result = {group: {event: _counts.get(group, {}).get(event, 0)
                      for event in ("hits", "misses")}
              for group in ("prefix", "result", "estimate")}
    result["prefix"].update({event: _counts.get("prefix", {}).get(event, 0)
                              for event in ("evictions", "oversize")})
    result["prefix"].update(entries=len(_prefix), bytes=_prefix_bytes, max_bytes=_prefix_limit)
    result["result"]["entries"] = len(_results)
    result["estimate"]["entries"] = len(_estimates)
    info = _fit.cache_info()
    result["cheb"] = dict(hits=info.hits, misses=info.misses, entries=info.currsize)
    return result


@dataclass
class _Entry:
    value: np.ndarray
    observations: dict


def _prefix_get(key):
    if key not in _prefix:
        _count("prefix", "misses")
        return None
    _count("prefix", "hits")
    _prefix.move_to_end(key)
    return _prefix[key]


def _prefix_put(key, entry):
    global _prefix_bytes
    if entry.value.nbytes > _prefix_limit:
        _count("prefix", "oversize")
        return
    if key in _prefix:
        _prefix_bytes -= _prefix.pop(key).value.nbytes
    while _prefix and _prefix_bytes + entry.value.nbytes > _prefix_limit:
        _, old = _prefix.popitem(last=False)
        _prefix_bytes -= old.value.nbytes
        _count("prefix", "evictions")
    # Keep caller-owned arrays' flags intact (pack may pass the input through).
    entry.value = entry.value.view()
    entry.value.flags.writeable = False
    _prefix[key] = entry
    _prefix_bytes += entry.value.nbytes


def _cfg(op):
    attrs = op.attrs
    if attrs["method"] == "exact":
        return ("exact",)
    if attrs["method"] == "minimax_comp":
        return ("minimax_comp", tuple(attrs["degrees"]), *attrs["domain"])
    return ("cheb", attrs["deg"], *(attrs["domain"] or ()))


def _conv_layout(layout, weight, stride, scale=None, shift=None):
    if weight.ndim != 4:
        raise KernelError("weight must have shape (Co, Ci, fh, fw)")
    co, ci, fh, fw = weight.shape
    if ci != layout.C or fh != fw or fh % 2 != 1:
        raise KernelError("convolution requires matching input channels and an odd square kernel")
    if type(stride) is not int or stride not in (1, 2) or layout.H % stride or layout.W % stride:
        raise KernelError("stride must be 1 or 2 and divide H and W")
    output = multi_mux_layout(layout.n, co, layout.H//stride, layout.W//stride, layout.k*stride)
    if scale is not None and scale.shape != (co,):
        raise KernelError("scale must have length Co")
    if shift is not None and shift.shape != (co,):
        raise KernelError("shift must have length Co")
    return output


def _metadata(program, weights):
    """Check lowering properties and describe its fused execution DAG."""
    keys = {p.name: _structure("input", (i, p.kind, p.shape, program.header.slots), ())
            for i, p in enumerate(program.params)}
    for op in program.ops:
        key = _structure(op.kind, op.attrs, [keys[name] for name in op.inputs])
        for output in op.outputs:
            keys[output] = key
    consumers = defaultdict(list)
    for op in program.ops:
        for name in op.inputs:
            consumers[name].append(op)
    layouts = {p.name: RawLayout(p.shape) for p in program.params}
    nodes, producer, ancestors = [], {}, {}
    skipped, input_bn = set(), {}
    for op in program.ops:
        if op.id in skipped:
            continue
        attrs, output = op.attrs, op.outputs[0]
        value_layout = layouts[op.inputs[0]]
        layout = value_layout
        extra = {}

        def weight(name):
            try:
                return np.asarray(weights[name], dtype=np.float64)
            except KeyError as exc:
                raise LoweringError(op.id, op.lineno, f"missing weight {name}") from exc

        def ref(name):
            return None if attrs.get(name) is None else weight(attrs[name].name)

        try:
            if op.kind == "pack":
                if not isinstance(layout, RawLayout):
                    raise KernelError("pack input must have RawLayout")
                layout = _pack_layout(op, layout.shape, program.header.slots)
            elif op.kind == "conv2d":
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)):
                    raise KernelError("conv2d input must have MuxLayout")
                w, bias = ref("w"), ref("b")
                if op.id in input_bn:
                    w, bias = cnn.fold_input_bn(w, bias, *input_bn[op.id], groups=attrs['groups'])
                if (attrs["algo"] != "mpcb" or attrs["dilation"] != (1, 1) or attrs["groups"] not in (1, layout.C)
                        or attrs["stride"][0] != attrs["stride"][1]
                        or attrs["pad"] != ((w.shape[2]-1)//2,)*2):
                    raise KernelError("conv2d requires mpcb, dilation=1, groups=1, equal strides and same padding")
                scale, shift = None, bias
                if attrs["bn"] is not None:
                    prefix = attrs["bn"]
                    scale, shift = fold_bn(weight(prefix+".weight"), weight(prefix+".bias"),
                                           weight(prefix+".running_mean"), weight(prefix+".running_var"),
                                           attrs["bn_eps"], bias)
                uses = consumers[output]
                if len(uses) == 1 and uses[0].kind == 'batchnorm' and output not in program.outputs:
                    bn = uses[0]; prefix = bn.attrs['bn']
                    extra_scale, extra_shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                        weight(prefix+'.running_mean'), weight(prefix+'.running_var'), bn.attrs['eps'])
                    shift = extra_shift if shift is None else extra_scale*shift+extra_shift
                    scale = extra_scale if scale is None else extra_scale*scale
                    output = bn.outputs[0]; skipped.add(bn.id)
                if attrs['groups'] > 1 and attrs['groups'] == layout.C and w.shape[:2] == (layout.C, 1):
                    _, stride, _, height, width = cnn.pool_shape(layout, w.shape[2], attrs['stride'], attrs['pad'])
                    layout = cnn.projected_layout(layout, height, width, stride)
                else:
                    layout = _conv_layout(layout, w, attrs["stride"][0], scale, shift)
                extra = dict(weight=w, scale=scale, shift=shift)
            elif op.kind in ('avgpool', 'maxpool'):
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)):
                    raise KernelError('pool input must have MuxLayout')
                _, stride, _, height, width = cnn.pool_shape(layout, attrs['k'], attrs['stride'], attrs['pad'])
                layout = cnn.projected_layout(layout, height, width, stride)
            elif op.kind == 'concat':
                if attrs['axis'] != 1:
                    raise KernelError('concat supports axis=1')
                source_layouts = [layouts[name] for name in op.inputs]
                _, layout = cnn.concat_blocks([None if isinstance(lay, MuxLayout) else [None]*len(lay.blocks)
                                                for lay in source_layouts], source_layouts)
            elif op.kind == 'flatten':
                if attrs['start_dim'] != 1 or not isinstance(layout, (MuxLayout, MultiMuxLayout, FlatLayout, MultiFlatLayout)):
                    raise KernelError('flatten requires a packed tensor and start_dim=1')
                if isinstance(layout, (MuxLayout, MultiMuxLayout)) and layout.H == layout.W == 1:
                    layout = multi_flat_layout(layout.n, layout.C)
            elif op.kind == 'batchnorm':
                prefix = attrs['bn']
                scale, shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                    weight(prefix+'.running_mean'), weight(prefix+'.running_var'), attrs['eps'])
                following = _forward_bn_target(op, consumers, program.outputs)
                if following is not None:
                    input_bn[following.id] = scale, shift
                extra = dict(scale=scale, shift=shift, fold_forward=following is not None)
            elif op.kind == "act":
                if attrs["method"] not in ("cheb", "minimax_comp") and not (
                        attrs["method"] == "exact" and attrs["fn"] == "square"):
                    raise KernelError("v1 lowering 不支持 this activation")
            elif op.kind == "add":
                other = layouts[op.inputs[1]]
                if isinstance(layout, (MultiMuxLayout, MultiFlatLayout)) or isinstance(other, (MultiMuxLayout, MultiFlatLayout)):
                    if layout != other:
                        raise KernelError("residual layouts must match")
                elif isinstance(layout, MuxLayout) and isinstance(other, MuxLayout):
                    if layout != other:
                        raise KernelError("residual layouts must match")
                elif not (isinstance(layout, FlatLayout) and layout == other):
                    raise KernelError("add requires compatible mux or identical flat layouts")
            elif op.kind == "slice":
                uses = consumers[output]
                if (not isinstance(layout, (MuxLayout, MultiMuxLayout)) or attrs["axis"] != 2 or attrs["start"] != 0
                        or attrs["end"] not in (None, layout.H) or attrs["step"] != 2 or len(uses) != 1):
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
                co = layout.C+pads[4]+pads[5]
                layout = _conv_layout(layout, np.empty((co, layout.C, 1, 1)), 2)
                extra = dict(pads=pads)
                skipped.update((second.id, pad.id))
                output = pad.outputs[0]
            elif op.kind == "pad":
                raise KernelError("standalone pad is unsupported")
            elif op.kind == "mean":
                if not isinstance(layout, (MuxLayout, MultiMuxLayout)) or attrs["axis"] != (2, 3) or attrs["keepdim"]:
                    raise KernelError("mean requires mux, axis=(2,3), keepdim=False")
                if layout.H & (layout.H-1) or layout.W & (layout.W-1):
                    raise KernelError("mean_mux requires power-of-two H and W")
                layout = multi_flat_layout(layout.n, layout.C)
            elif op.kind == "relayout":
                if attrs['layout'] == 'flat' and isinstance(layout, MultiMuxLayout):
                    if layout.H != 1 or layout.W != 1:
                        raise KernelError("multi mux to flat relayout requires H = W = 1; spatial tiling not supported")
                    layout = multi_flat_layout(layout.n, layout.C)
                elif attrs["layout"] != "flat" or not isinstance(layout, (FlatLayout, MultiFlatLayout)):
                    raise KernelError("relayout requires an already flat input and flat target")
            elif op.kind == "linear":
                if not isinstance(layout, (FlatLayout, MultiFlatLayout, MuxLayout, MultiMuxLayout)):
                    raise KernelError("linear input must have FlatLayout or a flattened mux input")
                w, bias = ref("w"), ref("b")
                if op.id in input_bn:
                    w, bias = cnn.fold_input_bn(w, bias, *input_bn[op.id])
                uses = consumers[output]
                if len(uses) == 1 and uses[0].kind == 'batchnorm' and output not in program.outputs:
                    bn = uses[0]; prefix = bn.attrs['bn']
                    scale, shift = fold_bn(weight(prefix+'.weight'), weight(prefix+'.bias'),
                        weight(prefix+'.running_mean'), weight(prefix+'.running_var'), bn.attrs['eps'], bias)
                    w, bias = w*scale[:, None], shift
                    output = bn.outputs[0]; skipped.add(bn.id)
                mux_input = isinstance(layout, (MuxLayout, MultiMuxLayout))
                input_size = layout.C*layout.H*layout.W if mux_input else layout.size
                if w.ndim != 2 or min(w.shape) < 1 or w.shape[1] != input_size:
                    raise KernelError("linear weight must have shape (O, lin.size)")
                count, size = w.shape
                ip, output_padded = 1 << (size-1).bit_length(), 1 << (count-1).bit_length()
                lout = multi_flat_layout(layout.n, count)
                multiple = mux_input or isinstance(layout, MultiFlatLayout) or isinstance(lout, MultiFlatLayout)
                if not multiple and (output_padded > ip or 2*ip > layout.n):
                    raise KernelError("linear requires Op <= Ip and 2*Ip <= n")
                if attrs["algo"] not in ("diag", "bsgs"):
                    raise KernelError("linear algo must be diag or bsgs")
                if bias is not None and bias.shape != (count,):
                    raise KernelError("linear bias must have length O")
                if attrs["algo"] == "bsgs":
                    sides = [1 << (max(li.size, lo.size)-1).bit_length()
                             for _, _, li in layout_blocks(layout) for _, _, lo in layout_blocks(lout)] if multiple and not mux_input else [output_padded]
                    for side in sides:
                        g = attrs["n1"] if attrs["n1"] is not None else 2**((side.bit_length()-1+1)//2)
                        if type(g) is not int or g < 1 or g & (g-1) or g > side:
                            raise KernelError("BSGS n1 must be a power of two <= Op")
                layout = lout
                extra = dict(weight=w, bias=bias)
            elif op.kind != "refresh":
                raise KernelError(f"v1 lowering 不支持 {op.kind}")
        except KernelError as exc:
            raise LoweringError(op.id, op.lineno, str(exc)) from exc
        index = len(nodes)
        lineage = set()
        for name in op.inputs:
            if name in producer:
                parent = producer[name]
                lineage.add(parent)
                lineage.update(ancestors[parent])
        nodes.append((op, output, value_layout, extra))
        ancestors[index] = lineage
        producer[output] = index
        layouts[output] = layout
    return nodes, producer, ancestors, keys


def run_tensor(program, weights, images, *, observer=None, cache=True,
               image_id=None, batch_start=0, act_mode="program"):
    """Evaluate one tensor batch; cache keys retain its original batch boundary.

    An observer normally receives ``(name, ndarray, cfg)``.  An observer with a
    ``restore`` method also accepts cached ancestor records ``(name,min,max,cfg)``;
    otherwise all operations run and ordinary observer tensors remain available.
    """
    if act_mode not in ('program', 'exact'):
        raise ValueError('act_mode must be program or exact')
    if len(program.params) != 1:
        raise LoweringError(None, None, "input count differs from program parameters")
    images_array = np.asarray(images, dtype=np.float64)
    param = program.params[0]
    packs = [op for op in program.ops if op.kind == "pack" and op.inputs == (param.name,)]
    if len(packs) != 1:
        raise LoweringError(None, param.lineno, "each parameter requires one pack operation")
    if images_array.ndim != len(param.shape) or images_array.shape[1:] != param.shape[1:]:
        raise LoweringError(packs[0].id, packs[0].lineno, "input array shape differs from parameter")
    if not np.isfinite(images_array).all():
        raise BackendError("Mock arithmetic produced a nonfinite value")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return _run(program, weights, images_array, observer, cache and act_mode == 'program',
                        identity_token(images) if image_id is None else image_id, batch_start, act_mode)
        except FloatingPointError as exc:
            raise BackendError("Mock arithmetic produced a nonfinite value") from exc


def _run(program, weights, images, observer, cache, image_id, batch_start, act_mode):
    nodes, producer, ancestors, keys = _metadata(program, weights)
    namespace = identity_token(weights), image_id, batch_start, len(images)
    hits, cache_keys = {}, {}
    for index, (op, output, layout, extra) in enumerate(nodes):
        if op.kind not in ("act", "maxpool"):
            continue
        source = op.inputs[0]
        cache_key = (keys[source], *namespace)
        cache_keys[index] = cache_key
        if cache:
            entry = _prefix_get(cache_key)
            if entry is not None:
                hits[source] = entry

    # Cut cached activation inputs from the DAG.  Unrelated or dead operations
    # still execute, retaining the original interpreter's error behavior.
    needed, covered, selected = set(), set(), set()
    shortcut = observer is None or hasattr(observer, "restore")

    def require(name):
        if name not in producer:
            return
        index = producer[name]
        if shortcut and name in hits:
            selected.add(name)
            covered.add(index)
            covered.update(ancestors[index])
            return
        if index not in needed:
            needed.add(index)
            for parent in nodes[index][0].inputs:
                require(parent)

    for output in program.outputs:
        require(output)
    for index in reversed(range(len(nodes))):
        if index not in needed and index not in covered:
            require(nodes[index][1])

    env = {program.params[0].name: images}
    restored = {}
    for name in selected:
        env[name] = hits[name].value
        restored.update(hits[name].observations)
    observations, pending = {}, {}
    remaining = Counter(name for op, _, _, _ in nodes for name in op.inputs)
    remaining.update(program.outputs)
    for index, (op, output, layout, extra) in enumerate(nodes):
        attrs = op.attrs
        if index not in needed:
            if op.kind in ("act", "maxpool"):
                record = restored[keys[output]]
                observations[keys[output]] = record
                if observer is not None:
                    observer.restore([(op.inputs[0], *record)])
        else:
            value = env[op.inputs[0]]
            try:
                if op.kind == "conv2d":
                    value = F.conv2d(torch.tensor(value, dtype=torch.float64),
                                     torch.tensor(extra["weight"], dtype=torch.float64),
                                     stride=attrs["stride"], padding=attrs["pad"], groups=attrs["groups"]).numpy()
                    if not np.isfinite(value).all():
                        raise BackendError("Mock arithmetic produced a nonfinite value")
                    if extra["scale"] is not None:
                        value = value*extra["scale"][None, :, None, None]
                    if extra["shift"] is not None:
                        value = value+extra["shift"][None, :, None, None]
                elif op.kind == "act":
                    cfg = _cfg(op)
                    if cache:
                        ancestor_observations = {keys[nodes[parent][1]]: observations[keys[nodes[parent][1]]]
                                                 for parent in ancestors[index]
                                                 if nodes[parent][0].kind in ("act", "maxpool")}
                        pending[cache_keys[index]] = _Entry(value, ancestor_observations)
                    record = float(value.min()), float(value.max()), cfg
                    observations[keys[output]] = record
                    if observer is not None:
                        observer(op.inputs[0], value, cfg)
                    if act_mode == 'exact':
                        from .kernels.poly import EXACT
                        value = EXACT[attrs['fn']](value)
                    elif attrs["method"] == "exact":
                        value = value*value
                    elif attrs["method"] == "cheb":
                        lo, hi = attrs["domain"]
                        coeffs = (_fit if cache else cheb_fit)(attrs["fn"], attrs["deg"], lo, hi)
                        a, c = 2/(hi-lo), -(hi+lo)/(hi-lo)
                        value = cheb_eval(None, value*a+c, coeffs)
                    else:
                        lo, hi = attrs["domain"]
                        if not (hi > 0 and lo == -hi):
                            raise KernelError("minimax_comp requires a symmetric domain (-B, B)")
                        sign = value*(1.0/hi)
                        for coeffs in minimax_coeffs(tuple(attrs["degrees"])):
                            sign = cheb_eval(None, sign, coeffs)
                        value = value*(sign+0.5)
                elif op.kind == 'avgpool':
                    value = F.avg_pool2d(torch.tensor(value, dtype=torch.float64), attrs['k'], attrs['stride'], attrs['pad'], count_include_pad=True).numpy()
                elif op.kind == 'maxpool':
                    cfg = ('exact',) if act_mode == 'exact' else _cfg(op)
                    if cache:
                        ancestor_observations = {keys[nodes[parent][1]]: observations[keys[nodes[parent][1]]]
                            for parent in ancestors[index] if nodes[parent][0].kind in ('act', 'maxpool')}
                        pending[cache_keys[index]] = _Entry(value, ancestor_observations)
                    stats = cnn.DifferenceStats()
                    if act_mode == 'exact':
                        apply = lambda difference: np.maximum(difference, 0)
                    elif attrs['method'] == 'cheb':
                        lo, hi = attrs['domain']
                        coeffs = (_fit if cache else cheb_fit)('relu', attrs['deg'], lo, hi)
                        apply = lambda difference: cheb_eval(None, difference*(2/(hi-lo))-(hi+lo)/(hi-lo), coeffs)
                    else:
                        lo, hi = attrs['domain']
                        coefficients = minimax_coeffs(tuple(attrs['degrees']))
                        def apply(difference):
                            sign = difference*(1.0/hi)
                            for coefficients_ in coefficients:
                                sign = cheb_eval(None, sign, coefficients_)
                            return difference*(sign+0.5)
                    def observe_difference(difference):
                        stats.add(difference)
                        if act_mode != 'exact':
                            lo, hi = attrs['domain']
                            max_u = float(np.max(np.abs(stats.value*(2/(hi-lo))-(hi+lo)/(hi-lo))))
                            if max_u > 1:
                                if observer is not None:
                                    observer(op.inputs[0], stats.value, cfg)
                                raise BackendError('maxpool comparison input leaves its domain')
                    computed = cnn.maxpool_tensor(value, kernel=attrs['k'], stride=attrs['stride'],
                        pad=attrs['pad'], apply=apply, observer=observe_difference)
                    record = stats.minimum, stats.maximum, cfg
                    observations[keys[output]] = record
                    if observer is not None:
                        observer(op.inputs[0], stats.value, cfg)
                    value = (F.max_pool2d(torch.tensor(value, dtype=torch.float64), attrs['k'], attrs['stride'], attrs['pad']).numpy()
                             if act_mode == 'exact' else computed)
                elif op.kind == 'concat':
                    value = np.concatenate([env[name] for name in op.inputs], axis=1)
                elif op.kind == 'flatten':
                    value = value.reshape(len(value), -1)
                elif op.kind == 'batchnorm':
                    if not extra['fold_forward']:
                        shape = (1, -1)+(1,)*(value.ndim-2)
                        value = value*extra['scale'].reshape(shape)+extra['shift'].reshape(shape)
                elif op.kind == "add":
                    value = value+env[op.inputs[1]]
                elif op.kind == "slice":
                    pads = extra["pads"]
                    value = np.pad(value[:, :, ::2, ::2],
                                   ((0, 0), (pads[4], pads[5]), (0, 0), (0, 0)))
                elif op.kind == "mean":
                    value = value.sum(axis=(2, 3))*(1/(layout.H*layout.W))
                elif op.kind == "relayout":
                    value = value.reshape(len(value), -1)
                elif op.kind == "linear":
                    value = value@extra["weight"].T
                    if extra["bias"] is not None:
                        value = value+extra["bias"]
                # pack and refresh preserve the tensor.
                if not np.isfinite(value).all():
                    raise BackendError("Mock arithmetic produced a nonfinite value")
                env[output] = value
            except KernelError as exc:
                raise LoweringError(op.id, op.lineno, str(exc)) from exc
        for name in op.inputs:
            remaining[name] -= 1
            if remaining[name] == 0:
                env.pop(name, None)
    result = env[program.outputs[0]]
    # An exception leaves all new batch entries uncommitted.
    for key, entry in pending.items():
        _prefix_put(key, entry)
    return result
