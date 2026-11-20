"""Trusted Torch FX layer translation for the S1 Orion availability check.

No model-name dispatch, fit, scheme initialization, or compiler call occurs here.
Shape propagation executes the supplied original model once on the supplied input.
"""
from __future__ import annotations

import builtins
import copy
import operator


def _trace_average_pool(*args, **kwargs):
    """A Python-named FX target for the unchanged functional average pool."""
    import torch.nn.functional as F
    return F.avg_pool2d(*args, **kwargs)


def build_from_torch(original, example):
    """Return an Orion Module and scalar provenance for supported Torch layers.

    Original parameters and input are untouched. Stateful layers are copied for
    each FX callsite; SiLU callsites receive distinct degree63 Orion modules.
    """
    import torch
    import torch.fx as fx
    import torch.nn.functional as F
    import orion.nn as on
    from unittest.mock import patch
    from torch.fx.passes.shape_prop import ShapeProp

    if not isinstance(original, torch.nn.Module):
        raise TypeError('Expected an original Torch module')
    if (not isinstance(example, torch.Tensor) or example.device.type != 'cpu'
            or example.dtype != torch.float64 or example.ndim < 2):
        raise ValueError('A real CPU float64 example tensor is required')
    source = original.module if isinstance(original, torch.nn.DataParallel) else original
    prefix = 'module.' if source is not original else ''
    private = copy.deepcopy(source).cpu().double()
    private.eval()
    private.requires_grad_(False)
    # The C-bound function rejects a Proxy kernel before FX sees the call. Record
    # just that function under a Python name, restoring it before any evaluation.
    original_pool = F.avg_pool2d

    def capture_pool(*args, **kwargs):
        if args and isinstance(args[0], fx.Proxy):
            return args[0].tracer.create_proxy('call_function', _trace_average_pool, args, kwargs)
        return original_pool(*args, **kwargs)

    with patch.object(F, 'avg_pool2d', capture_pool):
        traced_graph = fx.Tracer().trace(private)
    traced = fx.GraphModule(private, traced_graph)
    with torch.no_grad():
        ShapeProp(traced).propagate(example.detach().clone())

    graph = fx.Graph()
    root = torch.nn.Module()
    env = {}
    mapping = []
    state_rows = []
    selectors = []
    skipped = set()
    fused = {}
    counter = 0

    def shape(node):
        metadata = node.meta.get('tensor_meta')
        if metadata is None:
            raise ValueError('Tensor shape unavailable at ' + node.name)
        return tuple(metadata.shape)

    def argument(args, kwargs, index, name, default=None):
        return args[index] if len(args) > index else kwargs.get(name, default)

    def full_slice(value):
        return isinstance(value, slice) and value.start in (None, 0) and value.stop is None

    # Recognize a general zero channel-padding/strided-spatial selection.
    for node in traced.graph.nodes:
        if node.op != 'call_function' or node.target is not F.pad:
            continue
        inp = node.args[0]
        pad = argument(node.args, node.kwargs, 1, 'pad')
        mode = argument(node.args, node.kwargs, 2, 'mode', 'constant')
        value = argument(node.args, node.kwargs, 3, 'value', None)
        if (not isinstance(inp, fx.Node) or inp.op != 'call_function'
                or inp.target is not operator.getitem or set(inp.users) != {node}
                or mode != 'constant' or value not in (None, 0)
                or not isinstance(pad, (tuple, list)) or len(pad) != 6
                or any(type(v) is not int or v < 0 for v in pad)
                or any(pad[:4])):
            continue
        base, index = inp.args
        if (not isinstance(base, fx.Node) or len(shape(base)) != 4
                or not isinstance(index, tuple) or len(index) != 4
                or not all(full_slice(v) for v in index)
                or index[0].step not in (None, 1) or index[1].step not in (None, 1)):
            continue
        stride = tuple(v.step if v.step is not None else 1 for v in index[2:])
        if any(type(v) is not int or v <= 0 for v in stride):
            continue
        fused[node] = (base, pad[4], pad[5], stride, inp)
        skipped.add(inp)

    def load_state(dst, src, node, name):
        values = {}
        for key, value in src.state_dict().items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite original state at ' + str(node.target) + '.' + key)
            values[key] = value.detach().cpu().to(
                dtype=torch.float64 if value.is_floating_point() else value.dtype).clone()
        dst.cpu().double()
        # Preserve the integral BN counter dtype as well as its value.
        for key, value in values.items():
            if not value.is_floating_point():
                owner, _, leaf = key.rpartition('.')
                parent = dst.get_submodule(owner) if owner else dst
                if key not in dict(dst.named_buffers()):
                    raise ValueError('Unexpected integral parameter at ' + key)
                setattr(parent, leaf, getattr(parent, leaf).to(value.dtype))
        dst.load_state_dict(values, strict=True)
        dst.requires_grad_(False)
        for key, actual in dst.state_dict().items():
            if actual.dtype != values[key].dtype or not torch.equal(actual, values[key]):
                raise ValueError('Copied original state differs at ' + key)
            original_value = source.get_submodule(str(node.target)).state_dict()[key]
            if not torch.equal(actual.to(original_value.dtype), original_value.detach().cpu()):
                raise ValueError('Original parameter is not represented exactly')
            state_rows.append(dict(original_name=prefix + str(node.target) + '.' + key,
                translated_name=name + '.' + key, shape=list(actual.shape),
                original_dtype=str(original_value.dtype), translated_dtype=str(actual.dtype),
                numerically_equal=True))

    def install(node, module, args, src=None, role=None):
        nonlocal counter
        name = 'op_' + str(counter).zfill(4)
        counter += 1
        module.cpu().double()
        if src is not None:
            load_state(module, src, node, name)
        module.requires_grad_(False)
        module.eval()  # Orion eval intentionally returns None.
        root.add_module(name, module)
        env[node] = graph.call_module(name, args)
        mapping.append(dict(source_node=node.name, source_op=node.op,
            original_target=str(node.target), translated_target=name,
            translated_type=type(module).__name__, output_shape=list(shape(node)),
            role=role, degree=63 if isinstance(module, on.SiLU) else None))
        return name

    def flatten(node, inp):
        before, after = shape(inp), shape(node)
        if len(after) != 2 or after[0] != before[0]:
            raise ValueError('Only batch-preserving flatten is supported')
        total = 1
        for value in before[1:]:
            total *= value
        if after[1] != total:
            raise ValueError('View changes element ordering beyond flatten')
        install(node, on.Flatten(), (env[inp],))

    def avgpool(node, inp, kernel, stride, padding, ceil=False, count=True, divisor=None):
        if ceil or not count or divisor is not None:
            raise ValueError('Average pool options are outside the Orion layer contract')
        module = on.AvgPool2d(kernel, stride=stride, padding=padding)
        module.cpu().double()
        size = module.kernel_size[0] * module.kernel_size[1]
        with torch.no_grad():
            module.weight.fill_(1.0 / size)
        install(node, module, (env[inp],), role='fixed_average_pool')

    for node in traced.graph.nodes:
        if node in skipped:
            continue
        if node.op == 'placeholder':
            env[node] = graph.placeholder(str(node.target))
            continue
        if node.op == 'output':
            if not isinstance(node.args[0], fx.Node):
                raise ValueError('A single tensor output is required')
            graph.output(env[node.args[0]])
            continue
        if node in fused:
            base, left, right, stride, sliced = fused[node]
            channels = shape(base)[1]
            module = on.Conv2d(channels, channels + left + right, 1,
                              stride=stride, padding=0, bias=False).double()
            with torch.no_grad():
                module.weight.zero_()
                for channel in range(channels):
                    module.weight[left + channel, channel, 0, 0] = 1.0
            name = install(node, module, (env[base],), role='zero_channel_pad_strided_selector')
            selectors.append(dict(translated_target=name, source_nodes=[sliced.name, node.name],
                in_channels=channels, out_channels=channels + left + right,
                channel_pad=[left, right], stride=list(stride), fixed_nonzero_values=channels,
                weight_dtype='torch.float64', weight_values='0 and 1', trained=False))
            continue
        if node.op == 'call_module':
            src = traced.get_submodule(str(node.target))
            inp = node.args[0]
            if type(src) is torch.nn.Conv2d:
                if src.padding_mode != 'zeros' or isinstance(src.padding, str):
                    raise ValueError('Only explicit zero-padded convolution is supported')
                install(node, on.Conv2d(src.in_channels, src.out_channels, src.kernel_size,
                    stride=src.stride, padding=src.padding, dilation=src.dilation,
                    groups=src.groups, bias=src.bias is not None), (env[inp],), src)
            elif type(src) is torch.nn.Linear:
                install(node, on.Linear(src.in_features, src.out_features,
                    bias=src.bias is not None), (env[inp],), src)
            elif type(src) in (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d):
                if not src.track_running_stats:
                    raise ValueError('BatchNorm must retain original running statistics')
                cls = on.BatchNorm1d if type(src) is torch.nn.BatchNorm1d else on.BatchNorm2d
                install(node, cls(src.num_features, eps=src.eps, momentum=src.momentum,
                    affine=src.affine), (env[inp],), src)
            elif type(src) is torch.nn.SiLU:
                if src.inplace:
                    raise ValueError('In-place SiLU is outside the translation contract')
                install(node, on.SiLU(degree=63), (env[inp],), role='stateless_activation_callsite')
            elif type(src) is torch.nn.AvgPool2d:
                avgpool(node, inp, src.kernel_size, src.stride, src.padding,
                        src.ceil_mode, src.count_include_pad, src.divisor_override)
            elif type(src) is torch.nn.Flatten:
                if src.start_dim != 1 or src.end_dim != -1:
                    raise ValueError('Only flatten(start_dim=1,end_dim=-1) is supported')
                flatten(node, inp)
            elif type(src) in (torch.nn.Identity, torch.nn.Dropout):
                env[node] = env[inp]
                mapping.append(dict(source_node=node.name, original_target=str(node.target),
                    translated_type='elided_eval_identity', role=type(src).__name__))
            else:
                raise ValueError('Unsupported Torch layer: ' + type(src).__name__)
            continue
        if node.op == 'call_method' and node.target == 'size':
            dims = shape(node.args[0])
            env[node] = dims[node.args[1]] if len(node.args) > 1 else dims
            continue
        if node.op == 'call_function' and node.target is builtins.getattr:
            if node.args[1] != 'shape':
                raise ValueError('Only Tensor.shape metadata is supported')
            env[node] = shape(node.args[0])
            continue
        if node.op == 'call_function' and node.target is operator.getitem:
            base, index = node.args
            if isinstance(base, fx.Node) and isinstance(env.get(base), (tuple, list)):
                env[node] = env[base][index]
                continue
            raise ValueError('Tensor slicing must be a supported zero-pad selector pattern')
        if node.op == 'call_method' and node.target in ('view', 'reshape', 'flatten'):
            flatten(node, node.args[0])
            continue
        if node.op == 'call_function' and node.target is torch.flatten:
            if argument(node.args, node.kwargs, 1, 'start_dim', 0) != 1 or argument(
                    node.args, node.kwargs, 2, 'end_dim', -1) != -1:
                raise ValueError('Only batch-preserving functional flatten is supported')
            flatten(node, node.args[0])
            continue
        if node.op == 'call_function' and node.target in (operator.add, torch.add):
            if len(node.args) != 2 or node.kwargs or shape(node.args[0]) != shape(node.args[1]):
                raise ValueError('Only equal-shape tensor addition is supported')
            install(node, on.Add(), tuple(env[v] for v in node.args))
            continue
        if node.op == 'call_function' and node.target in (F.avg_pool2d, _trace_average_pool):
            args = fx.map_arg(node.args, lambda v: env[v])
            kwargs = fx.map_arg(node.kwargs, lambda v: env[v])
            avgpool(node, node.args[0], argument(args, kwargs, 1, 'kernel_size'),
                argument(args, kwargs, 2, 'stride'), argument(args, kwargs, 3, 'padding', 0),
                argument(args, kwargs, 4, 'ceil_mode', False),
                argument(args, kwargs, 5, 'count_include_pad', True),
                argument(args, kwargs, 6, 'divisor_override'))
            continue
        if node.op == 'call_function' and node.target is F.silu:
            if argument(node.args, node.kwargs, 1, 'inplace', False):
                raise ValueError('In-place functional SiLU is unsupported')
            install(node, on.SiLU(degree=63), (env[node.args[0]],),
                    role='stateless_activation_callsite')
            continue
        raise ValueError('Unsupported FX operation: ' + node.op + ' ' + str(node.target))

    graph.lint()
    lowered = fx.GraphModule(root, graph)

    class Translated(on.Module):
        def __init__(self):
            super().__init__()
            self.lowered = lowered

        def forward(self, x):
            return self.lowered(x)

    result = Translated()
    result.eval()
    return result, dict(schema='s1-orion-layer-translation-v1', translator='torch.fx',
        model_name_dispatch=False, data_parallel_unwrapped=source is not original,
        source_model_mutated=False, input_mutated=False, original_shape_propagation_calls=1,
        input_shape=list(example.shape), input_dtype=str(example.dtype),
        original_to_translated_calls=mapping, copied_state=state_rows,
        copied_state_values_equal=all(r['numerically_equal'] for r in state_rows),
        stateful_layers_copied_per_callsite=True, shared_SiLU_split_per_callsite=True,
        activation_degree=63, selectors=selectors, fit_calls=0, compile_calls=0,
        runtime_default_dtype_required='torch.float64', unknown_operations='reject')
