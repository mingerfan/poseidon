"""Exact wide-pool/interior-mask representation, then strict math and HEIR compile.

The original24-node graph, saved degree63 program and learned parameters remain
the reference. Only two pool/next-convolution representations change. The host
owns supervision and the registry; the numeric child receives explicit paths.
"""
import argparse
import ast
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import time
from types import FunctionType, SimpleNamespace

MODEL = 'alexnet_silu_avgpool'
VARIANT = 'alex-overlapping-wide-pool-interior-mask-v1'
ROUND = 'HEIR-ALEX-WIDE-POOL'
MODE = 'explicit-wide-pool-mask-export-strict-math-native-compile'
MODULE = 'pt_agent.experiments.heir_alexnet_wide_pool'
TOTAL_SECONDS = 10800
WORKER_SECONDS = 1800
MEMORY_GIB_CHOICES = (24,)
COUNTERS = ('graph_load', 'representation', 'export', 'constant_check', 'probe_load',
            'reference_forward', 'adapter_forward')
ORIGINAL_OPS = dict(act=7, avgpool=3, batchnorm=5, conv2d=5, linear=3, reshape=1)
PAIRS = (('n3', 'n4', (1, 96, 32, 32), (1, 96, 16, 16),
          (1, 256, 16, 16), 'Conv2d_2.weight', 5, 2),
         ('n7', 'n8', (1, 256, 16, 16), (1, 256, 8, 8),
          (1, 384, 8, 8), 'Conv2d_3.weight', 3, 1))


def _pair(value, expected):
    return isinstance(value, (list, tuple)) and all(type(x) is int for x in value) and tuple(value) == expected


def pair_spec(input_shape, pool_attrs, pooled_shape, conv_attrs, weight_shape, output_shape):
    """Describe the actual four-step representation, including finite borders."""
    from .heir_common_models import avgpool_spec
    source = avgpool_spec(input_shape, pool_attrs, pooled_shape)
    shape, pooled, output, weight = map(tuple, (input_shape, pooled_shape, output_shape, weight_shape))
    if (source['kernel'] != [3, 3] or source['stride'] != [2, 2] or source['padding'] != [1, 1]
            or shape[2] % 2 or shape[3] % 2
            or set(conv_attrs)-{'weight', 'bias', 'stride', 'padding', 'dilation', 'groups'}
            or not _pair(conv_attrs.get('stride'), (1, 1))
            or not _pair(conv_attrs.get('dilation'), (1, 1))
            or type(conv_attrs.get('groups')) is not int or conv_attrs['groups'] != 1
            or len(weight) != 4 or weight[1] != shape[1] or weight[2] != weight[3]
            or weight[2] not in (3, 5) or not _pair(conv_attrs.get('padding'), (weight[2]//2,)*2)
            or output != (1, weight[0], *pooled[2:])):
        raise ValueError('Wide-pool requires the original overlapping3/s2/p1 and adjacent odd learned convolution')
    p = weight[2]//2
    wide = (1, shape[1], pooled[2]+2*p, pooled[3]+2*p)
    return dict(input_shape=list(shape), pooled_shape=list(pooled), output_shape=list(output),
        weight_shape=list(weight), pool=source, learned_padding=p, wide_padding=1+2*p,
        wide_shape=list(wide), mask_interior=[p, p+pooled[2], p, p+pooled[3]],
        steps=['wide_channel_diagonal_pool', 'multiply_fixed_interior_mask', 'learned_conv_padding0'],
        divisor=9, bias_added_once=True, mask_elements=math.prod(wide), mask_bytes=math.prod(wide)*8,
        learned_weights_modified=False, learned_bias_modified=False)


def find_pairs(graph):
    """Gate exactly the two source edges using shapes, never parameter contents."""
    if graph.name != MODEL or len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError('Original single-input/output Alex graph required')
    ids = [n.id for n in graph.nodes]
    if len(set(ids)) != len(ids):raise ValueError('Unique original node identifiers required')
    positions = {name: i for i, name in enumerate(ids)}
    pairs = []
    for pool_id, conv_id, before, pooled, after, name, kernel, padding in PAIRS:
        if pool_id not in positions or positions[pool_id]+1 >= len(ids):
            raise ValueError('Original pool/conv edge missing')
        pool = graph.nodes[positions[pool_id]];conv = graph.nodes[positions[pool_id]+1]
        if (pool.op != 'avgpool' or conv.op != 'conv2d' or conv.id != conv_id
                or len(pool.inputs) != 1 or len(pool.outputs) != 1 or len(conv.outputs) != 1
                or list(conv.inputs) != list(pool.outputs) or pool.outputs[0] in graph.outputs
                or [n.id for n in graph.nodes for x in n.inputs if x == pool.outputs[0]] != [conv_id]
                or conv.attrs.get('weight') != name or conv.attrs.get('bias') != name.removesuffix('weight')+'bias'):
            raise ValueError('Exclusive adjacent original named pool/conv edge required')
        shapes = [tuple(graph.tensors[x].shape) for x in (pool.inputs[0], pool.outputs[0], conv.outputs[0])]
        if shapes != [before, pooled, after]:raise ValueError('Original Alex pair geometry differs')
        weight_shape = tuple(graph.tensors[name].shape)
        if weight_shape != (after[1], before[1], kernel, kernel):raise ValueError('Original learned weight shape differs')
        if tuple(graph.tensors[conv.attrs['bias']].shape) != (after[1],):raise ValueError('Original learned bias shape differs')
        spec = pair_spec(before, pool.attrs, pooled, conv.attrs, weight_shape, after)
        spec.update(pool_id=pool_id, conv_id=conv_id, input=pool.inputs[0], pooled=pool.outputs[0],
            output=conv.outputs[0], weight=name, bias=conv.attrs['bias'], mask_name='derived-wide-pool-mask:'+pool_id)
        pairs.append(spec)
    return pairs


def pair_constants(spec):
    import numpy as np
    from .heir_common_models import avgpool_weights
    diagonal, _ = avgpool_weights(tuple(spec['input_shape']),
        dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1], ceil_mode=False, count_include_pad=True),
        tuple(spec['pooled_shape']))
    mask = np.zeros(spec['wide_shape'], dtype=np.float64)
    top, bottom, left, right = spec['mask_interior'];mask[:, :, top:bottom, left:right] = 1.
    mask.setflags(write=False)
    return diagonal, mask


def execute_pair(spec, image, weight, bias, conv):
    """Execute the same representation consumed by the private MLIR emitter."""
    import numpy as np
    if (tuple(image.shape) != tuple(spec['input_shape']) or tuple(weight.shape) != tuple(spec['weight_shape'])
            or image.dtype != np.float64 or weight.dtype != np.float64
            or not np.isfinite(image).all() or not np.isfinite(weight).all()
            or (bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != np.float64 or not np.isfinite(bias).all()))):
        raise ValueError('Pair original finite float64 source shapes differ')
    diagonal, mask = pair_constants(spec)
    wide = conv(image, diagonal, stride=(2, 2), padding=(spec['wide_padding'],)*2)
    if tuple(wide.shape) != tuple(spec['wide_shape']):raise ValueError('Wide pool result shape differs')
    result = conv(wide*mask, weight, stride=(1, 1), padding=(0, 0))
    if bias is not None:result = result+bias[None, :, None, None]
    if tuple(result.shape) != tuple(spec['output_shape']) or not np.isfinite(result).all():
        raise ValueError('Pair output shape/finite contract differs')
    return result


def representation_contract(original, pairs):
    return dict(role='explicit-physical-wide-pool-representation', variant=VARIANT,
        original_logical_graph_nodes=original['graph_nodes'], original_activation_ids=
        [n['id'] for n in original['nodes'] if n['op'] == 'act'],
        pairs=pairs, wide_pool_count=2, new_mask_constant_count=2,
        mask_elements=sum(p['mask_elements'] for p in pairs), mask_bytes=sum(p['mask_bytes'] for p in pairs),
        emitted_tensor_constant_count=31, original_graph_modified=False, learned_weights_modified=False,
        learned_bias_modified=False, real_number_equivalence=True, binary64_math_passed=None,
        native_layout_success=None)


def export_graph(graph, program_text):
    """Reuse original exporter control flow; defer only the gated pool edges."""
    from . import heir_cnn_export as original
    from .heir_common_models import avgpool_spec, graph_contract, checked_export_contract
    pairs = find_pairs(graph)
    state = {}
    class WideEmitter(original.Emitter):
        def __init__(self):
            super().__init__();self.pending = {};self.mask_records = []
            state.update(pending=self.pending, masks=self.mask_records)
        def local_avgpool(self, value, attrs, expected):
            # An original tensor's SSA handle identifies the graph edge without a global mutation.
            matches = [p for p in pairs if tuple(p['input_shape']) == value.shape and tuple(p['pooled_shape']) == tuple(expected)]
            if not matches:return super().local_avgpool(value, attrs, expected)
            if len(matches) != 1:raise ValueError('Ambiguous deferred original pool')
            pair = matches[0];spec = avgpool_spec(value.shape, attrs, expected)
            if spec != pair['pool']:raise ValueError('Deferred pool attributes differ')
            placeholder = self.fresh(expected, 'deferred_pool')
            self.pending[placeholder.name] = (value, pair)
            return placeholder, spec
        def conv(self, value, weight, attrs, expected):
            pending = self.pending.pop(value.name, None)
            if pending is None:return super().conv(value, weight, attrs, expected)
            before, pair = pending
            if (attrs.get('weight') != pair['weight'] or tuple(weight.shape) != tuple(pair['weight_shape'])
                    or tuple(expected) != tuple(pair['output_shape']) or tuple(attrs['padding']) != (pair['learned_padding'],)*2):
                raise ValueError('Deferred pool was not consumed by its original learned convolution')
            diagonal, mask = pair_constants(pair)
            wide = super().conv(before, diagonal, dict(stride=[2, 2], padding=[pair['wide_padding']]*2,
                dilation=[1, 1], groups=1), tuple(pair['wide_shape']))
            constant = self.constant(mask)
            self.mask_records.append(dict(name=pair['mask_name'], ssa=constant.name, shape=list(mask.shape)))
            masked = self.elementwise(wide, constant, 'mulf')
            return super().conv(masked, weight, dict(attrs, padding=[0, 0]), expected)
    namespace = dict(vars(original));namespace['Emitter'] = WideEmitter
    function = original.export_graph
    delegated = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    delegated.__kwdefaults__ = function.__kwdefaults__
    text, metadata = delegated(graph, program_text, common_model=MODEL)
    if state['pending'] or len(state['masks']) != 2 or 'deferred_pool' in text:
        raise ValueError('Every deferred pool must disappear into its one original convolution')
    checked_export_contract(metadata, graph_contract(graph, MODEL))
    metadata.update(variant=VARIANT, wide_pool_pairs=pairs, mask_constants=state['masks'],
        physical_representation=representation_contract(graph_contract(graph, MODEL), pairs),
        mlir_bytes=len(text.encode()), original_graph_changed=False, original_weights_changed=False)
    return text, metadata


def check_constants(graph, text, exported):
    import numpy as np
    from .heir_cnn_math_check import check_constants as original, check_math
    masks = exported['mask_constants'];pairs = find_pairs(graph);without = text;mask_rows = {}
    for declaration, pair in zip(masks, pairs):
        if declaration['name'] != pair['mask_name']:raise ValueError('Mask declaration source identity differs')
        pattern = r'^\s*'+re.escape(declaration['ssa'])+r' = arith.constant dense<(.+)> : tensor<([^>]+)>$'
        found = list(re.finditer(pattern, without, re.MULTILINE))
        if len(found) != 1:raise ValueError('Exactly one explicit mask tensor constant required')
        match = found[0];actual = np.asarray(ast.literal_eval(match.group(1)), dtype=np.float64)
        _, wanted = pair_constants(pair)
        row = check_math(wanted, actual)
        row.update(name=pair['mask_name'], numerically_equal=bool(np.array_equal(wanted, actual)),
                   source_dtype='float64', emitted_type=match.group(2), public_fixed_mask=True)
        if not row['passed'] or not row['numerically_equal']:raise ValueError('Emitted fixed0/1 mask differs')
        mask_rows[pair['pool_id']] = row
        without = without[:match.start()]+without[match.end():]
    original_rows = original(graph, without, common_model=MODEL)
    rows = []
    for row in original_rows:
        rows.append(row)
        if row['name'].startswith('derived-exact-local-avgpool:'):
            key = row['name'].split(':')[-1]
            if key in mask_rows:rows.append(mask_rows[key])
    if len(rows) != 31:raise ValueError('All29 original/diagonal and2 new mask constants required')
    return rows


def evaluate_graph(graph, configs, image):
    """Original arithmetic outside the two explicitly inventoried pair adapters."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ..kernels.conv import fold_bn
    from .heir_cnn_export import chebyshev_tape
    from .heir_common_models import avgpool_spec, reshape_spec
    pairs = find_pairs(graph);deferred = {p['pool_id']: p for p in pairs};convs = {p['conv_id']: p for p in pairs}
    values = {graph.inputs[0]: image};observations = [];activation = 0
    remaining = Counter(x for n in graph.nodes for x in n.inputs)+Counter(graph.outputs)
    pending = {}
    def conv(x, w, *, stride, padding, bias=None):
        return F.conv2d(torch.tensor(x, dtype=torch.float64), torch.tensor(w, dtype=torch.float64),
                        bias=torch.tensor(bias, dtype=torch.float64) if bias is not None else None,
                        stride=stride, padding=padding).numpy()
    for node in graph.nodes:
        a = node.attrs
        if node.id in deferred:
            pair = deferred[node.id];pending[node.outputs[0]] = values[node.inputs[0]]
            # Only the immediately following, exclusive convolution can consume this token.
            values[node.outputs[0]] = None
        else:
            x = values[node.inputs[0]]
            if node.id in convs:
                pair = convs[node.id]
                value = execute_pair(pair, pending.pop(node.inputs[0]), graph.weights[a['weight']],
                    graph.weights[a['bias']] if a.get('bias') else None, conv)
            elif node.op == 'conv2d':
                value = conv(x, graph.weights[a['weight']], stride=a['stride'], padding=a['padding'],
                    bias=graph.weights[a['bias']] if a.get('bias') else None)
            elif node.op == 'batchnorm':
                scale, shift = fold_bn(*(graph.weights[a[k]] for k in ('weight', 'bias', 'mean', 'var')), a['eps'])
                broadcast = (1, x.shape[1])+((1,)*(x.ndim-2))
                value = x*scale.reshape(broadcast)+shift.reshape(broadcast)
            elif node.op == 'act':
                cfg = configs[activation];observations.append(x.copy())
                tape, output, _ = chebyshev_tape(cfg['fn'], cfg['deg'], cfg['domain'])
                value = tape.evaluate(x, output);activation += 1
            elif node.op == 'linear':
                value = x@graph.weights[a['weight']].T
                if a.get('bias'):value = value+graph.weights[a['bias']]
            elif node.op == 'avgpool':
                spec = avgpool_spec(x.shape, a, graph.tensors[node.outputs[0]].shape)
                ph, pw = spec['padding'];sh, sw = spec['stride'];kh, kw = spec['kernel']
                padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
                value = np.empty(spec['output_shape'], dtype=np.float64)
                for h in range(value.shape[2]):
                    for w in range(value.shape[3]):
                        value[:, :, h, w] = padded[:, :, h*sh:h*sh+kh, w*sw:w*sw+kw].sum((2, 3))*(1/spec['divisor'])
            elif node.op == 'reshape':
                spec = reshape_spec(x.shape, a, graph.tensors[node.outputs[0]].shape)
                value = x.reshape(spec['output_shape'], order='C')
            else:raise ValueError('Unexpected original Alex operation: '+node.op)
            if (value.shape != tuple(graph.tensors[node.outputs[0]].shape)
                    or value.dtype != np.float64 or not np.isfinite(value).all()):
                raise ValueError('Adapter original output shape/dtype/finite contract differs')
            values[node.outputs[0]] = value
        for name in node.inputs:
            remaining[name] -= 1
            if not remaining[name]:values.pop(name, None)
    if pending:raise ValueError('Unconsumed original pool edge')
    return values[graph.outputs[0]], observations


def source_gate(source):
    from .heir_common_model_pipeline import source_gate as original
    from .heir_vgg_fused_pipeline import read
    checked = original(MODEL, source);contract = checked['contract']
    if (contract['graph_nodes'] != 24 or contract['operator_counts'] != ORIGINAL_OPS
            or contract['activation_count'] != 7 or contract['original_tensor_constant_count'] != 26
            or contract['input_shape'] != [1, 3, 32, 32] or contract['output_shape'] != [1, 10]):
        raise ValueError('Original24-node Alex source architecture differs')
    artifacts = read(source['intake_result'])['worker']['artifacts']
    expected = dict(graph_dir=Path(artifacts['model_info'])/'graph', probes=Path(artifacts['training_probes']),
                    probe_source=Path(artifacts['training_probe_source']))
    if any(source[k].resolve() != p.resolve() for k, p in expected.items()):
        raise ValueError('Graph and probes must belong to the admitted original intake')
    for name, path in source.items():
        target = path/'graph.json' if name == 'graph_dir' else path
        if not target.is_file() or target.stat().st_size <= 0:raise ValueError('Source missing: '+name)
    # Only JSON shapes/edges are inspected here. No parameter archive is opened.
    metadata = read(source['graph_dir']/'graph.json')
    structural = SimpleNamespace(name=metadata['name'], inputs=metadata['inputs'], outputs=metadata['outputs'],
        nodes=[SimpleNamespace(id=n, op=op, inputs=ins, outputs=outs, attrs=a)
               for n, op, ins, outs, a, _ in metadata['nodes']],
        tensors={name: SimpleNamespace(shape=tuple(shape)) for name, shape, _ in metadata['tensors']})
    pairs = find_pairs(structural)
    checked.update(variant=VARIANT, original_model_contract=contract, wide_pool_pairs=pairs,
        lowered_adapter_contract=None, source_payloads_read=False,
        prior_failed_representation='original pooled-output tensor.pad before the next learned convolution')
    return checked


def _runtime():
    from .heir_vgg_fused_pipeline import _runtime as original
    api = original();api.update(export_graph=export_graph, check_constants=check_constants,
                                evaluate_graph=evaluate_graph)
    return api


def initial_worker():
    from .heir_vgg_fused_pipeline import ZERO_FIELDS, FORMAL
    return dict(status='running', error=None, model=MODEL, variant=VARIANT,
        attempted={k: 0 for k in COUNTERS}, completed={k: 0 for k in COUNTERS},
        activation_pairs=0, logit_elements=0, probes=[], constants=[], export=None,
        original_model_contract=None, lowered_adapter_contract=None, strict_math_passed=None,
        quality_selection=False, original_graph_modified=False, original_weights_modified=False,
        **{k: 0 for k in ZERO_FIELDS}, **{k: None for k in FORMAL})


def worker(args):
    from .heir_vgg_fused_pipeline import durable_json, read, SOURCE_FIELDS, StageRecorder, read_events, summarize_events
    out = args.out.resolve();out.mkdir(parents=True, exist_ok=False)
    result = initial_worker();recorder = StageRecorder(out/'events.jsonl')
    save = lambda: durable_json(out/'result.json', result)
    def stage(name, action):
        result['attempted'][name] += 1;save()
        with recorder.span(name):value = action()
        result['completed'][name] += 1;save();return value
    try:
        with recorder.span('wide-pool-export-and-three-training-math'):
            source = {name: getattr(args, name).resolve() for name in SOURCE_FIELDS}
            result['source'] = source_gate(source);api = _runtime();np = api['np']
            graph = stage('graph_load', lambda: api['load_graph'](source['graph_dir']))
            original = api['graph_contract'](graph, MODEL)
            if original != result['source']['contract']:raise ValueError('Loaded graph source contract differs')
            pairs = stage('representation', lambda: find_pairs(graph))
            result.update(original_model_contract=original, lowered_adapter_contract=representation_contract(original, pairs))
            text_source = source['program'].read_text();program, configs = api['saved_activations'](text_source)
            text, exported = stage('export', lambda: api['export_graph'](graph, text_source))
            result['export'] = exported
            (out/'model.mlir').write_text(text);durable_json(out/'export.json', exported)
            result['constants'] = stage('constant_check', lambda: api['check_constants'](graph, text, exported))
            def probes():
                with np.load(source['probes'], allow_pickle=False) as archive:
                    images, indices = archive['input'], archive['indices']
                if (images.dtype != np.float64 or images.shape != (3, 3, 32, 32)
                        or indices.tolist() != [0, 1, 2] or not np.isfinite(images).all()):
                    raise ValueError('Original saved training probe shape/dtype/indices differ')
                return images
            images = stage('probe_load', probes)
            for index, sample_id in enumerate(read(source['probe_source'])['sample_ids']):
                originals = [];result['attempted']['reference_forward'] += 1;save()
                with recorder.span('original-program-reference', sample_id=sample_id):
                    reference = api['run_tensor'](program, graph.weights, images[index:index+1], cache=False,
                        observer=lambda name, value, cfg: originals.append(value.copy()))
                result['completed']['reference_forward'] += 1;result['attempted']['adapter_forward'] += 1;save()
                with recorder.span('wide-pool-adapter-forward', sample_id=sample_id):
                    observed, activations = api['evaluate_graph'](graph, configs, images[index:index+1])
                result['completed']['adapter_forward'] += 1
                if (len(originals) != 7 or len(activations) != 7 or reference.shape != (1, 10)
                        or any(value.dtype != np.float64 for value in [reference, *originals])):
                    raise ValueError('Original7 activations and10 float64 logits required')
                checks = [api['check_math'](a, b) for a, b in zip(originals, activations)]
                check = api['check_math'](reference, observed)
                result['activation_pairs'] += 7;result['logit_elements'] += 10
                result['probes'].append(dict(sample_id=sample_id, reference=reference.tolist(), observed=observed.tolist(),
                    activation_checks=checks, logit_check=check));save()
                if not check['passed'] or not all(row['passed'] for row in checks):
                    raise ValueError('Original strict1e-12 mathematical criterion failed')
            result.update(status='ok', strict_math_passed=True)
            result['lowered_adapter_contract']['binary64_math_passed'] = True
    except BaseException as exc:
        result.update(status='error', error=type(exc).__name__+': '+str(exc)[:4000])
    finally:
        result['resource_ru_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns());save()
    return result


def check_worker(record, source):
    from .heir_vgg_fused_pipeline import ZERO_FIELDS, FORMAL
    from .heir_common_models import checked_export_contract, expected_constant_names
    if (record.get('status') != 'ok' or record.get('model') != MODEL or record.get('variant') != VARIANT
            or record.get('source') != source or record.get('original_model_contract') != source['contract']
            or record.get('strict_math_passed') is not True or record.get('quality_selection') is not False
            or record.get('original_graph_modified') is not False or record.get('original_weights_modified') is not False
            or record.get('activation_pairs') != 21 or record.get('logit_elements') != 30
            or any(record.get(k) != 0 for k in ZERO_FIELDS) or any(record.get(k) is not None for k in FORMAL)):
        raise ValueError('Completed source-bound Alex export/math required')
    for name in COUNTERS:
        count = 3 if name.endswith('forward') else 1
        if record['attempted'].get(name) != count or record['completed'].get(name) != count:
            raise ValueError('Single-pass mathematical stage counts differ')
    exported = record['export'];checked_export_contract(exported, source['contract'])
    lowered = copy.deepcopy(record['lowered_adapter_contract']);lowered['binary64_math_passed'] = None
    if (exported['wide_pool_pairs'] != source['wide_pool_pairs']
            or lowered != representation_contract(source['contract'], source['wide_pool_pairs'])):
        raise ValueError('Explicit physical representation contract differs')
    if len(exported['wide_pool_pairs']) != 2 or len(exported['mask_constants']) != 2:
        raise ValueError('Exactly2 wide pools and2 new masks required')
    names = expected_constant_names(source['contract'], exported);wanted = []
    for name in names:
        wanted.append(name)
        if name in ('derived-exact-local-avgpool:n3', 'derived-exact-local-avgpool:n7'):
            wanted.append('derived-wide-pool-mask:'+name.split(':')[-1])
    if ([r.get('name') for r in record['constants']] != wanted or len(wanted) != 31
            or not all(r.get('passed') is True and r.get('numerically_equal') is True for r in record['constants'])):
        raise ValueError('All31 original/public emitted constants must be checked')
    rows = record['probes']
    if [r['sample_id'] for r in rows] != ['train:0', 'train:1', 'train:2']:raise ValueError('Only the same three training inputs admitted')
    for row in rows:
        if len(row['activation_checks']) != 7 or not all(r['passed'] is True for r in row['activation_checks']):
            raise ValueError('All21 activation comparisons required')
        if (not row['logit_check']['passed'] or len(row['reference']) != 1 or len(row['observed']) != 1
                or len(row['reference'][0]) != 10 or len(row['observed'][0]) != 10):
            raise ValueError('All30 original logits required')
        for a, b in zip(row['reference'][0], row['observed'][0]):
            if (type(a) not in (int, float) or type(b) not in (int, float) or not math.isfinite(a)
                    or not math.isfinite(b) or abs(a-b) > 1e-12+1e-12*abs(a)):
                raise ValueError('Saved full logits fail the original criterion')
    return dict(passed=True, training_samples=3, activation_pairs=21, logit_elements=30,
                emitted_constants=31, added_masks=2, native_layout_success=None)


def validate_memory_gib(value):
    if type(value) is not int or value != 24:raise ValueError('Explicit24GiB address-space allowance required')
    return value


def _snapshot(out):
    from . import heir_vgg_fused_pipeline, heir_common_model_pipeline, heir_common_models, heir_cnn_export, heir_cnn_math_check, heir_cnn_preflight
    modules = [sys.modules[__name__], heir_vgg_fused_pipeline, heir_common_model_pipeline, heir_common_models,
               heir_cnn_export, heir_cnn_math_check, heir_cnn_preflight]
    directory = out/'source';directory.mkdir();rows = []
    for module in modules:
        source = Path(module.__file__).resolve();target = directory/source.name;shutil.copyfile(source, target)
        rows.append(dict(original=str(source), snapshot=str(target), bytes=target.stat().st_size))
    return rows


def _namespace():
    """Clone ordinary trusted driver functions only; imported contexts stay intact."""
    from . import heir_vgg_fused_pipeline as original
    namespace = dict(vars(original))
    for name, fn in vars(original).items():
        if isinstance(fn, FunctionType) and fn.__globals__ is vars(original):
            code = fn.__code__
            if name == 'execute':
                labels = {'9600s full-case allowance exhausted': '10800s full-case allowance exhausted',
                          'complete-fused-adapter-job': 'complete-wide-pool-adapter-job'}
                code = code.replace(co_consts=tuple(labels.get(value, value) if isinstance(value, str) else value
                                                    for value in code.co_consts))
            cloned = FunctionType(code, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
            cloned.__kwdefaults__ = fn.__kwdefaults__;namespace[name] = cloned
    namespace.update(MODEL=MODEL, VARIANT=VARIANT, ROUND=ROUND, MODE=MODE, MODULE=MODULE,
        TOTAL_SECONDS=TOTAL_SECONDS, WORKER_SECONDS=WORKER_SECONDS, MEMORY_GIB_CHOICES=MEMORY_GIB_CHOICES,
        COUNTERS=COUNTERS, source_gate=source_gate, worker=worker, check_worker=check_worker,
        validate_memory_gib=validate_memory_gib, _snapshot=_snapshot)
    return namespace


def execute(args):
    return _namespace()['execute'](args)


def parser():
    from .heir_vgg_fused_pipeline import SOURCE_FIELDS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True);p.add_argument('--worker', action='store_true')
    p.add_argument('--run', action='store_true')
    for name in SOURCE_FIELDS:p.add_argument('--'+name.replace('_', '-'), type=Path)
    for name in ('python', 'heir-opt', 'registry'):p.add_argument('--'+name, type=Path)
    p.add_argument('--memory-gib', type=int, choices=MEMORY_GIB_CHOICES, default=24)
    return p


def main(argv=None):
    from .heir_vgg_fused_pipeline import SOURCE_FIELDS, GIB
    p = parser();args = p.parse_args(argv)
    if args.worker:
        if args.run or args.registry or any(getattr(args, k) is None for k in SOURCE_FIELDS):
            p.error('Pure worker requires all explicit source paths and has no registry/run options')
        resource.setrlimit(resource.RLIMIT_AS, (24*GIB, 24*GIB));resource.setrlimit(resource.RLIMIT_FSIZE, (16*GIB, 16*GIB))
        signal.alarm(WORKER_SECONDS)
        try:result = worker(args)
        finally:signal.alarm(0)
    else:
        from .. import paths
        args.python = args.python or paths.INTAKE_PYTHON
        args.heir_opt = args.heir_opt or paths.WORKSPACE/'baselines/heir-2026.11.1-venv/bin/heir-opt'
        result = execute(args)
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'strict_math_passed', 'model_compiled')}))
    return 0 if result['status'] in ('prepared', 'ok') else 1


if __name__ == '__main__':raise SystemExit(main())
