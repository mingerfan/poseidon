"""Deterministic public arrays and fake drivers; no real model or compiler."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stderr
import copy
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_alexnet_wide_pool as entry


def numpy_conv(np, image, weight, *, stride, padding):
    """Independent ordinary convolution; no adapter-specific index shortcut."""
    ph, pw = padding;sh, sw = stride;kh, kw = weight.shape[2:]
    padded = np.pad(image, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    return np.einsum('nchwkl,ockl->nohw', windows, weight, optimize=False)


def independent_pair(np, image, weight, bias, padding):
    padded = np.pad(image, ((0, 0), (0, 0), (1, 1), (1, 1)))
    h, w = image.shape[2]//2, image.shape[3]//2
    pooled = np.empty((1, image.shape[1], h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):pooled[:, :, i, j] = padded[:, :, 2*i:2*i+3, 2*j:2*j+3].sum((2, 3))/9
    result = numpy_conv(np, pooled, weight, stride=(1, 1), padding=(padding, padding))
    return result+bias[None, :, None, None]


def metadata_graph(tiny=False, np=None):
    """Actual architecture geometry, metadata only; no weight values."""
    tensors = {'x': NS(name='x', shape=(1, 3, 32, 32), kind='input')}
    nodes, weights, previous, shape = [], {}, 'x', (1, 3, 32, 32)
    convs = {0: (96, 3, 1, 'Conv2d_1'), 4: (256, 5, 2, 'Conv2d_2'),
             8: (384, 3, 1, 'Conv2d_3'), 11: (384, 3, 1, 'Conv2d_4'), 14: (256, 3, 1, 'Conv2d_5')}
    if tiny:convs = {key: (1, kernel, padding, name) for key, (_, kernel, padding, name) in convs.items()}
    def parameter(name, dims):
        tensors[name] = NS(name=name, shape=tuple(dims), kind='const')
        if np is None:weights[name] = NS(shape=tuple(dims))
        else:
            value = np.full(dims, 1. if name.endswith('var') or name.endswith('weight') and len(dims) == 1 else .125, dtype=np.float64)
            value.flags.writeable = False;weights[name] = value
    for index in range(24):
        name = 'n'+str(index);attrs = {}
        if index in convs:
            op = 'conv2d';out, kernel, padding, prefix = convs[index]
            attrs = dict(weight=prefix+'.weight', bias=prefix+'.bias', stride=[1, 1], padding=[padding]*2,
                         dilation=[1, 1], groups=1)
            parameter(attrs['weight'], (out, shape[1], kernel, kernel));parameter(attrs['bias'], (out,))
            shape = (1, out, *shape[2:])
        elif index in (1, 5, 9, 12, 15):
            op = 'batchnorm'
            for key in ('weight', 'bias', 'mean', 'var'):
                attrs[key] = name+'.'+key;parameter(attrs[key], (shape[1],))
            attrs['eps'] = 1e-5
        elif index in (3, 7, 17):
            op = 'avgpool';attrs = dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1],
                                       ceil_mode=False, count_include_pad=True)
            shape = (1, shape[1], shape[2]//2, shape[3]//2)
        elif index == 18:
            op = 'reshape';shape = (1, shape[1]*shape[2]*shape[3]);attrs = dict(shape=list(shape))
        elif index in (19, 21, 23):
            op = 'linear';out = 10 if index == 23 else (1 if tiny else 32)
            attrs = dict(weight=name+'.weight', bias=name+'.bias')
            parameter(attrs['weight'], (out, shape[1]));parameter(attrs['bias'], (out,));shape = (1, out)
        else:op = 'act';attrs = dict(fn='silu')
        output = name+'.out';tensors[output] = NS(name=output, shape=shape, kind='activation')
        nodes.append(NS(id=name, op=op, inputs=[previous], outputs=[output], attrs=attrs, src='synthetic-metadata'))
        previous = output
    return NS(name=entry.MODEL, nodes=nodes, tensors=tensors, weights=weights, inputs=['x'], outputs=[previous])


def tiny_export_graph(np):
    """Two real adapter pairs, reduced channels and dimensions for generated MLIR."""
    tensors = {};weights = {};nodes = []
    def tensor(name, shape, kind='activation'):tensors[name] = NS(name=name, shape=shape, kind=kind)
    tensor('x', (1, 2, 8, 8), 'input');previous = 'x';shape = (1, 2, 8, 8);patched = []
    for pool_id, conv_id, out_channels, prefix in (('n3', 'n4', 3, 'Conv2d_2'), ('n7', 'n8', 2, 'Conv2d_3')):
        pooled = (1, shape[1], shape[2]//2, shape[3]//2);after = (1, out_channels, *pooled[2:])
        attrs = dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1])
        poolout = pool_id+'.out';tensor(poolout, pooled)
        nodes.append(NS(id=pool_id, op='avgpool', inputs=[previous], outputs=[poolout], attrs=attrs))
        w, b = prefix+'.weight', prefix+'.bias'
        weight = np.arange(out_channels*shape[1]*9, dtype=np.float64).reshape(out_channels, shape[1], 3, 3)/1024
        bias = np.arange(out_channels, dtype=np.float64)/16+.25
        for name, value in ((w, weight), (b, bias)):
            value.flags.writeable = False;weights[name] = value;tensor(name, value.shape, 'const')
        output = conv_id+'.out';tensor(output, after)
        nodes.append(NS(id=conv_id, op='conv2d', inputs=[poolout], outputs=[output],
            attrs=dict(weight=w, bias=b, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1)))
        patched.append((pool_id, conv_id, shape, pooled, after, w, 3, 1));previous = output;shape = after
    tensor('flat', (1, int(np.prod(shape[1:]))))
    nodes.append(NS(id='flat', op='reshape', inputs=[previous], outputs=['flat'], attrs=dict(shape=list(tensors['flat'].shape))))
    return NS(name=entry.MODEL, nodes=nodes, tensors=tensors, weights=weights, inputs=['x'], outputs=['flat']), tuple(patched)


class WidePoolMathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):os.environ[key] = '1'
        import numpy as np
        cls.np = np

    def compare(self, shape, out_channels, kernel):
        np = self.np;c = shape[1];p = kernel//2
        image = ((np.arange(np.prod(shape), dtype=np.float64).reshape(shape)%23)-11)/64
        wshape = (out_channels, c, kernel, kernel)
        weight = ((np.arange(np.prod(wshape), dtype=np.float64).reshape(wshape)%17)-8)/8192
        bias = (np.arange(out_channels, dtype=np.float64)%11-5)/16
        image.flags.writeable = False;weight.flags.writeable = False;bias.flags.writeable = False
        original = image.copy(), weight.copy(), bias.copy()
        pooled = (1, c, shape[2]//2, shape[3]//2);output = (1, out_channels, *pooled[2:])
        spec = entry.pair_spec(shape, dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1]), pooled,
            dict(stride=[1, 1], padding=[p, p], dilation=[1, 1], groups=1), wshape, output)
        observed = entry.execute_pair(spec, image, weight, bias, lambda *a, **kw: numpy_conv(np, *a, **kw))
        wanted = independent_pair(np, image, weight, bias, p)
        ratio = np.abs(observed-wanted)/(1e-12+1e-12*np.abs(wanted))
        self.assertLessEqual(float(ratio.max()), 1.)
        self.assertTrue(np.all(np.isfinite(observed)))
        for a, b in zip((image, weight, bias), original):np.testing.assert_array_equal(a, b)
        return spec, observed

    def test_first_actual_geometry_all_channels_outputs_and_nonzero_edges(self):
        spec, result = self.compare((1, 96, 32, 32), 256, 5)
        self.assertEqual(spec['wide_shape'], [1, 96, 20, 20]);self.assertEqual(spec['wide_padding'], 5)
        self.assertEqual(result.shape, (1, 256, 16, 16));self.assertNotEqual(float(result[0, 0, 0, 0]), 0.)

    def test_second_actual_geometry_all_channels_outputs_and_overlap(self):
        spec, result = self.compare((1, 256, 16, 16), 384, 3)
        self.assertEqual(spec['wide_shape'], [1, 256, 10, 10]);self.assertEqual(spec['wide_padding'], 3)
        self.assertEqual(result.shape, (1, 384, 8, 8))

    def test_mask_is_exact_readonly_binary_interior_and_divisor9(self):
        np = self.np;spec, _ = self.compare((1, 2, 8, 8), 3, 5)
        diagonal, mask = entry.pair_constants(spec)
        self.assertEqual(set(np.unique(mask)), {0., 1.});self.assertFalse(mask.flags.writeable)
        self.assertFalse(diagonal.flags.writeable);self.assertEqual(float(mask.sum()), 2*4*4)
        self.assertEqual(float(diagonal[0, 0, 0, 0]), 1/9);self.assertEqual(float(diagonal[0, 1, 0, 0]), 0.)

    def test_bias_is_added_exactly_once_and_zero_mask_removes_partial_border_windows(self):
        np = self.np;shape = (1, 2, 4, 4)
        spec = entry.pair_spec(shape, dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1]), (1, 2, 2, 2),
            dict(stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1), (3, 2, 3, 3), (1, 3, 2, 2))
        x = np.ones(shape, dtype=np.float64);w = np.zeros((3, 2, 3, 3), dtype=np.float64);bias = np.array([.25, -.5, 2.])
        observed = entry.execute_pair(spec, x, w, bias, lambda *a, **kw: numpy_conv(np, *a, **kw))
        np.testing.assert_array_equal(observed, np.broadcast_to(bias[None, :, None, None], observed.shape))
        diagonal, mask = entry.pair_constants(spec)
        wide = numpy_conv(np, x, diagonal, stride=(2, 2), padding=(3, 3))
        self.assertGreater(float(wide[0, 0, -1, -1]), 0.);self.assertEqual(float((wide*mask)[0, 0, -1, -1]), 0.)

    def test_geometry_fanout_and_original_attributes_reject(self):
        graph = metadata_graph();pairs = entry.find_pairs(graph)
        self.assertEqual(len(pairs), 2);self.assertEqual(sum(p['mask_bytes'] for p in pairs), 512000)
        for drift in ('fanout', 'odd', 'ceil', 'pad', 'stride', 'channels', 'bias'):
            changed = copy.deepcopy(graph)
            if drift == 'fanout':changed.nodes[-1].inputs.append(changed.nodes[3].outputs[0])
            elif drift == 'odd':changed.tensors[changed.nodes[3].inputs[0]].shape = (1, 96, 31, 32)
            elif drift == 'ceil':changed.nodes[3].attrs['ceil_mode'] = True
            elif drift == 'pad':changed.nodes[3].attrs['padding'] = [0, 0]
            elif drift == 'stride':changed.nodes[4].attrs['stride'] = [2, 2]
            elif drift == 'channels':changed.tensors['Conv2d_2.weight'].shape = (256, 95, 5, 5)
            else:changed.nodes[4].attrs['bias'] = 'foreign.bias'
            with self.subTest(drift=drift), self.assertRaises(ValueError):entry.find_pairs(changed)

    def test_nonfinite_float32_and_wrong_weight_rejected(self):
        np = self.np;shape = (1, 1, 4, 4)
        spec = entry.pair_spec(shape, dict(kernel=[3, 3], stride=[2, 2], padding=[1, 1]), (1, 1, 2, 2),
            dict(stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1), (1, 1, 3, 3), (1, 1, 2, 2))
        for dtype, scalar, wshape in ((np.float32, 1., (1, 1, 3, 3)), (np.float64, np.nan, (1, 1, 3, 3)),
                                     (np.float64, 1., (1, 1, 1, 1))):
            with self.assertRaises(ValueError):entry.execute_pair(spec, np.full(shape, scalar, dtype=dtype),
                np.zeros(wshape, dtype=np.float64), None, lambda *a, **kw: self.fail('Rejected input reached convolution'))

    def test_actual_generator_consumes_pair_structure_and_never_emits_placeholder_or_postpool_pad(self):
        from pt_agent.experiments import heir_cnn_export as original
        np = self.np;graph, pairs = tiny_export_graph(np)
        before = [(n.id, copy.deepcopy(n.attrs)) for n in graph.nodes];weight_copies = {k: v.copy() for k, v in graph.weights.items()}
        program = NS(params=[NS(shape=(1, 2, 8, 8))])
        original_emitter = original.Emitter
        with patch.object(entry, 'PAIRS', pairs), patch.object(original, 'saved_activations', return_value=(program, [])):
            text, metadata = entry.export_graph(graph, 'synthetic parsed fixture')
        self.assertIs(original.Emitter, original_emitter);self.assertNotIn('deferred_pool', text)
        self.assertEqual(text.count('linalg.conv_2d_nchw_fchw'), 4)
        self.assertEqual(text.count('tensor.pad'), 2);self.assertEqual(text.count('arith.mulf %s0, %s1'), 2)
        self.assertEqual(len(metadata['mask_constants']), 2)
        self.assertEqual(metadata['wide_pool_pairs'][0]['wide_padding'], 3)
        self.assertEqual([(n.id, n.attrs) for n in graph.nodes], before)
        for k, value in graph.weights.items():np.testing.assert_array_equal(value, weight_copies[k])


class WidePoolDriverChecks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory();self.addCleanup(temp.cleanup);self.root = Path(temp.name)
        self.graph = metadata_graph()

    def source_fixture(self):
        from pt_agent.experiments.heir_vgg_fused_pipeline import SOURCE_FIELDS
        base = self.root/'source';base.mkdir();source = {k: base/k for k in SOURCE_FIELDS}
        info = base/'info';source['graph_dir'] = info/'graph';source['graph_dir'].mkdir(parents=True)
        graph = dict(name=self.graph.name, inputs=self.graph.inputs, outputs=self.graph.outputs,
            nodes=[[n.id, n.op, n.inputs, n.outputs, n.attrs, n.src] for n in self.graph.nodes],
            tensors=[[n, t.shape, t.kind] for n, t in self.graph.tensors.items()])
        (source['graph_dir']/'graph.json').write_text(json.dumps(graph));source['program'].write_text('saved fixed63 fixture')
        source['probes'].write_bytes(b'only-stat-synthetic')
        source['probe_source'].write_text(json.dumps(dict(split='train', indices=[0, 1, 2], sample_ids=['train:0', 'train:1', 'train:2'])))
        source['intake_result'].write_text(json.dumps(dict(status='ok', model='pt_agent.experiments.r13_common_models:'+entry.MODEL,
            worker=dict(artifacts=dict(model_info=str(info), training_probes=str(source['probes']), training_probe_source=str(source['probe_source']))))))
        source['frontend_result'].write_text(json.dumps(dict(status='ok', round='R17a', model='pt_agent.experiments.r13_common_models:'+entry.MODEL,
            worker=dict(status='ok', model='pt_agent.experiments.r13_common_models:'+entry.MODEL, heldout_acceptance=True,
                completed_samples=1000, math_probes_passed=True, artifacts=dict(program=str(source['program'])),
                rule_options=dict(deg=63, margin=1.5, slots=16384, placement='depth-dp', refresh='compiler', act_method='cheb')))))
        self.parameters = dict(logN=16, Q=list(range(11)), P=[31, 37, 41], logDefaultScale=45)
        source['native_parameters'].write_text(json.dumps(self.parameters))
        source['pipeline'].write_text('--pass-pipeline=builtin.module(secret-insert-mgmt-ckks{after-mul=true parameter-file='+str(source['native_parameters'])+'},scheme-to-lattigo{parameter-file='+str(source['native_parameters'])+'})\n')
        return source

    def export_record(self, checked):
        from pt_agent.experiments.heir_common_models import avgpool_spec
        original = checked['contract'];pairs = checked['wide_pool_pairs']
        return dict(common_model=entry.MODEL, common_model_contract=original, function=entry.MODEL,
            graph_nodes=24, input_shape=[1, 3, 32, 32], output_shape=[1, 10],
            approximations=[dict(graph_node=n['id']) for n in original['nodes'] if n['op'] == 'act'],
            wide_pool_pairs=pairs, mask_constants=[dict(name=p['mask_name'], ssa='%mask'+p['pool_id'], shape=p['wide_shape']) for p in pairs],
            fixed_local_avgpool_lowerings=[dict(avgpool_spec(n['input_shapes'][0], n['attrs'], n['output_shapes'][0]),
                original_node=n['id'], input=n['inputs'][0], output=n['outputs'][0]) for n in original['nodes'] if n['op'] == 'avgpool'],
            exact_spatial_mean_lowerings=[])

    def successful_worker(self, checked):
        from pt_agent.experiments.heir_common_models import expected_constant_names
        exported = self.export_record(checked);names = []
        for name in expected_constant_names(checked['contract'], exported):
            names.append(name)
            if name in ('derived-exact-local-avgpool:n3', 'derived-exact-local-avgpool:n7'):
                names.append('derived-wide-pool-mask:'+name.split(':')[-1])
        record = entry.initial_worker();representation = entry.representation_contract(checked['contract'], checked['wide_pool_pairs'])
        representation['binary64_math_passed'] = True
        record.update(status='ok', source=checked, original_model_contract=checked['contract'],
            lowered_adapter_contract=representation, export=exported, strict_math_passed=True, activation_pairs=21, logit_elements=30,
            constants=[dict(name=name, passed=True, numerically_equal=True) for name in names],
            probes=[dict(sample_id='train:'+str(i), reference=[[float(k) for k in range(10)]], observed=[[float(k) for k in range(10)]],
                activation_checks=[dict(passed=True)]*7, logit_check=dict(passed=True)) for i in range(3)])
        for k in entry.COUNTERS:record['attempted'][k] = record['completed'][k] = 3 if k.endswith('forward') else 1
        return record

    def test_source_gate_only_json_binds_intake_paths_and_exact_original_pairs(self):
        source = self.source_fixture();checked = entry.source_gate(source)
        self.assertEqual(checked['contract']['graph_nodes'], 24);self.assertEqual(len(checked['wide_pool_pairs']), 2)
        self.assertFalse(checked['source_payloads_read'])
        foreign = self.root/'foreign.npz';foreign.write_bytes(b'fake')
        with self.assertRaisesRegex(ValueError, 'belong'):entry.source_gate(dict(source, probes=foreign))
        metadata = json.loads((source['graph_dir']/'graph.json').read_text());metadata['nodes'][4][4]['padding'] = [0, 0]
        (source['graph_dir']/'graph.json').write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):entry.source_gate(source)

    def test_namespace_clones_ordinary_functions_preserving_original_contextmanagers_and_globals(self):
        from pt_agent.experiments import heir_vgg_fused_pipeline as old
        namespace = entry._namespace()
        self.assertIs(namespace['serial_lock'], old.serial_lock)
        self.assertIs(namespace['execute'].__globals__, namespace)
        self.assertEqual(namespace['MODEL'], entry.MODEL);self.assertEqual(old.MODEL, 'vgg16_silu_avgpool')
        self.assertEqual(namespace['TOTAL_SECONDS'], 10800);self.assertEqual(old.TOTAL_SECONDS, 9600)

    def test_pure_worker_argv_explicit_paths_AS24_and_original_timing(self):
        source = self.source_fixture();namespace = entry._namespace()
        command = namespace['worker_command'](source, self.root/'worker', Path('/fixture/python'), 24)
        self.assertEqual(command[command.index('--memory-gib')+1], '24')
        self.assertEqual(command[command.index('--seconds')+1], '1860')
        self.assertEqual(command[command.index('--')+2], entry.MODULE)
        for name, path in source.items():self.assertEqual(command[command.index('--'+name.replace('_', '-'))+1], str(path))
        self.assertNotIn('--registry', command)

    def test_memory_and_cli_reject_other_allowances_and_missing_worker_paths(self):
        self.assertEqual(entry.validate_memory_gib(24), 24)
        for value in (True, False, 12, 24., float('nan'), 25):
            with self.assertRaises(ValueError):entry.validate_memory_gib(value)
        for argv in (['--out', 'x', '--memory-gib', '12'], ['--worker', '--out', str(self.root/'missing')]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):entry.main(argv)
        self.assertFalse((self.root/'missing').exists())

    def test_check_worker_requires_all31_constants_21_activations_30_logits_and_zero_excluded(self):
        source = self.source_fixture();checked = entry.source_gate(source);record = self.successful_worker(checked)
        self.assertEqual(entry.check_worker(record, checked)['emitted_constants'], 31)
        for drift in ('mask', 'source', 'constant', 'count', 'zero', 'formal', 'strict', 'logit'):
            changed = copy.deepcopy(record)
            if drift == 'mask':changed['export']['wide_pool_pairs'][0]['wide_padding'] = 4
            elif drift == 'source':changed['source'] = {}
            elif drift == 'constant':changed['constants'].pop()
            elif drift == 'count':changed['completed']['adapter_forward'] = 2
            elif drift == 'zero':changed['report_samples'] = 1
            elif drift == 'formal':changed['accuracy_acceptance'] = True
            elif drift == 'strict':changed['probes'][0]['activation_checks'][0]['passed'] = False
            else:changed['probes'][0]['observed'][0][0] += .01
            with self.subTest(drift=drift), self.assertRaises(ValueError):entry.check_worker(changed, checked)

    def test_real_small_emitter_checks_all31_tensor_constants_and_rejects_mutated_mask(self):
        import numpy as np
        from pt_agent.experiments import heir_cnn_export as exporter
        graph = metadata_graph(tiny=True, np=np)
        pairs = tuple((pool_id, conv_id, tuple(graph.tensors[graph.nodes[int(pool_id[1:])].inputs[0]].shape),
            tuple(graph.tensors[graph.nodes[int(pool_id[1:])].outputs[0]].shape),
            tuple(graph.tensors[graph.nodes[int(conv_id[1:])].outputs[0]].shape), name, kernel, padding)
            for pool_id, conv_id, _, _, _, name, kernel, padding in entry.PAIRS)
        program = NS(params=[NS(shape=(1, 3, 32, 32))]);configs = [dict(fn='silu', method='cheb', deg=63, domain=[-2., 2.]) for _ in range(7)]
        # Polynomial fitting is excluded; this fixture exercises the real pool,
        # convolution, batchnorm, linear, mask emitter and tensor literal checker.
        def activation(emitter, value, config):return value, dict(config, coefficients=[], scalar_operations=0)
        with patch.object(entry, 'PAIRS', pairs), patch.object(exporter, 'saved_activations', return_value=(program, configs)), \
                patch.object(exporter.Emitter, 'activation', activation):
            text, exported = entry.export_graph(graph, 'synthetic parsed fixture')
            rows = entry.check_constants(graph, text, exported)
            self.assertEqual(len(rows), 31);self.assertTrue(all(r['numerically_equal'] for r in rows))
            first = exported['mask_constants'][0]['ssa']+' = arith.constant dense<'
            offset = text.index(first)+len(first);end = text.index('> : tensor<', offset)
            literal = text[offset:end];changed = literal.replace('1.00000000000000000e+00', '5.00000000000000000e-01', 1)
            self.assertNotEqual(changed, literal)
            with self.assertRaisesRegex(ValueError, 'mask'):
                entry.check_constants(graph, text[:offset]+changed+text[end:], exported)

    def test_full_worker_uses_original_weights_and_three_saved_inputs_once_then_persists_counts(self):
        import numpy as np
        from pt_agent.experiments.heir_cnn_math_check import check_math
        from pt_agent.experiments.heir_common_models import graph_contract
        source = self.source_fixture();checked = entry.source_gate(source);good = self.successful_worker(checked)
        with source['probes'].open('wb') as stream:
            np.savez(stream, input=np.zeros((3, 3, 32, 32), dtype=np.float64), indices=np.array([0, 1, 2]))
        calls = []
        def reference(program, weights, image, cache, observer):
            calls.append(weights is self.graph.weights)
            for _ in range(7):observer('synthetic', np.array([.125], dtype=np.float64), {})
            return np.arange(10, dtype=np.float64).reshape(1, 10)
        api = dict(np=np, load_graph=lambda path: self.graph, graph_contract=graph_contract,
            saved_activations=lambda text: (object(), [{}]*7),
            export_graph=lambda *a: ('trusted synthetic MLIR\n', copy.deepcopy(good['export'])),
            check_constants=lambda *a: copy.deepcopy(good['constants']), run_tensor=reference, check_math=check_math,
            evaluate_graph=lambda *a: (np.arange(10, dtype=np.float64).reshape(1, 10), [np.array([.125], dtype=np.float64)]*7))
        args = Namespace(out=self.root/'worker', **source)
        with patch.object(entry, '_runtime', return_value=api):result = entry.worker(args)
        self.assertEqual(result['status'], 'ok');self.assertEqual(calls, [True]*3)
        self.assertTrue(entry.check_worker(json.loads((args.out/'result.json').read_text()), checked)['passed'])
        self.assertFalse((self.root/'registry.jsonl').exists())

    def test_worker_first_strict_failure_preserves_one_probe_and_no_later_forwards(self):
        import numpy as np
        from pt_agent.experiments.heir_cnn_math_check import check_math
        from pt_agent.experiments.heir_common_models import graph_contract
        source = self.source_fixture();checked = entry.source_gate(source);good = self.successful_worker(checked)
        with source['probes'].open('wb') as stream:
            np.savez(stream, input=np.zeros((3, 3, 32, 32), dtype=np.float64), indices=np.array([0, 1, 2]))
        def reference(program, weights, image, cache, observer):
            for _ in range(7):observer('synthetic', np.array([.125], dtype=np.float64), {})
            return np.zeros((1, 10), dtype=np.float64)
        api = dict(np=np, load_graph=lambda path: self.graph, graph_contract=graph_contract,
            saved_activations=lambda text: (object(), [{}]*7),
            export_graph=lambda *a: ('trusted synthetic MLIR\n', copy.deepcopy(good['export'])),
            check_constants=lambda *a: copy.deepcopy(good['constants']), run_tensor=reference, check_math=check_math,
            evaluate_graph=lambda *a: (np.ones((1, 10), dtype=np.float64), [np.array([.125], dtype=np.float64)]*7))
        with patch.object(entry, '_runtime', return_value=api):result = entry.worker(Namespace(out=self.root/'worker', **source))
        self.assertEqual(result['status'], 'error');self.assertIsNone(result['strict_math_passed'])
        self.assertEqual(len(result['probes']), 1);self.assertEqual(result['completed']['reference_forward'], 1)
        self.assertEqual(result['completed']['adapter_forward'], 1);self.assertFalse(result['probes'][0]['logit_check']['passed'])

    def run_outer(self, failure=None, run=True):
        from pt_agent import paths
        from pt_agent.experiments import heir_common_model_pipeline as common
        source = self.source_fixture();checked = entry.source_gate(source);namespace = entry._namespace()
        tool = self.root/'heir-opt';tool.write_text('fake executable');tool.chmod(0o700)
        args = Namespace(out=self.root/'case', run=run, python=Path('/fixture/python'), heir_opt=tool,
                         registry=self.root/'registry.jsonl', memory_gib=24, **{k: None for k in source})
        calls = []
        def process(command, **options):
            calls.append((command, options));Path(options['log']).write_text('fake stage\n');Path(options['resources']).write_text('')
            if '--worker' in command:
                directory = Path(command[command.index('--out')+1]);directory.mkdir()
                worker = self.successful_worker(checked)
                if failure == 'math':worker['probes'][0]['logit_check']['passed'] = False
                (directory/'result.json').write_text(json.dumps(worker)+'\n');(directory/'model.mlir').write_text('fake trusted adapter\n')
            else:
                target = Path(command[command.index('-o')+1])
                scale = 44 if failure == 'parameters' else 45
                target.write_text('paramsLiteral = <logN = 16, Q = ['+', '.join(map(str, self.parameters['Q']))+'], P = [31, 37, 41], logDefaultScale = '+str(scale)+'>\n%a = lattigo.ckks.bootstrap %b\n')
            return dict(exit_code=1 if failure == 'process' and '--worker' not in command else 0,
                timed_out=failure == 'timeout' and '--worker' not in command, signal=None, signals_sent=[], remaining_pids=[])
        namespace.update(_snapshot=lambda out: [], run_process=process, _available_ram_bytes=lambda: 16*1024**3)
        with patch.object(paths, 'RESULTS_DIR', self.root), patch.object(common, 'sources', return_value=source), \
                patch.object(entry.shutil, 'disk_usage', return_value=NS(free=100*1024**3)):
            result = namespace['execute'](args)
        return result, calls

    def test_prepared_driver_starts_no_process_and_preserves_pure_explicit_command(self):
        result, calls = self.run_outer(run=False)
        self.assertEqual(result['status'], 'prepared');self.assertEqual(calls, [])
        self.assertEqual(result['attempted'], dict(export_math=0, native_compile=0))
        self.assertEqual(result['limits']['AS_gib'], 24);self.assertEqual(result['limits']['total_seconds'], 10800)

    def test_fake_full_driver_compiles_only_after_math_and_records_unique_registry_completion(self):
        result, calls = self.run_outer()
        self.assertEqual(result['status'], 'ok');self.assertTrue(result['model_compiled'])
        self.assertEqual(len(calls), 2);self.assertEqual([options['mem_gib'] for _, options in calls], [24, 24])
        self.assertEqual([options['fsize_gib'] for _, options in calls], [16, 16])
        self.assertLessEqual(calls[0][1]['timeout'], 1920);self.assertLessEqual(calls[1][1]['timeout'], 7200)
        self.assertEqual(result['native_request']['parameters'], self.parameters)
        registry = [json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry), 1);self.assertEqual(set(registry[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        completion = json.loads((self.root/'case/completion.json').read_text())
        self.assertEqual(completion['status'], 'ok');self.assertTrue(completion['registry_persisted'])

    def test_math_failure_stops_before_compiler_and_retains_partial_worker(self):
        result, calls = self.run_outer(failure='math')
        self.assertEqual(result['status'], 'error');self.assertEqual(len(calls), 1)
        self.assertFalse(result['model_compiled']);self.assertEqual(result['attempted']['native_compile'], 0)
        self.assertFalse(result['worker']['probes'][0]['logit_check']['passed'])

    def test_native_parameter_drift_and_timeout_preserve_separate_failures(self):
        result, calls = self.run_outer(failure='parameters')
        self.assertEqual(result['status'], 'error');self.assertTrue(result['strict_math_passed'])
        self.assertIn('parameter', result['error']);self.assertEqual(result['completed']['native_compile'], 0)
        # Each subcase has a distinct isolated source/result directory.
        oldroot = self.root;self.root = oldroot/'timeout';self.root.mkdir()
        result, calls = self.run_outer(failure='timeout')
        self.assertEqual(result['status'], 'timeout');self.assertTrue(result['wall_timeout_observed'])
        self.assertTrue(result['phases']['native_compile']['observation']['timed_out'])


if __name__ == '__main__':unittest.main()
