"""Small deterministic fixtures and fake serial processes; no real model/tool."""
from argparse import Namespace
import copy
from contextlib import ExitStack, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_vgg_fused_pipeline as entry
from pt_agent.experiments import heir_common_model_pipeline as original
from pt_agent.experiments.heir_common_models import graph_contract, expected_constant_names
from pt_agent.experiments.r10_paper_metrics import StageRecorder


def fixture(np):
    """Full50-node topology with tiny channels, synthetic readonly parameters."""
    pools = {'n6', 'n13', 'n23', 'n33'}
    convs = {'n'+str(i) for i in (0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40)}
    bns = {'n'+str(i) for i in (1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41, 45)}
    linears = {'n44', 'n47', 'n49'}
    mandatory = {conv: name for _, conv, name in entry.fusion_helper.PAIRS}
    tensors = {'x': NS(name='x', shape=(1, 3, 32, 32), kind='input')}
    nodes, weights, previous, shape = [], {}, 'x', (1, 3, 32, 32)
    def weight(name, dimensions, value=0.):
        array = np.full(dimensions, value, dtype=np.float64);array.flags.writeable = False
        weights[name] = array;tensors[name] = NS(name=name, shape=tuple(dimensions), kind='const')
    for index in range(50):
        name, attrs = 'n'+str(index), {}
        if name in convs:
            op = 'conv2d';w = mandatory.get(name, name+'.weight');b = w.removesuffix('weight')+'bias'
            weight(w, (1, shape[1], 3, 3), .125);weight(b, (1,), .25)
            attrs = dict(weight=w, bias=b, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1)
            shape = (1, 1, *shape[2:])
        elif name in bns:
            op = 'batchnorm'
            for key, value in (('weight', 1.), ('bias', 0.), ('mean', 0.), ('var', 1.)):
                attrs[key] = name+'.'+key;weight(attrs[key], (shape[1],), value)
            attrs['eps'] = 1e-5
        elif name in pools:
            op = 'avgpool';attrs = dict(kernel=(2, 2), stride=(2, 2), padding=(0, 0))
            shape = (1, shape[1], shape[2]//2, shape[3]//2)
        elif name == 'n43':op = 'mean';attrs = dict(axes=(2, 3));shape = shape[:2]
        elif name in linears:
            op = 'linear';out = 10 if name == 'n49' else 1
            attrs = dict(weight=name+'.weight', bias=name+'.bias')
            weight(attrs['weight'], (out, shape[1]), .125);weight(attrs['bias'], (out,), .25)
            shape = (1, out)
        else:op = 'act';attrs = dict(fn='silu')
        output = name+'.out';tensors[output] = NS(name=output, shape=shape, kind='activation')
        nodes.append(NS(id=name, op=op, inputs=[previous], outputs=[output], attrs=attrs, src='synthetic'))
        previous = output
    return NS(name=entry.MODEL, nodes=nodes, tensors=tensors, weights=weights, inputs=['x'], outputs=[previous])


def export_record(contract):
    mean = next(node for node in contract['nodes'] if node['op'] == 'mean')
    shape = mean['input_shapes'][0]
    return dict(common_model=entry.MODEL, common_model_contract=contract, function=entry.MODEL,
        graph_nodes=46, input_shape=contract['input_shape'], output_shape=contract['output_shape'],
        approximations=[dict(graph_node=n['id']) for n in contract['nodes'] if n['op'] == 'act'],
        exact_spatial_mean_matvec=True, fixed_local_avgpool_lowerings=[],
        exact_spatial_mean_lowerings=[dict(original_node=mean['id'], input=mean['inputs'][0], output=mean['outputs'][0],
            input_shape=shape, output_shape=shape[:2], axes=[2, 3], coefficient=1/(shape[2]*shape[3]),
            matrix_shape=[shape[1], shape[1]*shape[2]*shape[3]], nonzeros=shape[1]*shape[2]*shape[3],
            trainable_weights_changed=False, bias=False,
            formula='W[c,c*H*W+p]=1/(H*W), p=0..H*W-1; all other coefficients0')])


class FusedPipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        cls.np = np

    def setUp(self):
        temporary = TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graph = fixture(self.np)
        self.fused, self.fusion = entry.fusion_helper.fuse_graph(self.graph)
        self.original = graph_contract(self.graph, entry.MODEL)
        self.lowered = graph_contract(self.fused, entry.MODEL)
        self.export = export_record(self.lowered)

    def source_fixture(self):
        base = self.root/'source';base.mkdir()
        source = {name: base/name for name in entry.SOURCE_FIELDS}
        info = base/'model_info';source['graph_dir'] = info/'graph';source['graph_dir'].mkdir(parents=True)
        graph = dict(name=self.graph.name, inputs=self.graph.inputs, outputs=self.graph.outputs,
            nodes=[[n.id, n.op, n.inputs, n.outputs, n.attrs, n.src] for n in self.graph.nodes],
            tensors=[[t.name, t.shape, t.kind] for t in self.graph.tensors.values()])
        (source['graph_dir']/'graph.json').write_text(json.dumps(graph))
        source['program'].write_text('saved synthetic fixed63 source')
        source['probes'].write_bytes(b'synthetic-source-stat-only')
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

    def successful_worker(self, checked):
        record = entry.initial_worker()
        record.update(status='ok', source=checked, original_model_contract=self.original,
            lowered_adapter_contract=self.lowered, fusion=self.fusion, export=self.export,
            strict_math_passed=True, activation_pairs=45, logit_elements=30,
            constants=[dict(name=name, passed=True, numerically_equal=True) for name in expected_constant_names(self.lowered, self.export)],
            probes=[dict(sample_id='train:'+str(i), reference=[[float(x) for x in range(10)]], observed=[[float(x) for x in range(10)]],
                activation_checks=[dict(passed=True)]*15, logit_check=dict(passed=True)) for i in range(3)])
        for name in entry.COUNTERS:
            record['attempted'][name] = record['completed'][name] = 3 if name in ('reference_forward', 'adapter_forward') else 1
        return record

    def test_compound_contract_accepts_exact_delta_and_keeps_original50_separate(self):
        before = copy.deepcopy(self.original)
        result = entry.checked_fusion_contract(self.original, self.lowered, self.fusion, self.export)
        self.assertTrue(result['passed']);self.assertEqual(self.original, before)
        self.assertEqual((self.original['graph_nodes'], self.lowered['graph_nodes']), (50, 46))
        self.assertEqual(result['emitted_tensor_constants'], 61)

    def test_compound_contract_rejects_extra_node_attr_activation_or_constant_provenance_changes(self):
        for target in ('node', 'activation', 'constant', 'bias'):
            lowered, fusion, exported = copy.deepcopy(self.lowered), copy.deepcopy(self.fusion), copy.deepcopy(self.export)
            if target == 'node':lowered['nodes'][0]['attrs']['padding'] = [0, 0]
            elif target == 'activation':exported['approximations'][0]['graph_node'] = 'other'
            elif target == 'constant':fusion['pairs'][0]['constant']['coefficient'] = .5
            else:fusion['pairs'][0]['bias_added_once'] = False
            with self.subTest(target=target), self.assertRaises(ValueError):
                entry.checked_fusion_contract(self.original, lowered, fusion, exported)

    def test_source_gate_binds_original_intake_saved_frontend_and_original_parameter_file(self):
        source = self.source_fixture();checked = entry.source_gate(source)
        self.assertEqual(checked['original_model_contract'], self.original)
        self.assertFalse(checked['source_payloads_read'])
        changed = dict(source, graph_dir=self.root/'foreign');changed['graph_dir'].mkdir()
        shutil = __import__('shutil');shutil.copyfile(source['graph_dir']/'graph.json', changed['graph_dir']/'graph.json')
        with self.assertRaisesRegex(ValueError, 'belong'):entry.source_gate(changed)

    def test_worker_command_supplies_every_path_and_no_registry_or_config(self):
        source = self.source_fixture();command = entry.worker_command(source, self.root/'worker', Path('/fixture/python'))
        self.assertEqual(command[command.index('--seconds')+1], '1860')
        self.assertEqual(command[command.index('--python')+1], '/fixture/python')
        for name, path in source.items():self.assertEqual(command[command.index('--'+name.replace('_', '-'))+1], str(path))
        self.assertIn('--worker', command);self.assertNotIn('--registry', command)

    def test_native_command_uses_original_pipeline_with_preserved_parameter_copy(self):
        source = self.source_fixture();checked = entry.source_gate(source)
        target = self.root/'native'
        command = entry.native_command(Path('/fixture/heir-opt'), self.root/'model.mlir', target, checked)
        self.assertIn('--pass-pipeline=', command[2]);self.assertNotIn('--scheme-to-lattigo', command)
        self.assertIn(str(target/'native-parameters.json'), command[2]);self.assertNotIn('parameter-file='+str(source['native_parameters']), command[2])
        self.assertEqual(json.loads((target/'native-parameters.json').read_text()), self.parameters)
        self.assertEqual(command[-2:], ['--ext-const-output-dir='+str(target/'resources'), '--ext-const-runtime-load-dir='+str(target/'resources')])

    def native_text(self, scale=45):
        return 'paramsLiteral = <logN = 16, Q = ['+', '.join(map(str, self.parameters['Q']))+'], P = [31, 37, 41], logDefaultScale = '+str(scale)+'>\n%a = lattigo.ckks.bootstrap %b\n'

    def test_native_artifact_requires_full_literal_equality(self):
        source = self.source_fixture();checked = entry.source_gate(source)
        target = self.root/'native';target.mkdir()
        (target/'model-lattigo.mlir').write_text(self.native_text())
        self.assertEqual(entry.check_native(target, checked)['static_bootstrap_operations'], 1)
        (target/'model-lattigo.mlir').write_text(self.native_text(scale=44))
        with self.assertRaisesRegex(ValueError, 'parameter'):entry.check_native(target, checked)

    def probe_run(self, *, failure=False, missing=False, reference_float32=False):
        np = self.np
        from pt_agent.experiments.heir_cnn_math_check import check_math
        calls = []
        def reference(program, weights, images, cache, observer):
            calls.append(('reference', weights is self.graph.weights, cache))
            for index in range(14 if missing else 15):observer('a'+str(index), np.array([.5], dtype=np.float64), {})
            return np.arange(10, dtype=np.float32 if reference_float32 else np.float64).reshape(1, 10)
        def adapted(graph, configs, images, common_model):
            self.assertIs(graph, self.fused);calls.append(('fused', common_model))
            value = np.arange(10, dtype=np.float64).reshape(1, 10)
            if failure:value[0, 0] = .001
            return value, [np.array([.5], dtype=np.float64) for _ in range(15)]
        api = dict(run_tensor=reference, evaluate_graph=adapted, check_math=check_math)
        result = entry.initial_worker();recorder = StageRecorder(self.root/'probe-events.jsonl')
        action = lambda: entry.run_probes(api, self.graph, self.fused, object(), [], np.zeros((3, 1), dtype=np.float64),
            dict(sample_ids=['train:0', 'train:1', 'train:2']), result, recorder, lambda: None)
        return action, result, calls

    def test_probe_checks_compare_original_weights_three_times_and_all45_activations30_logits(self):
        action, result, calls = self.probe_run();action()
        self.assertEqual(result['completed']['reference_forward'], 3);self.assertEqual(result['completed']['adapter_forward'], 3)
        self.assertEqual((result['activation_pairs'], result['logit_elements']), (45, 30))
        self.assertEqual(calls[::2], [('reference', True, False)]*3)

    def test_first_strict_math_failure_retains_one_pair_and_stops_remaining_inputs(self):
        action, result, calls = self.probe_run(failure=True)
        with self.assertRaisesRegex(ValueError, 'criterion'):action()
        self.assertEqual(len(calls), 2);self.assertEqual(len(result['probes']), 1)
        self.assertFalse(result['probes'][0]['logit_check']['passed'])
        self.assertIsNone(result['strict_math_passed'])

    def test_missing_activation_rejects_before_claiming_complete_math(self):
        action, result, calls = self.probe_run(missing=True)
        with self.assertRaisesRegex(ValueError, '15 activation'):action()
        self.assertEqual(result['activation_pairs'], 0);self.assertEqual(len(calls), 2)

    def test_original_reference_float32_is_rejected_without_claiming_strict_math(self):
        action, result, calls = self.probe_run(reference_float32=True)
        with self.assertRaisesRegex(ValueError, 'retain float64'):action()
        self.assertEqual(len(calls), 2);self.assertEqual(result['activation_pairs'], 0)
        self.assertIsNone(result['strict_math_passed'])

    def test_worker_gate_requires_exact_counts_original_math_and_zero_excluded_actions(self):
        source = self.source_fixture();checked = entry.source_gate(source);record = self.successful_worker(checked)
        self.assertTrue(entry.check_worker(record, checked)['passed'])
        for drift in ('count', 'zero', 'formal', 'logit', 'source'):
            changed = copy.deepcopy(record)
            if drift == 'count':changed['completed']['reference_forward'] = 2
            elif drift == 'zero':changed['report_samples'] = 1
            elif drift == 'formal':changed['accuracy_acceptance'] = True
            elif drift == 'logit':changed['probes'][0]['observed'][0][0] += .01
            else:changed['source'] = {}
            with self.subTest(drift=drift), self.assertRaises(ValueError):entry.check_worker(changed, checked)

    def test_full_worker_orchestration_uses_one_export_probe_load_and_original_reference(self):
        np = self.np
        from pt_agent.experiments.heir_cnn_math_check import check_math
        source = self.source_fixture()
        with source['probes'].open('wb') as stream:
            np.savez(stream, input=np.zeros((3, 3, 32, 32), dtype=np.float64), indices=np.array([0, 1, 2]))
        used_weights = []
        def reference(program, weights, image, cache, observer):
            used_weights.append(weights)
            for _ in range(15):observer('synthetic', np.array([.5], dtype=np.float64), {})
            return np.arange(10, dtype=np.float64).reshape(1, 10)
        api = dict(np=np, load_graph=lambda path: self.graph, graph_contract=graph_contract,
            saved_activations=lambda text: (object(), [{}]*15),
            export_graph=lambda *a, **kw: ('synthetic trusted MLIR\n', copy.deepcopy(self.export)),
            check_constants=lambda *a, **kw: [dict(name=name, passed=True, numerically_equal=True)
                for name in expected_constant_names(self.lowered, self.export)],
            run_tensor=reference, check_math=check_math,
            evaluate_graph=lambda *a, **kw: (np.arange(10, dtype=np.float64).reshape(1, 10),
                [np.array([.5], dtype=np.float64) for _ in range(15)]))
        args = Namespace(out=self.root/'worker', **source)
        with patch.object(entry, '_runtime', return_value=api):record = entry.worker(args)
        self.assertEqual(record['status'], 'ok');self.assertTrue(record['contract_check']['binary64_math_passed'])
        self.assertEqual(len(used_weights), 3);self.assertTrue(all(weights is self.graph.weights for weights in used_weights))
        self.assertTrue(entry.check_worker(entry.read(args.out/'result.json'), entry.source_gate(source))['passed'])
        self.assertTrue(entry.read(args.out/'export.json')['fusion']['binary64_strict_math_passed'])
        self.assertFalse((self.root/'registry.jsonl').exists())

    @staticmethod
    def observation(exit_code=0, **changes):
        record = dict(exit_code=exit_code, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        record.update(changes);return record

    def run_outer(self, failure=None, run=True):
        from pt_agent import paths
        source = self.source_fixture();checked = entry.source_gate(source)
        tool = self.root/'heir-opt';tool.write_text('fake');tool.chmod(0o700)
        args = Namespace(out=self.root/'case', run=run, python=Path('/fixture/python'), heir_opt=tool, registry=self.root/'registry.jsonl',
            **{name: None for name in entry.SOURCE_FIELDS})
        calls = []
        def process(command, **options):
            calls.append(dict(command=command, options=options))
            Path(options['log']).write_text('fake stage\n');Path(options['resources']).write_text('')
            if '--worker' in command:
                target = Path(command[command.index('--out')+1]);target.mkdir()
                record = self.successful_worker(checked)
                if failure == 'math':record['probes'][0]['logit_check']['passed'] = False
                (target/'result.json').write_text(json.dumps(record)+'\n');(target/'model.mlir').write_text('trusted fake model\n')
                return self.observation()
            target = Path(command[command.index('-o')+1]);target.write_text(self.native_text(scale=44 if failure == 'parameter' else 45))
            return self.observation(1 if failure == 'process' else 0, timed_out=(failure == 'timeout'))
        with ExitStack() as stack:
            stack.enter_context(patch.object(paths, 'RESULTS_DIR', self.root))
            stack.enter_context(patch.object(original, 'sources', return_value=source))
            stack.enter_context(patch.object(entry, '_snapshot', return_value=[]))
            stack.enter_context(patch.object(entry, '_available_ram_bytes', return_value=16*entry.GIB))
            stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=NS(free=100*entry.GIB)))
            stack.enter_context(patch.object(entry, 'run_process', side_effect=process))
            result = entry.execute(args)
        return result, calls

    def test_preparation_writes_commands_and_completion_without_starting_processes(self):
        result, calls = self.run_outer(run=False)
        self.assertEqual(result['status'], 'prepared');self.assertEqual(calls, [])
        self.assertEqual(result['attempted'], dict(export_math=0, native_compile=0))

    def test_fake_complete_pipeline_preserves_limits_one_registry_and_real_completion_endpoint(self):
        result, calls = self.run_outer()
        self.assertEqual(result['status'], 'ok');self.assertTrue(result['model_compiled']);self.assertTrue(result['strict_math_passed'])
        self.assertEqual(result['completed'], dict(export_math=1, native_compile=1))
        self.assertEqual(len(calls), 2)
        self.assertEqual([call['options']['mem_gib'] for call in calls], [12, 12])
        self.assertEqual([call['options']['fsize_gib'] for call in calls], [16, 16])
        self.assertLessEqual(calls[0]['options']['timeout'], 1920);self.assertLessEqual(calls[1]['options']['timeout'], 7200)
        registry = [json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry), 1)
        self.assertEqual(set(registry[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        marker = entry.read(self.root/'case/completion.json')
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['exclusive_attribution_ready'])
        self.assertEqual(marker['case_artifact_bytes'], entry.artifact_bytes(self.root/'case'))
        self.assertEqual(result['completion_metadata_reserved_bytes'], entry.COMPLETION_RESERVE)

    def test_failed_math_never_launches_native(self):
        result, calls = self.run_outer(failure='math')
        self.assertEqual(result['status'], 'error');self.assertEqual(len(calls), 1)
        self.assertEqual(result['attempted']['native_compile'], 0);self.assertFalse(result['model_compiled'])

    def test_native_nonzero_retains_successful_math_and_failure_completion(self):
        result, calls = self.run_outer(failure='process')
        self.assertEqual(result['status'], 'error');self.assertTrue(result['strict_math_passed'])
        self.assertEqual(result['completed']['native_compile'], 0)
        self.assertEqual(entry.read(self.root/'case/completion.json')['status'], 'error')

    def test_native_parameter_drift_is_rejected_after_clean_process(self):
        result, calls = self.run_outer(failure='parameter')
        self.assertEqual(len(calls), 2);self.assertEqual(result['status'], 'error')
        self.assertIn('parameter', result['error']);self.assertFalse(result['model_compiled'])

    def test_native_timeout_has_distinct_status_and_original_observation(self):
        result, calls = self.run_outer(failure='timeout')
        self.assertEqual(result['status'], 'timeout');self.assertTrue(result['wall_timeout_observed'])
        self.assertTrue(result['phases']['native_compile']['observation']['timed_out'])

    def test_process_clean_gate_rejects_signal_timeout_or_remaining_child(self):
        entry.clean_process(self.observation())
        for change in (dict(exit_code=1), dict(timed_out=True), dict(signal=15), dict(signals_sent=[15]), dict(remaining_pids=[1])):
            with self.subTest(change=change), self.assertRaises(RuntimeError):entry.clean_process(self.observation(**change))

    def test_terminal_budget_accounts_for_result_and_reserved_completion_metadata(self):
        out = self.root/'terminal';out.mkdir();(out/'payload').write_bytes(b'x'*100)
        record = dict(status='ok', error=None)
        with patch.object(entry, 'CASE_BYTES', 1100), patch.object(entry, 'COMPLETION_RESERVE', 1000):
            entry.terminal_result(out, record)
        self.assertEqual(record['status'], 'resource_stopped')
        self.assertEqual(record['artifact_bytes'], entry.artifact_bytes(out))
        self.assertEqual(record['completion_metadata_reserved_bytes'], 1000)

    def test_registry_identity_is_unique_and_worker_cli_refuses_missing_paths(self):
        record = dict(status='prepared', attempted={}, completed={}, strict_math_passed=None,
            model_compiled=False, static_native_bootstraps=None, **{name: None for name in entry.FORMAL})
        entry.append_registry(self.root, record, self.root/'registry.jsonl')
        with self.assertRaises(ValueError):entry.append_registry(self.root, record, self.root/'registry.jsonl')
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):entry.main(['--worker', '--out', str(self.root/'no-worker')])
        self.assertFalse((self.root/'no-worker').exists())


if __name__ == '__main__':unittest.main()
