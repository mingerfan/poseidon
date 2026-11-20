"""R14 inner cases; real model preparation is exclusively owned by the outer run."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.paths import RESULTS_DIR
from pt_agent.tests._artifacts import require


class RuleCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from pt_agent.experiments import r14_common_worker as worker
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.models.info import ModelInfo
        from pt_agent.data import cifar
        from pt_agent.experiments.r13_common_models import FACTORIES

        cls.np, cls.worker, cls.Graph, cls.Node = np, worker, Graph, Node
        cls.TensorInfo, cls.ModelInfo, cls.cifar, cls.factories = TensorInfo, ModelInfo, cifar, FACTORIES

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def fixture(self):
        np = self.np
        tensors = {'camera_pixels': self.TensorInfo('camera_pixels', (1, 3, 32, 32), 'input'),
                   'convolved': self.TensorInfo('convolved', (1, 16, 32, 32), 'act'),
                   'pooled': self.TensorInfo('pooled', (1, 16), 'act'),
                   'activated': self.TensorInfo('activated', (1, 16), 'act'),
                   'declared_logits': self.TensorInfo('declared_logits', (1, 10), 'act'),
                   'stem.weight': self.TensorInfo('stem.weight', (16, 3, 1, 1), 'weight'),
                   'head.weight': self.TensorInfo('head.weight', (10, 16), 'weight'),
                   'head.bias': self.TensorInfo('head.bias', (10,), 'weight')}
        graph = self.Graph('r14_fixture', [
            self.Node('n0', 'conv2d', ['camera_pixels'], ['convolved'],
                      dict(weight='stem.weight', stride=(1, 1), padding=(0, 0),
                           dilation=(1, 1), groups=1), 'fixture 1x1 convolution'),
            self.Node('n1', 'mean', ['convolved'], ['pooled'],
                      dict(axes=(2, 3), keepdim=False), 'fixture mean'),
            self.Node('n2', 'act', ['pooled'], ['activated'], dict(fn='silu'), 'fixture SiLU'),
            self.Node('n3', 'linear', ['activated'], ['declared_logits'],
                      dict(weight='head.weight', bias='head.bias'), 'fixture head')],
            tensors, {'stem.weight': np.arange(48, dtype=np.float64).reshape(16, 3, 1, 1) / 1000,
                      'head.weight': np.arange(160, dtype=np.float64).reshape(10, 16) / 100,
                      'head.bias': np.arange(10, dtype=np.float64) / 1000},
            ['camera_pixels'], ['declared_logits'])
        model = next(iter(self.factories))
        metadata = self.factories[model]
        normalization = dict(mean=list(self.cifar.MEAN), std=list(self.cifar.STD))
        source_record = dict(records=str(self.directory / 'records'),
                             selections=str(self.directory / 'selections.json'), split='train',
                             indices=[0, 1, 2], sample_ids=['train:0', 'train:1', 'train:2'],
                             labels=[0, 1, 2], entries=[dict(index=i, label=i, row=i,
                                                          shard='train-00000.records') for i in range(3)],
                             normalization=normalization)
        source = dict(type='torch', path=model, factory=model, spec_revision='R13',
                      intake=dict(original_nodes=4, folded_nodes=0, original_unmapped={}),
                      original_source=str(metadata['source'].resolve()),
                      checkpoint=str(metadata['checkpoint'].resolve()),
                      inference_preprocessing_source=str(metadata['inference_preprocessing_source'].resolve()),
                      normalization=normalization, calibration_source=source_record,
                      preprocessing_provenance='official inference test', training_preprocessing_verified=False)
        split = dict(kind='legacy_cifar', heldout=list(self.cifar.HELDOUT),
                     confirm=list(self.cifar.CONFIRM), report=list(self.cifar.REPORT),
                     calibration='frozen training selection')
        info = self.ModelInfo(source, graph, dict(mode='original tensors', internal_regression_source='legacy_cifar'),
                              [dict(name='camera_pixels', kind='features', dtype='float64', shape=[1, 3, 32, 32])],
                              None, 'classification', 'classification_top1', dict(acc_drop_pp=0.5), split,
                              [dict(task_inference='fixture classification')])
        worker = dict(round='R13', status='ok', model=model, model_source=copy.deepcopy(source),
                      split=copy.deepcopy(split), source=copy.deepcopy(source_record),
                      import_completed=True, real_probes_passed=True, graph_math_passed=True,
                      roundtrip_passed=True, static_coverage=True)
        r13 = dict(round='R13', mode='common-model-intake-preparation', status='ok', model=model, worker=worker)
        image = np.arange(3072, dtype=np.float64).reshape(1, 3, 32, 32) / 10000
        images = np.broadcast_to(image, (3, 3, 32, 32))
        from pt_agent.analyzer import reference_forward
        references = [dict(probe_index=i, input_name='camera_pixels', output_name='declared_logits',
                           reference=reference_forward(graph, {'camera_pixels': images[i:i + 1]})['declared_logits'].tolist(),
                           output=[[999.] * 10]) for i in range(3)]
        r13['worker']['graph_math'] = dict(passed=True, criterion=self.worker.GRAPH_CRITERION, samples=references)
        return info, r13, images, references

    def calibration(self):
        return {name: dict(min=0., max=1., absmax=1., p999=1., count=1000)
                for name in ('convolved', 'pooled', 'activated', 'declared_logits')}

    def main_fixture(self):
        info, r13, images, references = self.fixture()
        saved = self.directory / 'model_info'
        info.save(saved)
        probe = self.directory / 'training_probes.npz'
        self.np.savez(probe, input=images, labels=self.np.array([0, 1, 2]), indices=self.np.array([0, 1, 2]))
        source_path = self.directory / 'training_probe_source.json'
        source_path.write_text(json.dumps(info.source['calibration_source']))
        r13['worker']['artifacts'] = dict(model_info=str(saved), training_probes=str(probe),
                                         training_probe_source=str(source_path))
        r13_path = self.directory / 'r13-result.json'
        r13_path.write_text(json.dumps(r13))
        frontend = self.directory / 'frontend.py'
        frontend.write_text("raise AssertionError('Generated source must never execute')\n")
        work = self.directory / 'case/worker'
        cli = ['--model-info', str(saved), '--r13-result', str(r13_path), '--r13-probes', str(probe),
               '--r13-probe-source', str(source_path), '--inputs', str(self.directory / 'records'),
               '--selections', str(self.directory / 'selections.json'), '--frontend', str(frontend),
               '--work-dir', str(work)]
        labels = self.np.arange(1000) % 10
        source = dict(info.source['calibration_source'], count=1000, batch=1,
                      indices=list(range(1000)), sample_ids=['train:' + str(i) for i in range(1000)])
        arrays = {'camera_pixels': self.np.broadcast_to(images[:1], (1000, 3, 32, 32))}
        return cli, work, (arrays, labels, source), info, r13

    def test_saved_source_preserves_original_features_and_metadata(self):
        info, r13, _, _ = self.fixture()
        before = copy.deepcopy((info.source, info.inputs, info.split, info.assumptions))
        self.assertTrue(self.worker.check_saved_source(info, r13)['passed'])
        self.assertEqual((info.source, info.inputs, info.split, info.assumptions), before)
        self.assertEqual(info.inputs[0]['kind'], 'features')

    def test_saved_source_rejects_unapproved_or_unsuccessful_source(self):
        for case in ('failed_r13', 'checkpoint', 'normalization', 'split', 'kind', 'shape'):
            with self.subTest(case=case):
                info, r13, _, _ = self.fixture()
                if case == 'failed_r13': r13['worker']['roundtrip_passed'] = False
                elif case == 'checkpoint': info.source['checkpoint'] = str(self.directory / 'different.model')
                elif case == 'normalization': info.source['normalization'] = dict(mean=[0., 0., 0.], std=[1., 1., 1.])
                elif case == 'split': info.split['report'] = []
                elif case == 'kind': info.inputs[0]['kind'] = 'image'
                else: info.inputs[0]['shape'] = [1, 3, 16, 16]
                self.assertFalse(self.worker.check_saved_source(info, r13)['passed'])

    def test_reference_reads_reference_field_and_preserves_declared_names(self):
        info, _, images, references = self.fixture()
        observation = self.worker.compare_reference(info.graph, {'camera_pixels': images}, references)
        self.assertTrue(observation['passed'])
        self.assertEqual(observation['criterion'], '1e-12+1e-12*abs(reference)')
        self.assertEqual(len(observation['samples']), 3)
        self.assertEqual(observation['samples'][0]['reference'], references[0]['reference'])
        self.assertNotEqual(observation['samples'][0]['reference'], references[0]['output'])

    def test_reference_failure_stops_later_forward_and_keeps_original_values(self):
        from pt_agent import analyzer
        info, _, images, references = self.fixture()
        references[1]['reference'][0][0] += 1e-7
        with patch.object(analyzer, 'reference_forward', wraps=analyzer.reference_forward) as forward:
            observation = self.worker.compare_reference(info.graph, {'camera_pixels': images}, references)
        self.assertFalse(observation['passed'])
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(len(observation['samples']), 2)
        self.assertEqual(observation['samples'][1]['reference'], references[1]['reference'])
        self.assertFalse(observation['samples'][1]['passed'])

    def test_reference_rejects_identity_order_or_nonfinite_output(self):
        info, _, images, references = self.fixture()
        references[0]['probe_index'] = 2
        with self.assertRaises(ValueError):
            self.worker.compare_reference(info.graph, {'camera_pixels': images}, references)
        references[0]['probe_index'] = 0
        info.graph.weights['head.bias'][0] = self.np.nan
        observation = self.worker.compare_reference(info.graph, {'camera_pixels': images}, references)
        self.assertFalse(observation['passed'])
        self.assertEqual(len(observation['samples']), 1)
        json.dumps(observation, allow_nan=False)

    def test_calibration_rejects_missing_point_nonfinite_and_invalid_counts(self):
        info, _, _, _ = self.fixture()
        self.worker._check_calibration(info.graph, self.calibration())
        for case in ('missing', 'nan', 'count', 'range'):
            with self.subTest(case=case):
                observation = self.calibration()
                if case == 'missing': del observation['pooled']
                elif case == 'nan': observation['pooled']['max'] = self.np.nan
                elif case == 'count': observation['pooled']['count'] = 0
                else: observation['pooled']['min'] = 2.
                with self.assertRaises(self.worker.WorkerFailure):
                    self.worker._check_calibration(info.graph, observation)

    def test_signature_handles_features_without_mutation(self):
        from pt_agent.dsl.parser import parse
        from dataclasses import replace
        info, _, _, _ = self.fixture()
        program = parse('@fhe.program(slots=16384)\ndef fixture(camera_pixels: Cipher[1,3,32,32]):\n'
                        '    p = pack(camera_pixels, layout="mux", gap=1)\n    return p\n')
        self.assertTrue(self.worker.check_program_signature(info, program)['passed'])
        bad = replace(program, params=(replace(program.params[0], name='renamed'),))
        self.assertFalse(self.worker.check_program_signature(info, bad)['passed'])
        bad = replace(program, outputs=program.outputs + program.outputs)
        self.assertFalse(self.worker.check_program_signature(info, bad)['passed'])
        self.assertEqual(info.inputs[0]['kind'], 'features')

    def test_calibration_reader_checks_train_metadata_before_loading_arrays(self):
        from pt_agent.data.samples import LegacyCifar
        info, _, _, _ = self.fixture()
        records = self.directory / 'records'
        records.mkdir()
        metadata = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                        row_label_byte=0, preprocessing_applied=False, train_count=50000,
                        test_count=10000, pixel_layout='uint8 planar RGB 3x32x32, C order')
        (records / 'dataset.json').write_text(json.dumps(metadata))
        selections = self.directory / 'selections.json'
        original = [dict(index=i, label=i % 10, shard='train-00000.records', row=i) for i in range(1000)]
        args = SimpleNamespace(inputs=records, selections=selections)
        for case in ('test_shard', 'duplicate', 'out_of_range', 'wrong_row', 'short'):
            with self.subTest(case=case):
                entries = copy.deepcopy(original)
                if case == 'test_shard': entries[0]['shard'] = 'test-00000.records'
                elif case == 'duplicate': entries[1] = copy.deepcopy(entries[0])
                elif case == 'out_of_range': entries[0]['index'] = 50000
                elif case == 'wrong_row': entries[0]['row'] = 3
                else: entries.pop()
                selections.write_text(json.dumps(dict(calibration=entries)))
                with patch.object(self.cifar, 'RECORDS', records), patch.object(self.cifar, 'SELECTIONS', selections), \
                     patch.object(LegacyCifar, 'get', side_effect=AssertionError('Invalid metadata reached sample loading')) as get:
                    with self.assertRaises(self.worker.WorkerFailure): self.worker._read_calibration(args, info)
                get.assert_not_called()
        selections.write_text(json.dumps(dict(calibration=original)))
        array = self.np.broadcast_to(self.np.zeros((1, 3, 32, 32)), (1000, 3, 32, 32))
        labels = self.np.arange(1000) % 10
        with patch.object(self.cifar, 'RECORDS', records), patch.object(self.cifar, 'SELECTIONS', selections), \
             patch.object(LegacyCifar, 'get', return_value=({'camera_pixels': array}, labels)) as get:
            _, _, observed = self.worker._read_calibration(args, info)
        get.assert_called_once_with('calibration')
        self.assertEqual(observed['count'], 1000)
        self.assertEqual(observed['batch'], 1)
        self.assertEqual(observed['sample_ids'][:3], ['train:0', 'train:1', 'train:2'])

    def test_emit_budget_is_prewrite_estimate_with_separate_three_gib_limit(self):
        observed = self.worker._emit_budget('x', {'k0': SimpleNamespace(nbytes=16)})
        self.assertEqual(observed['estimated_bytes'], 1 + 16 + 1024 + 1024**2)
        self.assertEqual(observed['limit_bytes'], 3 * 1024**3)
        self.assertTrue(observed['within_budget'])
        oversized = self.worker._emit_budget('x', {'k0': SimpleNamespace(nbytes=3 * 1024**3)})
        self.assertFalse(oversized['within_budget'])

    def test_complete_small_graph_uses_fixed_rule_count_emit_and_only_parses_source(self):
        from pt_agent import analyzer, toolchain
        from pt_agent.planners import rule
        cli, work, calibration_source, info, _ = self.main_fixture()
        original_inputs = copy.deepcopy(info.inputs)
        with patch.object(self.worker, '_read_calibration', return_value=calibration_source), \
             patch.object(analyzer, 'calibrate', return_value=self.calibration()) as calibrate, \
             patch.object(rule, 'plan', wraps=rule.plan) as plan, \
             patch.object(toolchain, 'trace', side_effect=AssertionError('Unexpected trace')), \
             patch.object(toolchain, 'compile', side_effect=AssertionError('Unexpected compile')), \
             patch.object(toolchain, 'execute_cpu', side_effect=AssertionError('Unexpected CKKS')):
            code = self.worker.main(cli)
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(code, 0, {key: result.get(key) for key in ('status', 'error', 'failed_phase')})
        calibrate.assert_called_once()
        self.assertEqual(calibrate.call_args.kwargs, dict(batch=1))
        plan.assert_called_once()
        self.assertEqual(plan.call_args.kwargs, dict(deg=63, margin=1.25, slots=16384,
                                                   placement='depth-dp', refresh='compiler', act_method='cheb'))
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'ok')
        for key in self.worker.REQUIRED_STAGES: self.assertIs(result[key], True)
        for key in ('counts', 'ops_total', 'distinct_consts'):
            self.assertEqual(result['count_stats'][key], result['emit_stats'][key])
        for key in ('compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance'):
            self.assertIsNone(result[key])
        for key in ('llm_calls', 'real_ckks_attempts', 'trace_runs', 'compiler_runs', 'keygen_runs', 'downloads', 'installations'):
            self.assertEqual(result[key], 0)
        self.assertEqual(result['input_metadata'], original_inputs)
        metadata = json.loads(Path(result['artifacts']['emit_metadata']).read_text())
        self.assertTrue(metadata['source_parsed'])
        self.assertFalse(metadata['source_executed'])
        self.assertTrue(all(row['finite'] for row in metadata['constants']))
        from pt_agent.experiments.r10_paper_metrics import read_events, summarize_events
        timing = summarize_events(read_events(work / 'events.jsonl'))
        self.assertEqual(timing['diagnostics'], [])
        root = next(item for item in timing['spans'] if item['parent_id'] is None)
        self.assertEqual(sum(item['exclusive_ns'] for item in timing['spans']), root['inclusive_ns'])
        probes = [item for item in timing['spans'] if item['stage_id'] == 'reference_sample']
        self.assertEqual(len(probes), 3)
        self.assertTrue(result['case_budget_observations'])

    def test_math_failure_persists_result_and_prevents_calibration(self):
        from pt_agent import analyzer
        cli, work, source, _, r13 = self.main_fixture()
        r13['worker']['graph_math']['samples'][0]['reference'][0][0] += 1e-7
        (self.directory / 'r13-result.json').write_text(json.dumps(r13))
        with patch.object(self.worker, '_read_calibration', return_value=source), \
             patch.object(analyzer, 'calibrate', side_effect=AssertionError('Calibration after failed reference')) as calibrate:
            self.assertEqual(self.worker.main(cli), 1)
        calibrate.assert_not_called()
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'math_failed')
        self.assertEqual(result['failed_phase'], 'reference_check')
        self.assertFalse(result['reference_passed'])
        self.assertIsNone(result['calibration_completed'])
        self.assertEqual(result['calibration_attempts'], 0)
        self.assertFalse((work / 'emit').exists())

    def test_static_error_prevents_lowering(self):
        from pt_agent import analyzer, lower as lowering
        from pt_agent.dsl import typecheck
        cli, work, source, _, _ = self.main_fixture()
        with patch.object(self.worker, '_read_calibration', return_value=source), \
             patch.object(analyzer, 'calibrate', return_value=self.calibration()), \
             patch.object(typecheck, 'check', return_value=SimpleNamespace(ok=False, diags=[])), \
             patch.object(lowering, 'lower', side_effect=AssertionError('Lower after static error')) as lower:
            self.assertEqual(self.worker.main(cli), 1)
        lower.assert_not_called()
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'typecheck_failed')
        self.assertEqual(result['count_attempts'], 0)

    def test_count_emit_disagreement_prevents_writing_source(self):
        from pt_agent import analyzer
        from pt_agent.backends import EmitBackend
        cli, work, source, _, _ = self.main_fixture()
        original_stats = EmitBackend.stats

        def changed_stats(backend):
            observed = original_stats(backend)
            observed['counts']['add'] += 1
            return observed

        with patch.object(self.worker, '_read_calibration', return_value=source), \
             patch.object(analyzer, 'calibrate', return_value=self.calibration()), \
             patch.object(EmitBackend, 'stats', changed_stats), \
             patch.object(EmitBackend, 'write', side_effect=AssertionError('Emit write after mismatch')) as write:
            self.assertEqual(self.worker.main(cli), 1)
        write.assert_not_called()
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'count_emit_mismatch',
                         {key: result.get(key) for key in ('status', 'error', 'failed_phase')})
        self.assertFalse(result['stats_equal'])

    def test_emit_budget_failure_prevents_write(self):
        from pt_agent import analyzer
        from pt_agent.backends import EmitBackend
        cli, work, source, _, _ = self.main_fixture()
        budget = self.worker._emit_budget('x', {'k0': SimpleNamespace(nbytes=3 * 1024**3)})
        with patch.object(self.worker, '_read_calibration', return_value=source), \
             patch.object(analyzer, 'calibrate', return_value=self.calibration()), \
             patch.object(self.worker, '_emit_budget', return_value=budget), \
             patch.object(EmitBackend, 'write', side_effect=AssertionError('Oversize export reached disk')) as write:
            self.assertEqual(self.worker.main(cli), 1)
        write.assert_not_called()
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'artifact_budget_exceeded',
                         {key: result.get(key) for key in ('status', 'error', 'failed_phase')})
        self.assertTrue(result['stats_equal'])
        self.assertIsNone(result['emit_completed'])

    def test_memory_and_timeout_failures_persist_unmeasured_fields(self):
        for failure, status in ((MemoryError('fixture address-space failure'), 'resource_limit'),
                                (self.worker.WorkerFailure('fixture expiry', status='timeout'), 'timeout')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                work = Path(directory) / 'worker'
                cli = sum((['--' + name, str(work if name == 'work-dir' else Path(directory) / name)]
                           for name in ('model-info', 'r13-result', 'r13-probes', 'r13-probe-source',
                                        'inputs', 'selections', 'frontend', 'work-dir')), [])
                with patch.object(self.worker, 'run', side_effect=failure):
                    self.assertEqual(self.worker.main(cli), 1)
                result = json.loads((work / 'worker_result.json').read_text())
                self.assertEqual(result['status'], status)
                self.assertIsNone(result['lowering_ready'])
                self.assertIsNone(result['predicted_latency'])
                self.assertEqual(result['llm_calls'], 0)
                self.assertEqual(result['real_ckks_attempts'], 0)


R13_RUNS = tuple(RESULTS_DIR / name for name in (
    'r13-common-resnet20-silu-20261115', 'r13-common-vgg16-silu-avgpool-20261115',
    'r13-common-alexnet-silu-avgpool-20261115'))


@require(tuple(path / 'result.json' for path in R13_RUNS),
         tuple(path / 'worker/model_info/model_info.json' for path in R13_RUNS),
         tuple(path / 'worker/model_info/graph/graph.json' for path in R13_RUNS))
class SavedMetadataCases(unittest.TestCase):
    def test_three_actual_saved_sources_fit_contract_without_loading_weights(self):
        from pt_agent.experiments.r14_common_worker import check_saved_source
        for directory in R13_RUNS:
            with self.subTest(run=directory.name):
                record = json.loads((directory / 'result.json').read_text())
                metadata = json.loads((directory / 'worker/model_info/model_info.json').read_text())
                graph = json.loads((directory / 'worker/model_info/graph/graph.json').read_text())
                metadata['graph'] = SimpleNamespace(inputs=graph['inputs'], outputs=graph['outputs'],
                    tensors={name: SimpleNamespace(shape=shape) for name, shape, kind in graph['tensors']})
                self.assertTrue(check_saved_source(SimpleNamespace(**metadata), record)['passed'])


if __name__ == '__main__':
    unittest.main()
