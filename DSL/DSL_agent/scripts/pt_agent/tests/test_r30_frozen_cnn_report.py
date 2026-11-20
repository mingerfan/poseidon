"""Small synthetic report engineering checks; no model or CIFAR data reads."""
from argparse import Namespace
from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r30_common_model_e2e as fresh
from pt_agent.experiments import r30_frozen_cnn_report as report
from pt_agent.experiments.r13_common_models import FACTORIES


class Array:
    dtype = 'float64'
    shape = (1, 10)

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return [self.values]


class FrozenReportChecks(unittest.TestCase):
    def source_fixture(self, root, model='resnet20_silu'):
        case = root / ('fresh-' + model)
        work = case / 'work'
        agent = work / 'agent'
        intake = work / 'intake'
        (intake / 'model_info').mkdir(parents=True)
        agent.mkdir()
        (work / 'calibration').mkdir()
        (work / 'calibration/calibration.json').write_text('{}')
        (work / 'compile').mkdir()
        emit = case / 'post-freeze/emit'
        emit.mkdir(parents=True)
        (emit / 'candidate_trace.mlir').write_text('module { /* synthetic native source */ }')
        (work / 'compile/native-argv.json').write_text(json.dumps(['/compiler', str(emit / 'candidate_trace.mlir')]))
        selected_source = 'def frozen_candidate(x):\n    return x\n'
        program = agent / 'program.ptdsl'
        program.write_text(selected_source)
        (agent / 'SESSION.json').write_text(json.dumps(dict(phase='done', calls=1, failed=None)))
        (agent / 'api').mkdir()
        for suffix in ['request', 'response']:
            (agent / ('api/one.' + suffix + '.json')).write_text('{}')
        factory = fresh.FACTORY_PREFIX + model
        original = FACTORIES[factory]
        norm = dict(mean=[.485, .456, .406], std=[.229, .224, .225])
        info = dict(source=dict(factory=factory, type='torch', original_source=str(original['source']),
            checkpoint=str(original['checkpoint']), inference_preprocessing_source=str(original['inference_preprocessing_source']),
            normalization=norm, preprocessing_provenance='official inference test'),
            task='classification', metric='classification_top1', threshold={'acc_drop_pp': .5},
            split=dict(kind='legacy_cifar', report=list(range(8000))))
        info_path = intake / 'model_info/model_info.json'
        info_path.write_text(json.dumps(info))
        criterion = dict(passed=True, finite=True, dtype_ok=True, shape_ok=True,
                         criterion=report.CRITERION, max_error_ratio=0.)
        row = dict(iteration=1, origin='llm', passed=True, diverged=False, n_diverged=0,
                   total=1000, acc_drop_pp=0., max_u_calibrated=.5, max_u_analytic=.1)
        worker = dict(status='ok', frozen=True, model=model, source_model_info=str(info_path),
            source_calibration=str(work / 'calibration/calibration.json'), quality_acceptance=dict(passed=True),
            actual_api_calls=1, recovered_api_calls=0, report_samples=0, compiler_runs=0, real_ckks_runs=0,
            selected_program=str(program), agent=dict(best_program=selected_source, llm_calls=1,
                confirm=dict(selected=dict(iteration=1, origin='llm', source=selected_source), evals=[row])),
            strict_math_probes=[dict(sample_id='test:' + str(9000 + i), exact=criterion, approximate=criterion) for i in range(3)])
        (agent / 'worker-result.json').write_text(json.dumps(worker))
        audit = fresh.audit_agent(worker, model=model, model_info=info_path,
            calibration=work / 'calibration/calibration.json', agent_dir=agent)
        audit['source_result'] = str(agent / 'worker-result.json')
        (case / 'post-run-quality-audit.json').write_text(json.dumps(audit))
        clean = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        phases = {name: dict(status='ok') for name in fresh.PHASES}
        phases['intake'].update(import_completed=True, real_probes_passed=True, graph_math_passed=True,
            roundtrip_passed=True, static_coverage=True,
            graph_math=dict(passed=True, samples=[dict(passed=True, finite=True) for _ in range(3)]))
        for phase in ['trace', 'compile']:
            phases[phase]['process'] = clean
        result = dict(round='R30', model=model, status='ok', mode='fresh-common-model-agent-e2e', error=None,
            completed={name: 1 for name in fresh.PHASES}, phases=phases,
            processes=[dict(phase=name, observation=clean) for name in ['trace', 'compile']],
            quality_audit=str(case / 'post-run-quality-audit.json'),
            source=dict(records=str(paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records')))
        (case / 'result.json').write_text(json.dumps(result))
        (case / 'completion.json').write_text(json.dumps(dict(status='ok', registry_persisted=True,
            required_artifacts_persisted=True)))
        # Metadata-only placeholders; these are never opened as NPZ arrays by tests.
        (intake / 'training_probes.npz').write_text('synthetic saved probe placeholder')
        (intake / 'training_probe_source.json').write_text(json.dumps(dict(split='train', indices=[0, 1, 2],
            sample_ids=['train:0', 'train:1', 'train:2'])))
        return case

    def test_gate_supports_all_three_original_models_without_script_identity_requirement(self):
        for model in report.MODELS:
            with self.subTest(model=model), TemporaryDirectory() as directory:
                case = self.source_fixture(Path(directory), model)
                (case / 'source').mkdir()
                (case / 'source/r30_common_model_e2e.py').write_text('# Earlier approved source snapshot\n')
                if model == 'alexnet_silu_avgpool':
                    mlir = case / 'post-freeze/emit/candidate_trace.mlir'
                    mlir.rename(mlir.with_name(mlir.name + '.gz'))
                gate = report.source_gate(case, model)
                self.assertTrue(gate['source_program_ast_matches'])
                self.assertFalse(gate['current_script_ast_identity_required'])
                self.assertEqual(gate['report_count'], 8000)
                self.assertEqual(gate['original_r13_math_checks_reused'], 3)
                self.assertEqual(gate['compiled_input_retention'],
                    'reversible-gzip' if model == 'alexnet_silu_avgpool' else 'raw')

    def test_gate_rejects_unfinished_compile_bad_quality_and_program_changes(self):
        mutations = [lambda d: d.update(status='running'),
                     lambda d: d['completed'].pop('compile'),
                     lambda d: d['processes'][-1]['observation'].update(signals_sent=['TERM']),
                     lambda d: d.update(quality_audit='/foreign/audit.json')]
        with TemporaryDirectory() as directory:
            case = self.source_fixture(Path(directory))
            original = report.read(case / 'result.json')
            for mutate in mutations:
                value = deepcopy(original)
                mutate(value)
                (case / 'result.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    report.source_gate(case, 'resnet20_silu')
            (case / 'result.json').write_text(json.dumps(original))
            program = case / 'work/agent/program.ptdsl'
            program.write_text('def changed(x):\n    return x + 1\n')
            with self.assertRaises(ValueError):
                report.source_gate(case, 'resnet20_silu')
            self.assertFalse((Path(directory) / 'frozen-cnn-report-access').exists())

    def test_access_marker_is_exclusive_and_keeps_source_case_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.source_fixture(root)
            gate = report.source_gate(case, 'resnet20_silu')
            with patch.object(paths, 'RESULTS_DIR', root):
                marker = report.claim_access(root / 'report-one', gate)
                with self.assertRaises(FileExistsError):
                    report.claim_access(root / 'report-two', gate)
            self.assertNotIn(case, marker.parents)
            self.assertFalse(report.read(marker)['report_started'])
            self.assertFalse(report.read(marker)['automatic_retry'])

    def test_all_8000_ids_outputs_and_nonfinite_report_observation_are_retained(self):
        read_ids, reference_ids, approximate_ids = [], [], []
        def sample(index):
            read_ids.append(index)
            return index, index % 10
        def exact(index):
            reference_ids.append(index)
            return Array([1. if n == index % 10 else 0. for n in range(10)])
        def approximate(index):
            approximate_ids.append(index)
            if index == 7:
                return Array([float('inf')] + [0.] * 9)
            return exact(index)
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'rows.jsonl'
            progress = []
            result = report.report_rows(target, sample, exact, approximate, lambda **values: progress.append(values))
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual(read_ids, list(range(8000)))
            self.assertEqual(approximate_ids, read_ids)
            self.assertEqual([row['sample_id'] for row in rows], ['test:' + str(i) for i in range(8000)])
            self.assertTrue(all(len(row['reference']) == len(row['approximate']) == 10 for row in rows))
            self.assertEqual(rows[7]['approximate'][0], None)
            self.assertFalse(result['all_finite'])
            self.assertFalse(result['report_policy_passed'])
            self.assertTrue(result['report_completed'])
            self.assertEqual(result['reference_correct'], 8000)
            self.assertEqual(result['approximate_correct'], 7999)
            self.assertFalse(result['used_for_selection'])
            self.assertEqual(result['batch_size'], 1)

    def test_timeout_is_not_swallowed_by_per_sample_approximation_recording(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'partial.jsonl'
            seen = []
            def timeout(image):
                raise TimeoutError('synthetic finite allowance')
            with self.assertRaises(TimeoutError):
                report.report_rows(target, lambda i: (seen.append(i) or i, 0),
                    lambda image: Array([0.] * 10), timeout, lambda **values: None)
            self.assertEqual(seen, [0])
            self.assertTrue(target.is_file())
            self.assertEqual(target.stat().st_size, 0)

    def test_numeric_exceptions_are_quality_observations_and_finish_all_8000_rows(self):
        def approximate(index):
            if index == 3:
                raise FloatingPointError('synthetic numerical nonfinite value')
            if index == 4:
                raise OverflowError('synthetic numerical overflow')
            return Array([1.] + [0.] * 9)
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'numeric-rows.jsonl'
            result = report.report_rows(target, lambda index: (index, 0),
                lambda image: Array([1.] + [0.] * 9), approximate, lambda **values: None)
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual([row['index'] for row in rows], list(range(8000)))
            self.assertTrue(rows[3]['error'].startswith('FloatingPointError:'))
            self.assertTrue(rows[4]['error'].startswith('OverflowError:'))
            self.assertTrue(all(rows[index]['approximate'] == [None] * 10 for index in [3, 4]))
            self.assertEqual(result['approximate_correct'], 7998)
            self.assertEqual(result['finite_samples'], 7998)
            self.assertTrue(result['report_completed'])
            self.assertFalse(result['report_policy_passed'])
            self.assertFalse(result['used_for_selection'])

    def test_unexpected_approximation_errors_stop_at_current_row_and_keep_partial_progress(self):
        for kind in [NameError, TypeError, ValueError, RuntimeError]:
            with self.subTest(exception=kind.__name__), TemporaryDirectory() as directory:
                target = Path(directory) / 'partial.jsonl'
                reads, progress = [], {}
                def sample(index):
                    reads.append(index)
                    return index, 0
                def approximate(index):
                    if index == 3:
                        raise kind('synthetic implementation error')
                    return Array([1.] + [0.] * 9)
                with self.assertRaisesRegex(kind, 'synthetic implementation error'):
                    report.report_rows(target, sample, lambda image: Array([1.] + [0.] * 9),
                        approximate, lambda **values: progress.update(values))
                rows = [json.loads(line) for line in target.read_text().splitlines()]
                self.assertEqual(reads, [0, 1, 2, 3])
                self.assertEqual([row['sample_id'] for row in rows], ['test:0', 'test:1', 'test:2'])
                self.assertEqual(progress['report_samples_read'], 4)
                self.assertEqual(progress['report_samples_completed'], 3)
                self.assertEqual(progress['active_sample_id'], 'test:3')

    def test_tensor_adapter_only_reclassifies_its_exact_known_nonfinite_guard(self):
        class EngineError(Exception):
            pass
        backend = ModuleType('pt_agent.backends')
        backend.BackendError = EngineError
        tensor = ModuleType('pt_agent.tensor_sim')
        context = SimpleNamespace(program=object(), info=SimpleNamespace(graph=SimpleNamespace(weights={})))
        for message, expected in [('Mock arithmetic produced a nonfinite value', FloatingPointError),
                                  ('unsupported native operation', EngineError)]:
            with self.subTest(message=message), \
                    patch.dict(sys.modules, {'pt_agent.backends': backend, 'pt_agent.tensor_sim': tensor}):
                def fail(*args, **kwargs):
                    raise EngineError(message)
                tensor.run_tensor = fail
                with self.assertRaisesRegex(expected, message):
                    report.approximate_output(context, object())

    def test_worker_marks_implementation_failure_and_preserves_completed_rows_and_access(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.source_fixture(root)
            out = root / 'report-one'
            out.mkdir()
            args = Namespace(source_case=case, out=out, model='resnet20_silu', seconds=7200, memory_gib=6)
            with patch.object(paths, 'RESULTS_DIR', root):
                args.access_marker = report.claim_access(out, report.source_gate(case, args.model))
            seen = []
            def read_sample(context, index):
                seen.append(index)
                return index, 0
            def approximate(context, index):
                if index == 1:
                    raise NameError('synthetic missing implementation name')
                return Array([1.] + [0.] * 9)
            with patch.object(report.resource, 'setrlimit'), \
                    patch.object(report, 'load_context', return_value=object()), \
                    patch.object(report, 'training_math', return_value={'passed': True, 'checks_completed': 12}), \
                    patch.object(report, 'read_report_sample', side_effect=read_sample), \
                    patch.object(report, 'reference_output', return_value=Array([1.] + [0.] * 9)), \
                    patch.object(report, 'approximate_output', side_effect=approximate):
                self.assertEqual(report.worker(args), 1)
            saved = report.read(out / 'worker-result.json')
            marker = report.read(args.access_marker)
            rows = [json.loads(line) for line in (out / 'report-rows.jsonl').read_text().splitlines()]
            self.assertEqual(seen, [0, 1])
            self.assertEqual(saved['failure_kind'], 'execution_error')
            self.assertEqual(saved['report_samples_read'], 2)
            self.assertEqual(saved['report_samples_completed'], 1)
            self.assertFalse(saved['report_completed'])
            self.assertEqual([row['sample_id'] for row in rows], ['test:0'])
            self.assertTrue(marker['report_started'])
            self.assertFalse(marker['report_complete'])
            self.assertEqual(marker['failure_kind'], 'execution_error')
            self.assertFalse(marker['automatic_retry'])

    def test_failed_training_math_stops_before_report_reader(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.source_fixture(root)
            out = root / 'report-one'
            out.mkdir()
            args = Namespace(source_case=case, out=out, model='resnet20_silu', seconds=7200, memory_gib=6)
            with patch.object(paths, 'RESULTS_DIR', root):
                args.access_marker = report.claim_access(out, report.source_gate(case, args.model))
            with patch.object(report.resource, 'setrlimit'), \
                    patch.object(report, 'load_context', return_value=object()), \
                    patch.object(report, 'training_math', side_effect=ValueError('synthetic training math failure')), \
                    patch.object(report, 'read_report_sample') as reader:
                self.assertEqual(report.worker(args), 1)
            reader.assert_not_called()
            self.assertFalse(report.read(args.access_marker)['report_started'])
            self.assertEqual(report.read(out / 'worker-result.json')['report_samples_read'], 0)
            self.assertFalse((out / 'report-rows.jsonl').exists())

    def test_training_proof_routes_four_strict_checks_for_each_of_three_probes(self):
        from pt_agent.experiments import r15_rule_validation_worker as arithmetic
        class Probes:
            def __getitem__(self, selection):
                return selection.start
        array = Array([0.] * 10)
        analyzer = ModuleType('pt_agent.analyzer')
        analyzer.reference_forward = lambda graph, inputs: {'output': array}
        tensor = ModuleType('pt_agent.tensor_sim')
        tensor.run_tensor = lambda *args, **kwargs: array
        context = SimpleNamespace(probes=Probes(), program=object(),
            info=SimpleNamespace(graph=SimpleNamespace(inputs=['input'], outputs=['output'], weights={})))
        with patch.dict(sys.modules, {'pt_agent.analyzer': analyzer, 'pt_agent.tensor_sim': tensor}), \
                patch.object(report, 'reference_output', return_value=array) as original, \
                patch.object(report, 'approximate_output', return_value=array) as approximate, \
                patch.object(arithmetic, '_packed_logits', return_value=array) as packed, \
                patch.object(arithmetic, 'check_math', return_value={'passed': True}) as strict:
            proof = report.training_math(context)
        self.assertEqual(proof['checks_completed'], 12)
        self.assertEqual(strict.call_count, 12)
        self.assertEqual(original.call_count, 3)
        self.assertEqual(approximate.call_count, 3)
        self.assertEqual(packed.call_count, 6)
        self.assertEqual([row['sample_id'] for row in proof['samples']], ['train:0', 'train:1', 'train:2'])
        self.assertTrue(all(len(row['checks']) == 4 for row in proof['samples']))
        self.assertEqual(proof['criterion'], '1e-12+1e-12*abs(reference)')

    def test_outer_failure_retains_partial_worker_and_registers_terminal_eight_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.source_fixture(root)
            args = Namespace(source_case=case, out=root / 'new-report', model='resnet20_silu', seconds=7200,
                memory_gib=6, intake_python=Path('/existing/python'))
            def run(command, **kwargs):
                self.assertEqual(kwargs['mem_gib'], 6)
                self.assertEqual(kwargs['timeout'], 7320)
                report.write_json(args.out / 'worker-result.json', dict(status='error', report_completed=False,
                    report_samples_read=11, report_samples_completed=10, llm_calls=0, compiler_runs=0, ckks_runs=0))
                return dict(exit_code=1, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
            with patch.object(paths, 'RESULTS_DIR', root), \
                    patch('pt_agent.experiments.bert_sst2.budget_guard', return_value=nullcontext({})), \
                    patch('pt_agent.registry.git_rev', return_value='synthetic'), \
                    patch.object(report, 'run_process', side_effect=run):
                self.assertEqual(report.execute(args), 1)
            saved = report.read(args.out / 'result.json')
            self.assertEqual(saved['report_samples_read'], 11)
            self.assertEqual(saved['report_samples_completed'], 10)
            self.assertFalse(saved['automatic_retry'])
            self.assertFalse(saved['feedback_to_agent'])
            self.assertTrue(Path(saved['access_marker']).is_file())
            rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            self.assertEqual(rows[0]['metrics']['status'], 'error')
            self.assertTrue(report.read(args.out / 'completion.json')['registry_persisted'])

    def test_cli_limits_report_to_7200_seconds_and_six_gibibytes(self):
        from contextlib import redirect_stderr
        for option, value in [('--seconds', '7201'), ('--memory-gib', '6.1')]:
            with patch.object(report, 'execute') as execute, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                report.main(['--model', 'resnet20_silu', '--source-case', '/source', '--out', '/report', option, value])
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
