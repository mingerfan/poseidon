"""Current-round Emit-resume cases using generated fixtures only."""

from contextlib import ExitStack
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


class AlexEmitResumeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from pt_agent.experiments import r14b_emit_worker as worker
        from pt_agent.tests import _r14_common_cases as fixtures

        fixtures.RuleCases.setUpClass()
        cls.np, cls.worker, cls.fixtures = np, worker, fixtures

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='r14b-emit-fixture-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self):
        """Save a small original-interface graph and a synthetic prior budget stop."""
        from pt_agent.backends import EmitBackend
        from pt_agent.dsl.parser import parse
        from pt_agent.lower import lower

        owner = self.fixtures.RuleCases()
        owner.directory = self.directory
        info, r13, _, _ = owner.fixture()
        metadata = owner.factories[self.worker.MODEL]
        info.source.update(path=self.worker.MODEL, factory=self.worker.MODEL,
                           original_source=str(metadata['source'].resolve()),
                           checkpoint=str(metadata['checkpoint'].resolve()),
                           inference_preprocessing_source=str(metadata['inference_preprocessing_source'].resolve()))
        r13.update(model=self.worker.MODEL)
        r13['worker'].update(model=self.worker.MODEL, model_source=copy.deepcopy(info.source))
        r13_dir, original_dir = self.directory / 'r13', self.directory / 'r14'
        saved, original_work = r13_dir / 'worker/model_info', original_dir / 'worker'
        original_work.mkdir(parents=True)
        info.save(saved)
        r13['worker']['artifacts'] = dict(model_info=str(saved))
        r13_path = r13_dir / 'result.json'
        r13_path.write_text(json.dumps(r13))
        frontend = self.directory / 'expr.py'
        frontend.write_text("raise AssertionError('The generated source must only be parsed')\n")
        program_path = original_work / 'program.ptdsl'
        program_path.write_text(
            '@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")\n'
            'def tiny(camera_pixels: Cipher[1,3,32,32]):\n'
            '    p = pack(camera_pixels, layout="mux", gap=1)\n'
            '    c = conv2d(p, w=W["stem.weight"], stride=(1,1), pad=(0,0), algo="mpcb")\n'
            '    m = mean(c, axis=(2,3))\n'
            '    a = act(m, fn="silu", method="cheb", deg=63, domain=(-1.25,1.25))\n'
            '    y = linear(a, w=W["head.weight"], b=W["head.bias"], algo="bsgs")\n'
            '    return y\n')
        # The fixture's saved statistic is generated without Count or a real model.
        backend = EmitBackend(16384)
        lower(parse(program_path.read_text()), info.graph.weights, backend, act_mode='program')
        statistics = backend.stats()
        del backend
        gc.collect()
        count = dict(stats=copy.deepcopy(statistics), output_layouts=[dict(kind='flat', n=16384, size=10)])
        emitted = dict(stats=copy.deepcopy(statistics), differences={}, matched=True)
        count_path, emit_path = original_work / 'count.json', original_work / 'emit_stats.json'
        count_path.write_text(json.dumps(count))
        emit_path.write_text(json.dumps(emitted))
        original_worker = dict(round='R14', status='artifact_budget_exceeded', model=self.worker.MODEL,
                               failed_phase='emit_write', emit_completed=None, lowering_ready=None,
                               rule_options=dict(self.worker.previous.RULE_OPTIONS),
                               count_stats=copy.deepcopy(statistics), emit_stats=copy.deepcopy(statistics),
                               artifacts=dict(program=str(program_path), count=str(count_path), emit_stats=str(emit_path)),
                               r13_source_result=str(r13_path))
        original_worker.update({key: True for key in self.worker.PRIOR_STAGES})
        original_worker.update({key: 1 for key in self.worker.PRIOR_ATTEMPTS})
        source = dict(round='R14', status='artifact_budget_exceeded', model=self.worker.MODEL,
                      worker=original_worker, processes=[dict(command=[
                          '-m', 'pt_agent.experiments.r14_common_worker', '--frontend', str(frontend)])],
                      source_r13=dict(result=str(r13_path), run=str(r13_dir), artifacts=dict(model_info=str(saved))),
                      timing=dict(spans=[dict(event_id='prior:1', stage_id='calibrate',
                                             inclusive_ns=123, exclusive_ns=123)]))
        source_path = original_dir / 'result.json'
        source_path.write_text(json.dumps(source))
        work = self.directory / 'r14b/worker'
        selected = dict(source_result=source_path, program=program_path, count=count_path, emit_stats=emit_path,
                        r13_result=r13_path, model_info=saved, frontend=frontend, work_dir=work)
        cli = sum((['--' + key.replace('_', '-'), str(value)] for key, value in selected.items()), [])
        return SimpleNamespace(cli=cli, work=work, source=source, r13=r13, count=count,
                               emitted=emitted, paths=selected)

    def result(self, fixture):
        return json.loads((fixture.work / 'worker_result.json').read_text())

    def forbidden_stages(self):
        from pt_agent import analyzer, backends, toolchain
        from pt_agent.data.samples import LegacyCifar
        from pt_agent.planners import rule
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, name in ((analyzer, 'reference_forward'), (analyzer, 'calibrate'),
                             (rule, 'plan'), (backends, 'CountBackend'), (LegacyCifar, 'get'),
                             (toolchain, 'trace'), (toolchain, 'compile'), (toolchain, 'execute_cpu')):
            stack.enter_context(patch.object(target, name, side_effect=AssertionError('Forbidden stage: ' + name)))
        return stack

    def test_saved_statistics_require_all_four_original_records_and_original_stages(self):
        fixture = self.fixture()
        self.assertTrue(self.worker.validate_resume_source(
            fixture.source, fixture.count, fixture.emitted, fixture.r13)['passed'])
        for case in ('model', 'round', 'status', 'phase', 'prior_stage', 'prior_attempt',
                     'completed', 'r13', 'vgg_source', 'rule', 'count', 'worker_count', 'emit', 'worker_emit'):
            with self.subTest(case=case):
                source, count, emitted, r13 = copy.deepcopy((fixture.source, fixture.count, fixture.emitted, fixture.r13))
                if case == 'model': source['model'] = 'different'
                elif case == 'round': source['worker']['round'] = 'R14b'
                elif case == 'status': source['status'] = 'error'
                elif case == 'phase': source['worker']['failed_phase'] = 'emit_lower'
                elif case == 'prior_stage': source['worker']['reference_passed'] = False
                elif case == 'prior_attempt': source['worker']['calibration_attempts'] = 2
                elif case == 'completed': source['worker']['emit_completed'] = True
                elif case == 'r13': r13['worker']['roundtrip_passed'] = False
                elif case == 'vgg_source':
                    model = 'pt_agent.experiments.r13_common_models:vgg16_silu_avgpool'
                    source['model'] = source['worker']['model'] = model
                    r13['model'] = r13['worker']['model'] = model
                elif case == 'rule': source['worker']['rule_options']['deg'] = 31
                else:
                    target = {'count': count['stats'], 'worker_count': source['worker']['count_stats'],
                              'emit': emitted['stats'], 'worker_emit': source['worker']['emit_stats']}[case]
                    target['counts']['add'] += 1
                    target['ops_total'] += 1
                self.assertFalse(self.worker.validate_resume_source(source, count, emitted, r13)['passed'])

    def test_budget_keeps_original_and_vgg_defaults_and_uses_exact_new_allowance(self):
        from pt_agent.experiments import r14a_emit_worker as vgg
        constants = {'k0': SimpleNamespace(nbytes=3 * 1024**3)}
        old = self.worker.previous._emit_budget('x', constants)
        new = self.worker._emit_budget('x', constants)
        self.assertEqual(old['limit_bytes'], 3221225472)
        self.assertFalse(old['within_budget'])
        self.assertEqual(new['limit_bytes'], 3758096384)
        self.assertEqual(vgg.EMIT_BYTES, 3489660928)
        vgg_observation = vgg._emit_budget('x', constants)
        self.assertEqual(vgg_observation['limit_bytes'], 3489660928)
        self.assertTrue(vgg_observation['within_budget'])
        between_limits = {'k0': SimpleNamespace(nbytes=3489660928)}
        self.assertFalse(vgg._emit_budget('x', between_limits)['within_budget'])
        self.assertTrue(self.worker._emit_budget('x', between_limits)['within_budget'])
        self.assertTrue(new['within_budget'])
        for key in ('text_bytes', 'array_bytes', 'entry_overhead_bytes', 'global_overhead_bytes', 'estimated_bytes'):
            self.assertEqual(new[key], old[key])
        excess = self.worker._emit_budget('x', {'k0': SimpleNamespace(nbytes=3758096384)})
        self.assertFalse(excess['within_budget'])

    def test_one_small_emit_releases_arrays_before_lightweight_readback(self):
        from pt_agent import backends, lower as lowering
        fixture = self.fixture()
        forbidden = self.forbidden_stages()
        base, original_lower = backends.EmitBackend, lowering.lower
        readback = self.worker.previous._readback_emit
        observed = dict(construct=0, lower=0, write=0, readback=0, backends=[], arrays=[])

        class ObservedEmit(base):
            def __init__(backend, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observed['construct'] += 1
                observed['backends'].append(weakref.ref(backend))

            def render(backend):
                text, constants = super().render()
                observed['arrays'].extend(weakref.ref(array) for array in constants.values())
                return text, constants

            def write(backend, directory):
                observed['write'] += 1
                return super().write(directory)

        def lower(*args, **kwargs):
            observed['lower'] += 1
            self.assertEqual(kwargs, dict(act_mode='program'))
            return original_lower(*args, **kwargs)

        def metadata_readback(directory, expected):
            observed['readback'] += 1
            self.assertTrue(observed['arrays'])
            self.assertTrue(all(reference() is None for reference in observed['backends']))
            self.assertTrue(all(reference() is None for reference in observed['arrays']))
            self.assertTrue(all(isinstance(value, SimpleNamespace) for value in expected.values()))
            self.assertTrue(all(set(vars(value)) == {'dtype', 'shape'} for value in expected.values()))
            return readback(directory, expected)

        with forbidden, patch.object(backends, 'EmitBackend', ObservedEmit), \
             patch.object(lowering, 'lower', lower), \
             patch.object(self.worker.previous, '_readback_emit', metadata_readback):
            code = self.worker.main(fixture.cli)
        result = self.result(fixture)
        self.assertEqual(code, 0, {key: result.get(key) for key in ('status', 'error', 'failed_phase')})
        self.assertEqual({key: observed[key] for key in ('construct', 'lower', 'write', 'readback')},
                         dict(construct=1, lower=1, write=1, readback=1))
        for key in self.worker.REQUIRED_STAGES: self.assertIs(result[key], True)
        for key in ('reference_passed', 'calibration_completed', 'plan_completed', 'count_completed',
                    'compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance'):
            self.assertIsNone(result[key])
        for key in ('reference_attempts', 'calibration_attempts', 'plan_attempts', 'count_attempts',
                    'llm_calls', 'real_ckks_attempts', 'real_ckks_completed', 'trace_runs', 'compiler_runs',
                    'keygen_runs', 'model_imports', 'downloads', 'installations'):
            self.assertEqual(result[key], 0)
        self.assertEqual(result['round'], 'R14b')
        self.assertEqual(result['model'], self.worker.MODEL)
        self.assertEqual(result['emit_attempts'], 1)
        self.assertTrue(result['constants_released_before_readback'])
        self.assertFalse(result['source_resume']['included_in_current_turnaround'])
        self.assertEqual(result['source_resume']['prior_timing'], fixture.source['timing'])
        self.assertTrue(all(value is True for value in result['source_resume']['prior_stages'].values()))
        for key in self.worker.STAT_FIELDS:
            self.assertEqual(result['count_stats'][key], fixture.count['stats'][key])
            self.assertEqual(result['count_stats'][key], result['emit_stats'][key])
        metadata = json.loads(Path(result['artifacts']['emit_metadata']).read_text())
        self.assertTrue(metadata['source_parsed'])
        self.assertFalse(metadata['source_executed'])
        self.assertTrue(all(row['finite'] for row in metadata['constants']))
        self.assertEqual(result['case_artifact_bytes'], self.worker.previous._case_size(fixture.work.parent))
        from pt_agent.experiments.r10_paper_metrics import read_events, summarize_events
        timing = summarize_events(read_events(fixture.work / 'events.jsonl'))
        self.assertEqual(timing['diagnostics'], [])
        root = next(span for span in timing['spans'] if span['parent_id'] is None)
        self.assertEqual(sum(span['exclusive_ns'] for span in timing['spans']), root['inclusive_ns'])
        self.assertNotIn('prior:1', {span['event_id'] for span in timing['spans']})
        self.assertFalse({'calibrate', 'count_lower', 'rule_plan', 'reference_check'} &
                         {span['stage_id'] for span in timing['spans']})

    def test_invalid_path_is_rejected_before_loading_model_or_lowering(self):
        from pt_agent.models.info import ModelInfo
        fixture = self.fixture()
        extra = self.directory / 'outside.ptdsl'
        extra.write_text(fixture.paths['program'].read_text())
        fixture.cli[fixture.cli.index('--program') + 1] = str(extra)
        with patch.object(ModelInfo, 'load', side_effect=AssertionError('Invalid source loaded weights')) as load:
            self.assertEqual(self.worker.main(fixture.cli), 1)
        load.assert_not_called()
        result = self.result(fixture)
        self.assertEqual(result['status'], 'source_failed')
        self.assertEqual(result['emit_attempts'], 0)

    def test_ambiguous_original_frontend_command_is_rejected(self):
        fixture = self.fixture()
        fixture.source['processes'][0]['command'] += ['--frontend', str(fixture.paths['frontend'])]
        fixture.paths['source_result'].write_text(json.dumps(fixture.source))
        self.assertEqual(self.worker.main(fixture.cli), 1)
        self.assertEqual(self.result(fixture)['status'], 'source_failed')
        self.assertEqual(self.result(fixture)['emit_attempts'], 0)

    def test_saved_count_disagreement_stops_before_model_load(self):
        from pt_agent.models.info import ModelInfo
        fixture = self.fixture()
        fixture.count['stats']['counts']['add'] += 1
        fixture.count['stats']['ops_total'] += 1
        fixture.paths['count'].write_text(json.dumps(fixture.count))
        with patch.object(ModelInfo, 'load', side_effect=AssertionError('Invalid Count loaded weights')) as load:
            self.assertEqual(self.worker.main(fixture.cli), 1)
        load.assert_not_called()
        result = self.result(fixture)
        self.assertEqual(result['status'], 'source_failed')
        self.assertFalse(result['source_checked'])

    def test_static_failure_prevents_emit(self):
        from pt_agent import backends
        from pt_agent.dsl import typecheck
        fixture = self.fixture()
        with patch.object(typecheck, 'check', return_value=SimpleNamespace(ok=False, diags=[])), \
             patch.object(backends, 'EmitBackend', side_effect=AssertionError('Emit after static error')) as emit:
            self.assertEqual(self.worker.main(fixture.cli), 1)
        emit.assert_not_called()
        result = self.result(fixture)
        self.assertEqual(result['status'], 'typecheck_failed')
        self.assertFalse(result['typecheck_passed'])
        self.assertEqual(result['emit_attempts'], 0)

    def test_original_header_and_approximation_are_checked_without_replanning(self):
        from pt_agent.dsl.parser import parse
        fixture = self.fixture()
        program = parse(fixture.paths['program'].read_text())
        self.assertTrue(self.worker._strategy_check(program)['passed'])
        changed = replace(program, header=replace(program.header, slots=8192))
        self.assertFalse(self.worker._strategy_check(changed)['passed'])
        operation = next(op for op in program.ops if op.kind == 'act')
        altered = replace(operation, attrs={**operation.attrs, 'deg': 31})
        changed = replace(program, ops=tuple(altered if op is operation else op for op in program.ops))
        self.assertFalse(self.worker._strategy_check(changed)['passed'])

    def test_new_emit_statistics_disagreement_prevents_write(self):
        from pt_agent.backends import EmitBackend
        fixture = self.fixture()
        original = EmitBackend.stats

        def changed(backend):
            statistics = original(backend)
            statistics['counts']['add'] += 1
            statistics['ops_total'] += 1
            return statistics

        with patch.object(EmitBackend, 'stats', changed), \
             patch.object(EmitBackend, 'write', side_effect=AssertionError('Write after mismatch')) as write:
            self.assertEqual(self.worker.main(fixture.cli), 1)
        write.assert_not_called()
        result = self.result(fixture)
        self.assertEqual(result['status'], 'count_emit_mismatch', result.get('error'))
        self.assertFalse(result['stats_equal'])
        self.assertEqual(result['emit_attempts'], 1)
        self.assertIsNone(result['emit_completed'])

    def test_export_budget_failure_preserves_partial_stats_and_prevents_write(self):
        from pt_agent.backends import EmitBackend
        fixture = self.fixture()
        budget = self.worker._emit_budget('x', {'k0': SimpleNamespace(nbytes=self.worker.EMIT_BYTES)})
        with patch.object(self.worker, '_emit_budget', return_value=budget), \
             patch.object(EmitBackend, 'write', side_effect=AssertionError('Oversize write')) as write:
            self.assertEqual(self.worker.main(fixture.cli), 1)
        write.assert_not_called()
        result = self.result(fixture)
        self.assertEqual(result['status'], 'artifact_budget_exceeded', result.get('error'))
        self.assertEqual(result['failed_phase'], 'emit_write')
        self.assertTrue(result['stats_equal'])
        self.assertIsNone(result['emit_completed'])
        self.assertFalse((fixture.work / 'emit/constants.npz').exists())

    def fake_cli(self, work):
        return sum((['--' + name, str(work if name == 'work-dir' else self.directory / name)]
                    for name in ('source-result', 'program', 'count', 'emit-stats', 'r13-result',
                                 'model-info', 'frontend', 'work-dir')), [])

    def test_terminal_case_budget_includes_result_itself_and_retains_partial_file(self):
        work = self.directory / 'case/worker'

        def partial(args, recorder, result):
            (args.work_dir / 'partial.constants').write_bytes(b'0' * 4096)

        with patch.object(self.worker, 'resume_emit', partial), patch.object(self.worker, 'CASE_BYTES', 4096):
            self.assertEqual(self.worker.main(self.fake_cli(work)), 1)
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'artifact_budget_exceeded')
        self.assertEqual(result['case_artifact_bytes'], self.worker.previous._case_size(work.parent))
        self.assertFalse(result['case_budget_observations'][-1]['within_budget'])
        self.assertTrue((work / 'partial.constants').exists())

    def test_memory_and_timeout_failures_keep_zero_attempts_and_null_metrics(self):
        for index, (failure, status) in enumerate(((MemoryError('fixture allocation'), 'resource_limit'),
                (self.worker.WorkerFailure('fixture expiry', status='timeout'), 'timeout'))):
            work = self.directory / ('failure' + str(index)) / 'worker'
            with self.subTest(status=status), patch.object(self.worker, 'resume_emit', side_effect=failure), \
                 patch.object(self.worker.signal, 'alarm') as alarm:
                self.assertEqual(self.worker.main(self.fake_cli(work)), 1)
            self.assertEqual([call.args for call in alarm.call_args_list], [(1800,), (0,)])
            result = json.loads((work / 'worker_result.json').read_text())
            self.assertEqual(result['status'], status)
            self.assertEqual(result['emit_attempts'], 0)
            self.assertIsNone(result['lowering_ready'])
            self.assertIsNone(result['predicted_latency'])
            self.assertIsNone(result['CKKS_latency'])
            self.assertEqual(result['llm_calls'], 0)
            self.assertEqual(result['real_ckks_attempts'], 0)

    def test_missing_source_retains_dependency_failure_and_existing_result_is_preserved(self):
        work = self.directory / 'missing/worker'
        self.assertEqual(self.worker.main(self.fake_cli(work)), 1)
        result = json.loads((work / 'worker_result.json').read_text())
        self.assertEqual(result['status'], 'dependency_missing')
        self.assertEqual(result['emit_attempts'], 0)
        with self.assertRaises(FileExistsError):
            self.worker.main(self.fake_cli(work))


if __name__ == '__main__':
    unittest.main()
