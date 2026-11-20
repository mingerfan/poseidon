"""Small standard-library fake checks; no model/toolchain/NumPy imports."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import r29_local_compile_matrix as matrix


def template():
    return ['{hecate_opt}', '{input}', '--runtime-plan-format=json', '--runtime-plan-bundle-format=files',
            '--pars', '--ckks-config={package}/config/compiler-config.json', '--waterline=43',
            '--mlir-disable-threading', '--verify-each', '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-operator-spec-sha256=sha256:' + '0' * 64, '-o', '{output_dir}/lowered.mlir']


def agent_fixture(root):
    program, info, result_path = root / 'program.ptdsl', root / 'model_info.json', root / 'result.json'
    worker = dict(model='resnet20_silu', status='ok', frozen=True, selected_program=str(program),
                  source_model_info=str(info), quality_acceptance=dict(passed=True),
                  strict_math_probes=[dict(sample_id='test:' + str(9000 + i), exact=dict(passed=True),
                                          approximate=dict(passed=True)) for i in range(3)])
    result = dict(round='R28', status='ok', model='resnet20_silu', worker=worker)
    audit = dict(status='quality_accepted', formal_frozen=True, quality_acceptance_passed=True,
                 strict_math_passed=True, model='resnet20_silu', source_result=str(result_path), selected_program=str(program))
    return result, audit, result_path, info


class LocalCompileMatrixTests(unittest.TestCase):
    def test_debug_snapshots_are_off_by_default_and_explicitly_optional(self):
        source=template();source.insert(source.index('--waterline=43')+1,'--enable-debug-printer')
        for mode in ('pars','dacapo-optimized','dacapo-depth-dp'):
            kwargs=dict(binary='compiler',source='input',profile='profile',output='output')
            default=matrix.compiler_argv(source,mode,**kwargs)
            enabled=matrix.compiler_argv(source,mode,debug_snapshots=True,**kwargs)
            self.assertNotIn('--enable-debug-printer',default)
            self.assertEqual(enabled.count('--enable-debug-printer'),1)
            self.assertEqual([arg for arg in enabled if arg!='--enable-debug-printer'],default)
        self.assertIn('--enable-debug-printer',source)

    def test_registry_identity_includes_queue_and_system(self):
        first=matrix.case_run_id(Path('/results/queue-one/dacapo-depth-dp'),'dacapo-depth-dp')
        second=matrix.case_run_id(Path('/results/queue-two/dacapo-depth-dp'),'dacapo-depth-dp')
        agent=matrix.case_run_id(Path('/results/queue-one/agent-pars'),'agent-pars')
        self.assertEqual(first,'queue-one-dacapo-depth-dp')
        self.assertEqual(second,'queue-two-dacapo-depth-dp')
        self.assertEqual(len({first,second,agent}),3)

    def test_saved_metadata_and_paths_no_digest_computation(self):
        original = template()
        argv = matrix.compiler_argv(original, 'pars', binary='/tools/hecate-opt', source='/case/input.mlir',
                                    profile='/tools/profile.json', output='/case/lowered.mlir')
        self.assertEqual(argv[:2], ['/tools/hecate-opt', '/case/input.mlir'])
        self.assertIn('--ckks-config=/tools/profile.json', argv)
        self.assertIn('--runtime-plan-operator-spec-sha256=sha256:' + '0' * 64, argv)
        self.assertEqual(argv[-1], '/case/lowered.mlir')
        self.assertEqual(original, template())

    def test_optimized_and_depth_are_separate_recipes(self):
        for mode, placement in (('dacapo-optimized', 'optimized'), ('dacapo-depth-dp', 'depth-dp')):
            argv = matrix.compiler_argv(template(), mode, binary='compiler', source='input', profile='profile', output='output')
            self.assertNotIn('--pars', argv)
            self.assertIn('--dacapo', argv)
            self.assertIn('--boot-placement=' + placement, argv)
            self.assertIn('--output-val=10', argv)

    def test_different_parameter_group_is_rejected(self):
        source = template()
        source[source.index('--waterline=43')] = '--waterline=45'
        with self.assertRaises(ValueError):
            matrix.compiler_argv(source, 'pars', binary='c', source='i', profile='p', output='o')

    def test_quality_audit_and_strict_pairs_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            result, audit, source, info = agent_fixture(Path(directory))
            self.assertTrue(matrix.validate_agent_source(result, audit, result_path=source, model_info=info)['independent_audit_passed'])
            for field in ('formal_frozen', 'quality_acceptance_passed', 'strict_math_passed'):
                changed = dict(audit, **{field: False})
                with self.assertRaises(ValueError):
                    matrix.validate_agent_source(result, changed, result_path=source, model_info=info)
            changed = deepcopy(result)
            changed['worker']['strict_math_probes'][1]['approximate']['passed'] = False
            with self.assertRaises(ValueError):
                matrix.validate_agent_source(changed, audit, result_path=source, model_info=info)

    def test_audit_for_another_program_or_model_info_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, audit, source, info = agent_fixture(Path(directory))
            for changed in (dict(audit, selected_program=str(Path(directory) / 'other.ptdsl')),
                            dict(audit, source_result=str(Path(directory) / 'other-result.json'))):
                with self.assertRaises(ValueError):
                    matrix.validate_agent_source(result, changed, result_path=source, model_info=info)
            with self.assertRaises(ValueError):
                matrix.validate_agent_source(result, audit, result_path=source, model_info=info.parent / 'other-info.json')

    def test_plan_positions_and_latency_units(self):
        plan = dict(values=[dict(id='x', level=1), dict(id='b', level=5), dict(id='y', level=4)],
                    execution=[dict(kind='compute', op='boot', id='boot0', inputs=['x'], outputs=['b']),
                               dict(kind='compute', op='mul_cc', inputs=['b', 'b'], outputs=['y'])])
        profile = dict(latencyTable={'earth.bootstrap_single': [0, 20, 0, 0, 0, 0], 'earth.mul_double': [0, 1, 2, 3, 4, 5]})
        result = matrix.plan_observation(plan, profile)
        self.assertEqual(result['compiled_bootstraps'], 1)
        self.assertEqual(result['predicted_latency_us'], 25)
        self.assertEqual(result['bootstrap_positions'][0]['execution_index'], 0)
        self.assertEqual(result['bootstrap_positions'][0]['input_levels'], [1])
        self.assertFalse(result['prediction']['measured_local_ckks'])

    def test_missing_latency_does_not_invent_full_prediction(self):
        result = matrix.plan_observation(dict(values=[dict(id='x', level=10)],
                    execution=[dict(kind='compute', op='boot', inputs=['x'], outputs=['x'])]),
                    dict(latencyTable={'earth.bootstrap_single': [1]}))
        self.assertEqual(result['compiled_bootstraps'], 1)
        self.assertIsNone(result['predicted_latency_us'])
        self.assertEqual(len(result['prediction']['missing']), 1)

    def test_worker_timeout_arguments_are_independent(self):
        args = Namespace(compile_seconds=7200, trace_seconds=1800, model_info=Path('/explicit/info'),
                         metadata_template=Path('/explicit/argv'), profile=Path('/explicit/profile'),
                         frontend=Path('/explicit/frontend'), memory_gib=6, case_gib=4)
        for phase, expected in (('trace', 2040), ('compile', 7440)):
            argv, timeout = matrix._launch_worker(args, 'dacapo-optimized', phase, Path('/explicit/program'), Path('/explicit/work'))
            self.assertEqual(timeout, expected)
            self.assertIn('--worker-phase', argv)
            self.assertEqual(argv[argv.index('--compile-seconds') + 1], '7200')

    def test_failed_case_does_not_prevent_next_system(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'queue'
            args = Namespace(out=out, systems=['dacapo-optimized', 'dacapo-depth-dp', 'agent-pars'])
            def fake_process(argv, **kwargs):
                Path(kwargs['log']).write_text('fake-revision\n')
                return dict(exit_code=0)
            with patch.object(matrix, 'run_process', side_effect=fake_process), patch.object(matrix, 'run_case',
                    side_effect=[dict(status='error'), dict(status='ok'), dict(status='error')]) as cases:
                self.assertEqual(matrix.outer(args), 1)
            self.assertEqual(cases.call_count, 3)
            self.assertEqual([row['status'] for row in json.loads((out / 'queue-result.json').read_text())['results']],
                             ['error', 'ok', 'error'])

    def test_legacy_metadata_adapter_restores_builder_on_error(self):
        from pt_agent import toolchain
        before = toolchain.compile_argv
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with matrix.native_metadata_commands(template()):
                argv = toolchain.compile_argv('dacapo-depth-dp', 'qwen48', Path('.'))
                self.assertIn('--ckks-config=/out/boot-compiler-config.json', argv)
                raise RuntimeError('fixture')
        self.assertIs(toolchain.compile_argv, before)

    def test_case_records_three_phases_and_unique_eight_field_registry(self):
        from pt_agent import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program, info, profile, command = (root / name for name in ('rule.ptdsl', 'info.json', 'profile.json', 'argv.json'))
            for path in (program, info, profile, command):
                path.write_text('{}\n')
            prior = root / 'r17.json'
            prior.write_text(json.dumps(dict(round='R17', status='ok', model=matrix.MODEL,
                                worker=dict(heldout_acceptance=True, artifacts=dict(program=str(program))))))
            args = Namespace(rule_result=prior, model_info=info, profile=profile, metadata_template=command,
                             git_rev='fixture', case_gib=4, memory_gib=6, compile_seconds=7200, trace_seconds=1800,
                             frontend=root / 'frontend.py')
            calls = []
            def fake_process(argv, **kwargs):
                phase = argv[argv.index('--worker-phase') + 1]
                work = Path(argv[argv.index('--work-dir') + 1])
                body = dict(status='ok')
                if phase == 'compile':
                    body.update(compiled_bootstraps=2, predicted_latency_us=123)
                (work / (phase + '-result.json')).write_text(json.dumps(body))
                calls.append((phase, kwargs['timeout']))
                return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[])
            with patch.object(paths, 'RESULTS_DIR', root), patch.object(matrix, 'run_process', side_effect=fake_process):
                result = matrix.run_case(args, 'dacapo-optimized', root / 'case')
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(calls, [('emit', 2040), ('trace', 2040), ('compile', 7440)])
            self.assertEqual(result['compiled_bootstraps'], 2)
            self.assertIsNone(result['CKKS_latency'])
            self.assertEqual(result['completion']['case_artifact_bytes'], matrix._size(root / 'case'))
            rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]), 8)
            self.assertEqual(rows[0]['metrics'], result['metrics'])

    def test_unapproved_agent_case_never_launches_worker(self):
        from pt_agent import paths
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(agent_result=None, agent_audit=None, model_info=root / 'info.json',
                             git_rev='fixture', case_gib=4)
            with patch.object(paths, 'RESULTS_DIR', root), patch.object(matrix, 'run_process') as process:
                result = matrix.run_case(args, 'agent-pars', root / 'case')
            process.assert_not_called()
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['attempted'], {})
            self.assertIsNone(result['compiled_bootstraps'])

    def test_completion_endpoint_follows_required_artifacts_and_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'result.json').write_text('required-result\n')
            (root / 'registry-fixture.jsonl').write_text('required-registry\n')
            observed = []
            def clock():
                self.assertTrue((root / 'result.json').is_file())
                self.assertTrue((root / 'registry-fixture.jsonl').is_file())
                self.assertFalse((root / 'completion.json').exists())
                observed.append(True)
                return 1200000000
            timing = dict(status='timing_invalid', attribution_ready=False,
                          diagnostics=['overlapping fixture spans'], spans=[])
            marker = matrix.completion_marker(root, started_ns=200000000, status='error', timing=timing,
                                              required_artifact_bytes=matrix._size(root), clock=clock)
            self.assertEqual(observed, [True])
            self.assertEqual(marker['complete_case_elapsed_ns'], 1000000000)
            self.assertEqual(marker['complete_case_elapsed_seconds'], 1)
            self.assertEqual(marker['case_artifact_bytes'], matrix._size(root))
            self.assertEqual(marker['overlap_and_timing_diagnostics'], ['overlapping fixture spans'])
            self.assertFalse(marker['exclusive_attribution_ready'])


if __name__ == '__main__':
    unittest.main()
