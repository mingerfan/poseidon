"""Fresh-clock integration with tiny metadata files and fake R10 children."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import dacapo_fresh_cold as entry
from pt_agent.experiments import dacapo_original_compile as original
from pt_agent.tests.test_dacapo_original_compile import fixtures, plan_fixture


class FreshOriginalColdChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('subprocess.Popen', 'socket.getaddrinfo', 'socket.socket.connect'):
            blocker = patch(name, side_effect=AssertionError('No real child or network in fresh-cold tests'))
            blocker.start();self.addCleanup(blocker.stop)
        blocker = patch('os.kill')
        blocker.start();self.addCleanup(blocker.stop)

    def arguments(self, model='vgg16'):
        upstream = self.root/'upstream'
        selected = original.model_paths(upstream, model)
        compiler = self.root/'build/bin/hecate-opt'
        frontend = self.root/'frontend'
        poly = upstream/'python/poly'
        einops = self.root/'deps'
        r13 = self.root/'r13.json'
        for path in (selected['script'], selected['checkpoint'], selected['model_source'], compiler,
                     compiler.parent.parent/'lib/libHecateFrontend.so', frontend/'python/hecate/hecate/expr.py',
                     poly/'poly/Poly.py', poly/'poly/Func.py', poly/'poly/MPCB.py', einops/'einops/__init__.py',
                     *(poly/'poly/data'/name for name in original.POLY_FILES)):
            path.parent.mkdir(parents=True, exist_ok=True);path.write_text('small public fixture\n')
        r13.write_text(json.dumps(dict(round='R13', status='ok', model=selected['factory'],
            worker=dict(model_source=dict(factory=selected['factory'], checkpoint=str(selected['checkpoint']),
                original_source=str(selected['model_source']), training_preprocessing_verified=False)))))
        profile, imported = fixtures()
        (upstream/'profiled_HEAAN_CPU.json').write_text(json.dumps(profile))
        imported_path = self.root/'imported.json';imported_path.write_text(json.dumps(imported))
        template_path = self.root/'template.json'
        # This public opaque source marker is never computed or compared as a digest.
        template_path.write_text(json.dumps(['unused', '--runtime-plan-format=json',
            '--runtime-plan-bundle-format=files', '--runtime-plan-operator-spec-sha256=sha256:'+'a'*64]))
        return SimpleNamespace(out=self.root/'case', model=model, upstream=upstream,
            source=selected['script'], checkpoint=selected['checkpoint'], compiler=compiler,
            frontend=frontend, poly_root=poly, einops=einops, intake_python=self.root/'fixture-python',
            source_r13_result=r13, compiler_profile=upstream/'profiled_HEAAN_CPU.json',
            imported_spec=imported_path, metadata_template=template_path, registry=self.root/'registry.jsonl',
            waterline=40, memory_gib=12, file_gib=16, case_gib=32, planning_gib=24,
            trace_seconds=1800, compile_seconds=7200, diagnostic_elide_final_ir=True)

    @staticmethod
    def observation(options, exit_code=0, **changes):
        now = time.perf_counter_ns()
        result = dict(exit_code=exit_code, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
            started_ns=now, ended_ns=now+1, limits=dict(memory_gib=options['mem_gib'],
                file_gib=options.get('fsize_gib', 8), wall_seconds=options['timeout']))
        result.update(changes);return result

    def run_fake(self, model='vgg16', *, failure=None, via_main=False):
        args = self.arguments(model)
        original_functions = {name: getattr(original, name) for name in
                              ('main', 'run', '_nix_worker', '_budget_guard', '_register')}
        calls = [];runtime = dict(native_reached=False)
        def process(command, **options):
            calls.append(dict(command=list(command), options=dict(options), observed_ns=time.perf_counter_ns()))
            Path(options['log']).write_text('synthetic child\n')
            Path(options['resources']).write_text('')
            if command[:2] == ['git', 'rev-parse']:
                Path(options['log']).write_text('public-fixture-revision\n')
                return self.observation(options)
            if '--worker-trace' in command:
                if failure == 'exception':
                    raise RuntimeError('synthetic trace supervision failure')
                work = Path(entry._one_argument(command, '--worker-dir'))
                target = work/'traced'/(original.MODEL_SOURCES[model]['entry']+'.mlir')
                target.parent.mkdir(parents=True, exist_ok=True);target.write_text('fixture fresh trace\n')
                failed = failure == 'trace'
                (work/'worker_result.json').write_text(json.dumps(dict(status='error' if failed else 'ok',
                    trace_attempts=1, trace_completed=0 if failed else 1,
                    mlir=original._metadata(target), file_gib=16)))
                return self.observation(options, exit_code=1 if failed else 0,
                    remaining_pids=[999999] if failure == 'remaining' else [])
            if '--worker-compile' in command:
                if failure == 'compile_worker_missing':
                    raise RuntimeError('synthetic compile supervisor failure before worker record')
                # Run the actual original worker parser and compile orchestration;
                # only its R10 native process is fake. No model/trace worker runs.
                inner = command[command.index('--')+1:]
                self.assertEqual(inner[:2], ['-m', 'pt_agent.experiments.dacapo_original_compile'])
                code = original.main(inner[2:])
                return self.observation(options, exit_code=code)
            self.assertEqual(command[0], str(args.compiler.resolve()))
            runtime['native_reached'] = True
            target = Path(command[command.index('-o')+1]);target.write_text('fixture native output\n')
            plan = plan_fixture()
            plan['plaintext_bundle'] = dict(id='small-public-bundle', version=1)
            plan['initialization'] = [dict(payload=dict(kind='bundle', content='public:data'))]
            plan_path = target.with_name('lowered._hecate_'+original.MODEL_SOURCES[model]['entry']+'.runtime-plan.json')
            plan_path.write_text(json.dumps(plan))
            bundle = plan_path.with_name(plan_path.name.removesuffix('.runtime-plan.json')+'.bundle')
            (bundle/'data').mkdir(parents=True)
            (bundle/'data/data.bin').write_bytes(b'fixture1')
            (bundle/'manifest.json').write_text(json.dumps(dict(bundle_format_version=1, version=1,
                bundle_id='small-public-bundle', blobs=[dict(content='public:data', byte_length=8)])))
            return self.observation(options, exit_code=1 if failure in ('native', 'native_guard') else 0)
        previous_handler = signal.getsignal(signal.SIGTERM)
        with ExitStack() as stack:
            stack.enter_context(patch.object(paths, 'RESULTS_DIR', self.root))
            stack.enter_context(patch.object(original, 'run_process', side_effect=process))
            stack.enter_context(patch.object(original.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000*entry.GIB)))
            stack.enter_context(patch.object(entry.supervision, 'available_ram', return_value=16*entry.GIB))
            def latest_resource(path):
                if failure == 'admission_guard' or (failure in ('native_guard', 'native_guard_completed')
                                                   and runtime['native_reached']):
                    return dict(subtree_plus_supervisor_rss_bytes=17*entry.GIB, sample_ts_ns=1)
                return None
            stack.enter_context(patch.object(entry.supervision, 'latest_resource', side_effect=latest_resource))
            stack.enter_context(patch.dict(os.environ, {'HECATE_PYTHON_LIBRARY_PATH': '/fixture/lib'}))
            if via_main:
                argv = []
                for name, value in vars(args).items():
                    if name == 'diagnostic_elide_final_ir':
                        argv.append('--diagnostic-elide-final-ir')
                    else:
                        argv += ['--'+name.replace('_', '-'), str(value)]
                code = entry.main(argv)
            else:
                code = entry.run(args)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)
        for name, function in original_functions.items():
            self.assertIs(getattr(original, name), function)
        result = json.loads((args.out/'result.json').read_text())
        completion = json.loads((args.out/'completion.json').read_text())
        return args, code, calls, result, completion

    def assert_complete(self, model, *, via_main=False):
        args, code, calls, result, closed = self.run_fake(model, via_main=via_main)
        self.assertEqual(code, 0);self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['round'], original.ROUND);self.assertEqual(result['mode'], original.MODE)
        self.assertTrue(result['fresh_compile_admission']['passed'])
        self.assertTrue(result['cold_end_to_end_run']);self.assertTrue(result['cold_trace_compile_completed'])
        self.assertFalse(result['original_trace_reused']);self.assertFalse(result['prior_times_included'])
        self.assertEqual([row['phase'] for row in result['resource_adapter']['phases']], ['trace', 'compile'])
        outer = [row for row in calls if entry._phase_from_command(row['command'])]
        native = [row for row in calls if row['command'][0] == str(args.compiler.resolve())]
        selected = entry.POLICIES[model]
        self.assertEqual([row['options']['mem_gib'] for row in outer], [12, selected['native_AS_gib']])
        self.assertEqual([row['options']['fsize_gib'] for row in outer], [16, 16])
        self.assertEqual([row['options']['timeout'] for row in outer], [1920, 7320])
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0]['options']['mem_gib'], selected['native_AS_gib'])
        self.assertEqual(native[0]['options']['fsize_gib'], 16)
        self.assertEqual(native[0]['options']['timeout'], 7200)
        self.assertEqual(args.memory_gib, 12)
        self.assertEqual(result['limits']['subtree_plus_supervisor_RSS_gib'], selected['RSS_gib'])
        self.assertEqual(result['limits']['minimum_available_RAM_gib'], 1.25)
        self.assertEqual(result['api_calls'], 0);self.assertEqual(result['real_ckks_attempts'], 0)
        for name in original.FORMAL_FIELDS:
            self.assertIsNone(result[name])
        rows = [json.loads(line) for line in args.registry.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual(rows[0]['mode'], original.MODE)
        self.assertEqual(closed['status'], 'ok');self.assertTrue(closed['registry_persisted'])
        self.assertEqual(closed['case_artifact_bytes'], original.artifact_bytes(args.out))
        self.assertGreater(closed['complete_case_elapsed_seconds'], 0)
        self.assertLessEqual(closed['started_ns'], min(row['observed_ns'] for row in calls))
        self.assertGreaterEqual(closed['ended_ns'], max(row['observed_ns'] for row in calls))
        command = json.loads((args.out/'compile-argv.json').read_text())
        expected = original.compiler_argv(json.loads(args.metadata_template.read_text()), binary=args.compiler,
            source=args.out/'work/trace'/original.model_paths(args.upstream, model)['trace_relative'],
            profile=args.out/'compiler-config.json', output=args.out/'work/compile/lowered.mlir',
            waterline=40, diagnostic_elide_final_ir=True)
        self.assertEqual(command, expected)
        self.assertEqual(native[0]['command'], expected)
        self.assertTrue((args.out/'source/dacapo_original_compile.py').is_file())
        self.assertTrue((args.out/'source/dacapo_fresh_cold.py').is_file())
        events = [json.loads(line) for line in (args.out/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['stage_id'] for event in events if event['type'] == 'begin'],
                         ['source.preflight', 'original.trace', 'original.compile.optimized', 'original.export.inspect'])

    def test_vgg_complete_original_run_uses12_then24_and_same_registry_completion(self):
        self.assert_complete('vgg16')

    def test_alex_complete_original_main_worker_parsers_use12_then32(self):
        self.assert_complete('alexnet', via_main=True)

    def test_trace_failure_preserves_one_attempt_and_never_compiles(self):
        _, code, calls, result, closed = self.run_fake(failure='trace')
        self.assertEqual(code, 1);self.assertEqual(result['trace_attempts'], 1)
        self.assertEqual(result['trace_completed'], 0);self.assertEqual(result['compile_attempts'], 0)
        self.assertEqual([row['phase'] for row in result['resource_adapter']['phases']], ['trace'])
        self.assertEqual(closed['status'], 'error');self.assertFalse(result['cold_trace_compile_completed'])
        self.assertFalse(any('--worker-compile' in row['command'] for row in calls))

    def test_native_failure_keeps_fresh_trace_and_has_no_second_native_attempt(self):
        _, code, calls, result, closed = self.run_fake('alexnet', failure='native')
        self.assertEqual(code, 1);self.assertEqual(result['trace_completed'], 1)
        self.assertEqual(result['native_compile_attempts'], 1);self.assertEqual(result['native_compile_completed'], 0)
        self.assertEqual(len([row for row in calls if '--worker-compile' in row['command']]), 1)
        self.assertEqual(result['limits']['native_AS_gib'], 32)
        self.assertEqual(closed['status'], 'error')

    def test_supervision_exception_stops_its_watcher_and_persists_resources_before_registry(self):
        args, code, _, result, closed = self.run_fake(failure='exception')
        self.assertEqual(code, 1);self.assertEqual(closed['status'], 'error')
        self.assertIn('synthetic trace supervision failure', result['error'])
        self.assertTrue(result['resource_adapter']['phases'][0]['watcher_stopped'])
        self.assertEqual(result['limits']['trace_AS_gib'], 12)
        self.assertEqual(json.loads(args.registry.read_text())['metrics']['status'], result['status'])

    def test_post_process_guard_keeps_saved_native_attempt_and_completion_counters(self):
        args, code, _, result, closed = self.run_fake(failure='native_guard')
        self.assertEqual(code, 1);self.assertEqual(result['status'], 'resource_stopped')
        self.assertEqual(result['native_compile_attempts'], 1)
        self.assertEqual(result['native_compile_completed'], 0)
        self.assertEqual(result['compile_worker']['compile_attempts'], 1)
        self.assertEqual(result['compile_process']['exit_code'], 1)
        self.assertTrue(result['resource_adapter']['phases'][1]['current_worker_record_available'])
        registry = json.loads(args.registry.read_text())
        self.assertEqual(registry['metrics']['native_compile_attempts'], 1)
        self.assertEqual(registry['metrics']['native_compile_completed'], 0)
        self.assertEqual(closed['status'], 'resource_stopped')

    def test_guard_after_completed_native_keeps1_1_and_overall_resource_failure(self):
        _, code, _, result, closed = self.run_fake(failure='native_guard_completed')
        self.assertEqual(code, 1);self.assertEqual(result['status'], 'resource_stopped')
        self.assertEqual(result['native_compile_attempts'], 1)
        self.assertEqual(result['native_compile_completed'], 1)
        self.assertEqual(result['compile_completed'], 0)
        self.assertFalse(result['cold_trace_compile_completed'])
        self.assertEqual(closed['status'], 'resource_stopped')

    def test_started_compile_without_worker_record_retains_unknown_native_counts(self):
        args, code, _, result, closed = self.run_fake(failure='compile_worker_missing')
        self.assertEqual(code, 1);self.assertEqual(result['status'], 'error')
        self.assertEqual(result['compile_attempts'], 1)
        self.assertIsNone(result['native_compile_attempts']);self.assertIsNone(result['native_compile_completed'])
        registry = json.loads(args.registry.read_text())
        self.assertIsNone(registry['metrics']['native_compile_attempts'])
        self.assertFalse(result['resource_adapter']['phases'][1]['current_worker_record_available'])
        self.assertEqual(closed['status'], 'error')

    def test_admission_guard_starts_no_trace_worker_and_retains_failure_completion(self):
        _, code, calls, result, closed = self.run_fake(failure='admission_guard')
        self.assertEqual(code, 1);self.assertEqual(result['status'], 'resource_stopped')
        self.assertFalse(any('--worker-trace' in row['command'] for row in calls))
        self.assertFalse(result['resource_adapter']['phases'][0]['process_launched'])
        self.assertEqual(result['native_compile_attempts'], 0)
        self.assertEqual(closed['status'], 'resource_stopped')

    def test_remaining_child_observation_rejects_without_retry(self):
        _, code, _, result, _ = self.run_fake(failure='remaining')
        self.assertEqual(code, 1);self.assertEqual(result['compile_attempts'], 0)
        self.assertEqual(result['trace_process']['remaining_pids'], [999999])

    def test_physical_guard_boundaries_preserve14_or16_and_available1point25(self):
        for model, values in entry.POLICIES.items():
            policy = dict(model=model, **values)
            sample = dict(subtree_plus_supervisor_rss_bytes=values['RSS_gib']*entry.GIB, sample_ts_ns=1)
            self.assertFalse(entry.physical_observation(policy, sample, 1.25*entry.GIB, 10*entry.GIB)['stop'])
            self.assertTrue(entry.physical_observation(policy, dict(sample,
                subtree_plus_supervisor_rss_bytes=sample['subtree_plus_supervisor_rss_bytes']+1),
                16*entry.GIB, 10*entry.GIB)['stop'])
            self.assertTrue(entry.physical_observation(policy, sample, 1.25*entry.GIB-1, 10*entry.GIB)['stop'])
            self.assertTrue(entry.physical_observation(policy, sample, 16*entry.GIB, 10*entry.GIB-1)['stop'])

    def test_phase_args_are_copies_and_original_worker_target_keeps_two_module_flags(self):
        args = self.arguments('alexnet')
        with patch.object(paths, 'RESULTS_DIR', self.root):
            policy = entry.resource_policy(args)
        command = entry.phase_worker(args, 'compile', self.root/'worker',
            command_file=self.root/'argv.json', policy=policy)
        self.assertEqual(args.memory_gib, 12)
        self.assertEqual(command.count('-m'), 2)
        self.assertEqual(entry._one_argument(command, '--memory-gib'), '32')
        self.assertIn('pt_agent.experiments.dacapo_original_compile', command)
        self.assertEqual(entry._one_argument(command, '--file-gib'), '16')

    def test_only_plain_entry_functions_are_cloned_and_contextmanager_globals_stay_original(self):
        namespace = dict(vars(original))
        clone = entry._ordinary(original.run, namespace)
        self.assertIs(clone.__code__, original.run.__code__)
        self.assertIs(clone.__globals__, namespace)
        self.assertIs(namespace['_budget_guard'], original._budget_guard)
        self.assertIs(namespace['_budget_guard'].__globals__, original._budget_guard.__globals__)
        self.assertIn('_GeneratorContextManager', original._budget_guard.__globals__)

    def test_inconsistent_or_unsupported_resources_reject_before_case_or_children(self):
        args = self.arguments()
        for name, value in (('model', 'resnet20'), ('memory_gib', 24), ('file_gib', 8),
                            ('case_gib', 16), ('planning_gib', 12), ('waterline', 45),
                            ('trace_seconds', 1801), ('compile_seconds', 7201),
                            ('diagnostic_elide_final_ir', False)):
            with self.subTest(name=name), patch.object(paths, 'RESULTS_DIR', self.root):
                changed = SimpleNamespace(**vars(args));setattr(changed, name, value)
                with self.assertRaises(ValueError):
                    entry.resource_policy(changed)
        self.assertFalse(args.out.exists())

    def test_wrapper_does_not_offer_a_worker_bypass(self):
        for flag in ('--worker-trace', '--worker-compile'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'complete original outer'):
                entry.main([flag])


if __name__ == '__main__':
    unittest.main()
