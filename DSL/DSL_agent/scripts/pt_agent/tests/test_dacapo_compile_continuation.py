"""One saved synthetic trace, fake native commands, and real terminal records."""
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import signal
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_compile_continuation as entry
from pt_agent.tests import test_dacapo_diagnostic_output as prior


def clean():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
                limits=dict(memory_gib=24, file_gib=16, wall_seconds=7200))


def fixture(root):
    case = root / 'old-failed-case'
    case.mkdir()
    trace = case / 'work/trace/traced/VGG16.mlir'
    trace.parent.mkdir(parents=True)
    trace.write_text('unexecuted synthetic original trace')
    compiler = root / 'build/bin/compiler'
    compiler.parent.mkdir(parents=True)
    compiler.write_text('unexecuted compiler fixture')
    profile, imported = prior.fixtures()
    imported_path = root / 'original-imported-profile.json'
    imported_path.write_text(json.dumps(imported))
    profile_path = root / 'original-compiler-profile.json'
    profile_path.write_text(json.dumps(profile))
    contract, _ = entry.original.make_export_contract(imported, profile, waterline=40)
    (case / 'compiler-config.json').write_text(json.dumps(profile))
    (case / 'export-contract.json').write_text(json.dumps(contract))
    command = entry.original.compiler_argv(prior.template(), binary=compiler, source=trace,
        profile=case / 'compiler-config.json', output=case / 'work/compile/lowered.mlir',
        diagnostic_elide_final_ir=True)
    (case / 'compile-argv.json').write_text(json.dumps(command))
    failed = dict(clean(), exit_code=1)
    old = dict(round=entry.original.ROUND, status='error', model=entry.original.model_source('vgg16')['result_model'],
        source_checked=True, trace_attempts=1, trace_completed=1, compile_attempts=1, compile_completed=0,
        trace_process=clean(), compile_process=failed,
        trace_worker=dict(status='ok', trace_completed=1, mlir=entry.original._metadata(trace)),
        sources=dict(imported_spec=dict(path=str(imported_path)), compiler_profile=dict(path=str(profile_path))),
        diagnostic_elide_final_ir=True, error='prior AS12 allocation failed',
        trace_argv=['python', '-m', 'pt_agent.tools.nixrun', '--python', '/trusted/intake-python', '--', 'original'],
        compile_worker=dict(status='error', process=dict(clean(), exit_code=-6)))
    (case / 'result.json').write_text(json.dumps(old))
    (case / 'completion.json').write_text(json.dumps(dict(round=entry.original.ROUND, status='error',
        required_artifacts_persisted=True, registry_persisted=True, complete_case_elapsed_seconds=52.8)))
    return SimpleNamespace(source_case=case, out=root / 'new-case', registry=root / 'registry.jsonl',
        memory_gib=24, file_gib=16, case_gib=8, minimum_free_gib=4, compile_seconds=7200)


class DaCapoCompileContinuationChecks(unittest.TestCase):
    def patches(self, root, process):
        stack = ExitStack()
        stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', root))
        stack.enter_context(patch.object(entry, 'available_ram', return_value=4 * entry.GIB))
        stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=13 * entry.GIB)))
        stack.enter_context(patch.object(entry, 'run_process', side_effect=process))
        return stack

    def success(self, argv, **kwargs):
        self.assertEqual((kwargs['mem_gib'], kwargs['fsize_gib'], kwargs['timeout']), (24, 16, 7320))
        self.assertIn('--worker-compile', argv)
        self.assertNotIn('--worker-trace', argv)
        self.assertEqual(argv[argv.index('--memory-gib') + 1], '24')
        self.assertEqual(argv[argv.index('--file-gib') + 1], '16')
        command = json.loads(Path(argv[argv.index('--command-file') + 1]).read_text())
        work = Path(argv[argv.index('--worker-dir') + 1])
        self.assertEqual(command[command.index('-o') + 1], str(work / 'lowered.mlir'))
        self.assertIn('--waterline=40', command)
        self.assertEqual(command.count(entry.original.DIAGNOSTIC_ELIDE_OPTION), 1)
        prior.exported(work)
        (work / 'worker_result.json').write_text(json.dumps(dict(status='ok', compile_attempts=1,
            compile_completed=1, file_gib=16, process=clean())))
        Path(kwargs['log']).write_text('synthetic successful compile')
        Path(kwargs['resources']).write_text('')
        return clean()

    def test_source_gate_changes_only_output_and_never_reads_the_saved_trace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            trace = args.source_case / 'work/trace/traced/VGG16.mlir'
            original_read = Path.read_text
            def bounded_read(path, *a, **kw):
                if path == trace:
                    raise AssertionError('A saved large trace must remain stat-only')
                return original_read(path, *a, **kw)
            with patch.object(Path, 'read_text', bounded_read):
                record = entry.source_gate(args.source_case, args.out / 'emit/lowered.mlir')
            before, after = record['original_argv'], record['argv']
            self.assertEqual([i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]], [before.index('-o') + 1])
            self.assertFalse(record['trace_content_read'])
            self.assertFalse(record['original_case_modified'])

    def test_source_gate_rejects_partial_trace_changed_profile_flags_and_live_children(self):
        for defect in ('trace', 'profile', 'waterline', 'children', 'closure'):
            with self.subTest(defect=defect), TemporaryDirectory() as directory:
                args = fixture(Path(directory))
                if defect in ('trace', 'children'):
                    record = json.loads((args.source_case / 'result.json').read_text())
                    if defect == 'trace':
                        record['trace_completed'] = 0
                    else:
                        record['compile_process']['remaining_pids'] = [123]
                    (args.source_case / 'result.json').write_text(json.dumps(record))
                elif defect == 'profile':
                    profile = json.loads((args.source_case / 'compiler-config.json').read_text())
                    profile['polynomialDegree'] = 65536
                    (args.source_case / 'compiler-config.json').write_text(json.dumps(profile))
                elif defect == 'waterline':
                    command = json.loads((args.source_case / 'compile-argv.json').read_text())
                    command[command.index('--waterline=40')] = '--waterline=32'
                    (args.source_case / 'compile-argv.json').write_text(json.dumps(command))
                else:
                    (args.source_case / 'completion.json').unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    entry.source_gate(args.source_case, args.out / 'emit/lowered.mlir')

    def test_one_fake_compile_preserves_original_and_records_registry_completion_and_full_bundle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            old_result = json.loads((args.source_case / 'result.json').read_text())
            with self.patches(root, self.success):
                self.assertEqual(entry.run(args), 0)
            result = json.loads((args.out / 'result.json').read_text())
            self.assertEqual((result['compile_attempts'], result['compile_completed']), (1, 1))
            self.assertEqual((result['trace_attempts'], result['trace_completed']), (0, 0))
            self.assertTrue(result['original_trace_reused'])
            self.assertFalse(result['cold_end_to_end_run'])
            self.assertEqual(result['bootstrap_count'], 1)
            self.assertEqual(result['compiler_output']['plaintext_bundle']['blob_bytes'], 16)
            self.assertEqual(json.loads((args.source_case / 'result.json').read_text()), old_result)
            for key in ('api_calls', 'checkpoint_reads', 'model_forwards', 'report_samples', 'keygen_runs', 'encryptions', 'decryptions', 'real_ckks_attempts'):
                self.assertEqual(result[key], 0)
            rows = [json.loads(line) for line in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            completion = json.loads((args.out / 'completion.json').read_text())
            self.assertEqual(completion['status'], 'ok')
            self.assertTrue(completion['required_artifacts_persisted'])
            self.assertGreater(completion['complete_case_elapsed_ns'], 0)
            self.assertEqual(completion['case_artifact_bytes'], entry.artifact_bytes(args.out))

    def test_outer_as24_is_also_passed_to_existing_native_worker(self):
        with TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            record = entry.source_gate(args.source_case, args.out / 'emit/lowered.mlir')
            args.out.mkdir()
            command_path = args.out / 'compile-argv.json'
            command_path.write_text(json.dumps(record['argv']))
            worker_args = SimpleNamespace(worker_dir=args.out / 'emit', command_file=command_path,
                compiler=Path(record['compiler']), worker_seconds=7200, memory_gib=24, file_gib=16,
                diagnostic_elide_final_ir=True)
            with patch.dict('os.environ', {'HECATE_PYTHON_LIBRARY_PATH': '/trusted/lib'}), \
                    patch.object(entry.original, 'run_process', return_value=clean()) as native:
                self.assertEqual(entry.original.main(['--worker-compile', '--worker-dir', str(worker_args.worker_dir),
                    '--command-file', str(command_path), '--compiler', str(worker_args.compiler),
                    '--worker-seconds', '7200', '--memory-gib', '24', '--file-gib', '16',
                    '--diagnostic-elide-final-ir']), 0)
            self.assertEqual((native.call_args.kwargs['mem_gib'], native.call_args.kwargs['fsize_gib'], native.call_args.kwargs['timeout']), (24, 16, 7200))

    def test_missing_executable_payload_stops_after_completed_compile_and_preserves_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            def incomplete(argv, **kw):
                observation = self.success(argv, **kw)
                (args.out / 'emit/lowered._hecate_VGG16.bundle/manifest.json').unlink()
                return observation
            with self.patches(root, incomplete):
                self.assertEqual(entry.run(args), 1)
            result = json.loads((args.out / 'result.json').read_text())
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['compile_completed'], 1)
            self.assertEqual(result['native_compile_completed'], 1)
            self.assertTrue((args.out / 'emit/lowered.mlir').is_file())
            self.assertTrue((args.out / 'emit/lowered._hecate_VGG16.runtime-plan.json').is_file())

    def test_unexpected_supervisor_error_preserves_launch_partial_and_stops(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            calls = []
            def failed(argv, **kw):
                calls.append(argv)
                (args.out / 'partial-observation.txt').write_text('partial fake compiler output')
                raise OSError('synthetic supervision failure')
            with self.patches(root, failed):
                self.assertEqual(entry.run(args), 1)
            result = json.loads((args.out / 'result.json').read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['compile_completed'], 0)
            self.assertIsNone(result['processes'][0]['observation'])
            self.assertIn('OSError', result['processes'][0]['supervision_exception'])
            self.assertTrue((args.out / 'partial-observation.txt').is_file())
            self.assertEqual(json.loads((args.out / 'completion.json').read_text())['status'], 'error')

    def test_external_signal_resource_stop_and_timeout_keep_distinct_terminal_records(self):
        for reason, wanted in (('external', 'interrupted'), ('resource', 'resource_stopped'), ('timeout', 'timeout')):
            with self.subTest(reason=reason), TemporaryDirectory() as directory:
                root = Path(directory)
                args = fixture(root)
                real_receive = entry.receive_signal
                def receive(result, signum):
                    if reason == 'resource':
                        result['guard_trigger'] = dict(reasons=['synthetic physical RSS stop'])
                    return real_receive(result, signum)
                def interrupted(argv, **kw):
                    signum = signal.SIGALRM if reason == 'timeout' else signal.SIGTERM
                    signal.getsignal(signum)(signum, None)
                with self.patches(root, interrupted), patch.object(entry, 'receive_signal', side_effect=receive):
                    self.assertEqual(entry.run(args), 1)
                result = json.loads((args.out / 'result.json').read_text())
                self.assertEqual(result['status'], wanted)
                self.assertEqual(json.loads((args.out / 'completion.json').read_text())['status'], wanted)
                self.assertEqual(result['incoming_signal'], signal.SIGTERM if reason == 'external' else None)
                if reason == 'timeout':
                    self.assertTrue(result['wall_timeout_observed'])

    def test_r10_signal_and_native_timeout_are_preserved_without_retry(self):
        for incoming, native_timeout, wanted in ((signal.SIGINT, False, 'interrupted'), (None, True, 'timeout'), (None, False, 'error')):
            with self.subTest(wanted=wanted), TemporaryDirectory() as directory:
                root = Path(directory)
                args = fixture(root)
                def failed(argv, **kw):
                    work = Path(argv[argv.index('--worker-dir') + 1])
                    (work / 'worker_result.json').write_text(json.dumps(dict(status='error', compile_attempts=1,
                        compile_completed=0, file_gib=16, process=dict(clean(), exit_code=-6, timed_out=native_timeout))))
                    return dict(clean(), exit_code=1, signal=incoming)
                with self.patches(root, failed):
                    self.assertEqual(entry.run(args), 1)
                result = json.loads((args.out / 'result.json').read_text())
                self.assertEqual(result['status'], wanted)
                self.assertEqual(result['native_compile_completed'], 0)
                self.assertEqual(result['compile_attempts'], 1)

    def test_physical_guard_uses_nix_parent_sum_and_retains_cause_if_log_write_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            args.out.mkdir()
            sample = dict(ts_ns=1, processes=[dict(pid=1, vmrss_kib=8 * 1024**2),
                dict(pid=2, vmrss_kib=6 * 1024**2)], outer_process=dict(vmrss_kib=1024))
            resource = args.out / 'resources.jsonl'
            resource.write_text(json.dumps(sample) + '\n' + '{unfinished')
            latest = entry.latest_resource(resource)
            self.assertEqual(latest['subtree_rss_bytes'], 14 * entry.GIB)
            result = dict(guard_trigger=None)
            with patch.object(entry, 'available_ram', return_value=4 * entry.GIB), \
                    patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=13 * entry.GIB)), \
                    patch.object(entry, 'durable', side_effect=OSError('synthetic disk write failure')):
                with self.assertRaisesRegex(OSError, 'disk write'):
                    entry.check_resources(args, result, resource, float('inf'))
            self.assertEqual(entry.status(result), 'resource_stopped')
            self.assertIn('Nix parents', result['guard_trigger']['reasons'][0])

    def test_free_disk_reserve_blocks_before_any_child_and_default_remains_ten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            with self.patches(root, lambda *a, **k: self.fail('Disk admission must precede process launch')), \
                    patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=3 * entry.GIB)):
                self.assertEqual(entry.run(args), 1)
            result = json.loads((args.out / 'result.json').read_text())
            self.assertEqual(result['status'], 'resource_stopped')
            self.assertEqual(result['compile_attempts'], 0)
        captured = []
        with patch.object(entry, 'run', side_effect=lambda args: captured.append(args) or 0):
            self.assertEqual(entry.main(['--source-case', '/old', '--out', '/new']), 0)
        self.assertEqual(captured[0].minimum_free_gib, 10)

    def test_monitor_still_stops_fake_child_when_guard_and_monitor_log_writes_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixture(root)
            killed = threading.Event()
            writes = entry.durable
            calls = []
            def durable(path, value):
                if path.name in ('resource-guard.json', 'monitor-stop.json'):
                    raise OSError('synthetic guard log failure')
                return writes(path, value)
            def process(argv, **kw):
                resource = Path(kw['resources'])
                resource.write_text(json.dumps(dict(ts_ns=1, processes=[dict(pid=111,
                    vmrss_kib=15 * 1024**2)], outer_process=dict(vmrss_kib=1024))) + '\n')
                self.assertTrue(killed.wait(2), 'A monitor log failure must still interrupt the child supervisor')
                return dict(clean(), exit_code=-15, signal=signal.SIGTERM)
            def kill(pid, signum):
                calls.append((pid, signum))
                killed.set()
            with self.patches(root, process), patch.object(entry, 'durable', side_effect=durable), \
                    patch.object(entry.os, 'kill', side_effect=kill):
                self.assertEqual(entry.run(args), 1)
            result = json.loads((args.out / 'result.json').read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['status'], 'resource_stopped')
            self.assertIsNone(result['incoming_signal'])
            self.assertEqual(result['guard_stop_signal'], signal.SIGTERM)
            self.assertIn('guard log failure', result['monitor_record_error'])

    def test_invalid_finite_limits_stop_before_execution(self):
        for option, value in (('--memory-gib', '25'), ('--memory-gib', 'nan'), ('--file-gib', '17'),
                ('--case-gib', '9'), ('--minimum-free-gib', '3.9'), ('--minimum-free-gib', 'inf'), ('--compile-seconds', '7201')):
            with self.subTest(option=option, value=value), redirect_stderr(StringIO()), \
                    patch.object(entry, 'run', side_effect=AssertionError('Invalid policy cannot run')):
                with self.assertRaises(SystemExit):
                    entry.main(['--source-case', '/old', '--out', '/new', option, value])


if __name__ == '__main__':
    unittest.main()
