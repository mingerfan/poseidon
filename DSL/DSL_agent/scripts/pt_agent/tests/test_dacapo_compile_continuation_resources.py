"""Explicit address-space/RSS variants with small saved-source fixtures only."""
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_compile_continuation as entry
from pt_agent.tests import test_dacapo_compile_continuation as fixtures


class DaCapoContinuationResourceVariants(unittest.TestCase):
    def cli(self, *options):
        captured = []
        with patch.object(entry, 'run', side_effect=lambda args: captured.append(args) or 0):
            self.assertEqual(entry.main(['--source-case', '/old', '--out', '/new', *options]), 0)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_default_caps_and_explicit_independent_caps(self):
        default = self.cli()
        self.assertEqual((default.memory_gib, default.rss_gib, default.minimum_free_gib), (24, 14, 10))
        chosen = self.cli('--memory-gib', '32', '--rss-gib', '14.2')
        self.assertEqual((chosen.memory_gib, chosen.rss_gib), (32, 14.2))
        self.assertEqual((chosen.file_gib, chosen.case_gib, chosen.compile_seconds), (16, 8, 7200))
        self.assertEqual(self.cli('--rss-gib', '16').rss_gib, 16)
        self.assertEqual(self.cli('--case-gib', '32').case_gib, 32)

    def test_nonfinite_and_out_of_range_caps_stop_before_run(self):
        for flag, value in (('--memory-gib', '32.1'), ('--memory-gib', 'nan'),
                            ('--rss-gib', '13.9'), ('--rss-gib', '16.1'),
                            ('--rss-gib', 'nan'), ('--rss-gib', 'inf'), ('--case-gib', '32.1')):
            with self.subTest(flag=flag, value=value), patch.object(entry, 'run') as run, redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    entry.main(['--source-case', '/old', '--out', '/new', flag, value])
                run.assert_not_called()

    def test_rss_and_available_ram_guards_operate_independently(self):
        for rss_gib, rss_bytes, available, accepted, cause in (
            (14, int(14.1 * entry.GIB), 4 * entry.GIB, False, 'subtree including Nix parents and supervisor exceeds14GiB RSS'),
            (14.2, int(14.1 * entry.GIB), 4 * entry.GIB, True, None),
            (16, 2 * entry.GIB, entry.GIB, False, 'available physical memory below1.25GiB'),
            (16, int(16.1 * entry.GIB), 4 * entry.GIB, False, 'subtree including Nix parents and supervisor exceeds16GiB RSS'),
        ):
            with self.subTest(rss=rss_gib, available=available), TemporaryDirectory() as directory:
                args = SimpleNamespace(out=Path(directory), rss_gib=rss_gib, minimum_free_gib=10, case_gib=8)
                sample = dict(subtree_plus_supervisor_rss_bytes=rss_bytes)
                result = dict(guard_trigger=None)
                with patch.object(entry, 'latest_resource', return_value=sample), \
                        patch.object(entry, 'available_ram', return_value=available), \
                        patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=12 * entry.GIB)):
                    if accepted:
                        observed = entry.check_resources(args, result, None, float('inf'))
                        self.assertEqual(observed['latest_process_sample'], sample)
                        self.assertIsNone(result['guard_trigger'])
                    else:
                        with self.assertRaises(RuntimeError):
                            entry.check_resources(args, result, None, float('inf'))
                        self.assertEqual(result['guard_trigger']['reasons'], [cause])
                        self.assertEqual(json.loads((args.out / 'resource-guard.json').read_text())['reasons'], [cause])

    def test_as32_reaches_outer_and_worker_without_changing_saved_compiler_argv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixtures.fixture(root)
            args.memory_gib, args.rss_gib, args.minimum_free_gib = 32, 14.2, 10
            original_result = deepcopy(entry.read(args.source_case / 'result.json'))
            saved_argv = entry.read(args.source_case / 'compile-argv.json')
            launches = []
            observation = dict(fixtures.clean(), limits=dict(memory_gib=32, file_gib=16, wall_seconds=7200))

            def compile_once(argv, **kwargs):
                launches.append(argv)
                self.assertEqual((kwargs['mem_gib'], kwargs['fsize_gib'], kwargs['timeout']), (32, 16, 7320))
                self.assertEqual(argv[argv.index('--memory-gib') + 1], '32')
                self.assertEqual(argv[argv.index('--file-gib') + 1], '16')
                self.assertIn('--worker-compile', argv)
                command = entry.read(argv[argv.index('--command-file') + 1])
                self.assertEqual([i for i, (a, b) in enumerate(zip(saved_argv, command)) if a != b],
                                 [saved_argv.index('-o') + 1])
                work = Path(argv[argv.index('--worker-dir') + 1])
                fixtures.prior.exported(work)
                entry.write_json(work / 'worker_result.json', dict(status='ok', compile_attempts=1,
                    compile_completed=1, file_gib=16, process=observation))
                Path(kwargs['log']).write_text('one synthetic compile')
                Path(kwargs['resources']).write_text('')
                return observation

            with ExitStack() as stack:
                stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', root))
                stack.enter_context(patch.object(entry, 'available_ram', return_value=4 * entry.GIB))
                stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=12 * entry.GIB)))
                stack.enter_context(patch.object(entry, 'run_process', side_effect=compile_once))
                self.assertEqual(entry.run(args), 0)
            self.assertEqual(len(launches), 1)
            result = entry.read(args.out / 'result.json')
            self.assertEqual(result['limits']['subtree_plus_supervisor_RSS_gib'], 14.2)
            self.assertEqual(result['limits']['memory_gib'], 32)
            self.assertEqual(result['limits']['minimum_available_RAM_gib'], 1.25)
            self.assertEqual((result['trace_attempts'], result['compile_attempts'], result['compile_completed']), (0, 1, 1))
            self.assertEqual(entry.read(args.source_case / 'result.json'), original_result)
            self.assertFalse(result['source']['trace_content_read'])
            self.assertEqual(entry.read(args.out / 'completion.json')['status'], 'ok')
            rows = [json.loads(row) for row in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})

    def test_original_worker_accepts_as32_and_uses_it_for_native_child(self):
        with TemporaryDirectory() as directory:
            args = fixtures.fixture(Path(directory))
            work = args.out / 'emit'
            source = entry.source_gate(args.source_case, work / 'lowered.mlir')
            args.out.mkdir()
            command_file = args.out / 'compile-argv.json'
            entry.write_json(command_file, source['argv'])
            observation = dict(fixtures.clean(), limits=dict(memory_gib=32, file_gib=16, wall_seconds=7200))
            with patch.dict('os.environ', {'HECATE_PYTHON_LIBRARY_PATH': '/synthetic/lib'}), \
                    patch.object(entry.original, 'run_process', return_value=observation) as native:
                self.assertEqual(entry.original.main(['--worker-compile', '--worker-dir', str(work),
                    '--command-file', str(command_file), '--compiler', source['compiler'],
                    '--worker-seconds', '7200', '--memory-gib', '32', '--file-gib', '16',
                    '--diagnostic-elide-final-ir']), 0)
            self.assertEqual((native.call_args.kwargs['mem_gib'], native.call_args.kwargs['fsize_gib'],
                              native.call_args.kwargs['timeout']), (32, 16, 7200))
            self.assertEqual(entry.read(work / 'worker_result.json')['compile_completed'], 1)


if __name__ == '__main__':
    unittest.main()
