"""Resource routing only: fake processes, no model arrays or native compiler."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_vgg_fused_pipeline as entry


class FusedMemoryRoutingChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = {name: self.root/('saved-'+name) for name in entry.SOURCE_FIELDS}

    def test_only_original12_and_explicit24_are_accepted(self):
        for value in (12, 24):
            self.assertEqual(entry.validate_memory_gib(value), value)
        for value in (True, False, 0, 4, 16, 32, 12., '24', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.validate_memory_gib(value)

    def test_worker_command_forwards24_inside_the_pure_child_arguments(self):
        command = entry.worker_command(self.source, self.root/'worker', Path('/fixture/python'), 24)
        separator = command.index('--')
        self.assertEqual(command[command.index('--python')+1], '/fixture/python')
        self.assertEqual(command[command.index('--seconds')+1], '1860')
        self.assertGreater(command.index('--memory-gib'), separator)
        self.assertEqual(command[command.index('--memory-gib')+1], '24')
        self.assertEqual(command.count('--memory-gib'), 1)
        for name, path in self.source.items():
            self.assertEqual(command[command.index('--'+name.replace('_', '-'))+1], str(path))
        self.assertNotIn('--registry', command)

    def test_worker_command_default_remains12(self):
        command = entry.worker_command(self.source, self.root/'worker', Path('/fixture/python'))
        self.assertEqual(command[command.index('--memory-gib')+1], '12')

    def worker_cli(self, allowance):
        command = ['--worker', '--out', str(self.root/'no-real-worker'), '--memory-gib', str(allowance)]
        for name, path in self.source.items():
            command += ['--'+name.replace('_', '-'), str(path)]
        with ExitStack() as stack:
            setlimit = stack.enter_context(patch.object(entry.resource, 'setrlimit'))
            alarm = stack.enter_context(patch.object(entry.signal, 'alarm'))
            worker = stack.enter_context(patch.object(entry, 'worker', return_value={'status': 'ok'}))
            stack.enter_context(redirect_stdout(StringIO()))
            code = entry.main(command)
        return code, setlimit, alarm, worker

    def test_direct_worker_cli_sets24_rlimit_without_modifying_file_or_time_allowances(self):
        code, setlimit, alarm, worker = self.worker_cli(24)
        self.assertEqual(code, 0)
        self.assertEqual(setlimit.call_args_list[0].args,
                         (resource.RLIMIT_AS, (24*entry.GIB, 24*entry.GIB)))
        self.assertEqual(setlimit.call_args_list[1].args,
                         (resource.RLIMIT_FSIZE, (16*entry.GIB, 16*entry.GIB)))
        self.assertEqual([call.args for call in alarm.call_args_list], [(1800,), (0,)])
        self.assertEqual(worker.call_args.args[0].memory_gib, 24)
        self.assertFalse((self.root/'no-real-worker').exists())

    def test_direct_worker_cli_keeps_original12_explicitly(self):
        code, setlimit, _, worker = self.worker_cli(12)
        self.assertEqual(code, 0)
        self.assertEqual(setlimit.call_args_list[0].args,
                         (resource.RLIMIT_AS, (12*entry.GIB, 12*entry.GIB)))
        self.assertEqual(worker.call_args.args[0].memory_gib, 12)

    def test_cli_refuses_other_memory_values_before_worker_or_outer_execution(self):
        for value in ('16', '32', '24.0'):
            with self.subTest(value=value), ExitStack() as stack:
                worker = stack.enter_context(patch.object(entry, 'worker'))
                execute = stack.enter_context(patch.object(entry, 'execute'))
                stack.enter_context(redirect_stderr(StringIO()))
                with self.assertRaises(SystemExit):
                    entry.main(['--out', str(self.root/'unused'), '--memory-gib', value])
                worker.assert_not_called();execute.assert_not_called()
        self.assertFalse((self.root/'unused').exists())

    def test_outer_cli_routes24_to_execute(self):
        with patch.object(entry, 'execute', return_value={'status': 'prepared'}) as execute, redirect_stdout(StringIO()):
            code = entry.main(['--out', str(self.root/'case'), '--python', '/fixture/python',
                               '--heir-opt', '/fixture/heir-opt', '--memory-gib', '24'])
        self.assertEqual(code, 0)
        self.assertEqual(execute.call_args.args[0].memory_gib, 24)

    def fake_outer(self, *, allowance=24, fail=False, omit_allowance=False):
        from pt_agent import paths
        from pt_agent.experiments import heir_common_model_pipeline
        tool = self.root/'fake-heir-opt'
        tool.write_text('fixture executable metadata only\n');tool.chmod(0o700)
        args = Namespace(out=self.root/'case', run=True, python=Path('/fixture/python'),
                         heir_opt=tool, registry=self.root/'registry.jsonl',
                         **{name: None for name in entry.SOURCE_FIELDS})
        if not omit_allowance:
            args.memory_gib = allowance
        checked = dict(contract={'role': 'fake-original'}, native_parameters={'role': 'fake-native'})
        calls = []
        def process(command, **options):
            calls.append({'command': command, 'options': options})
            Path(options['log']).write_text('fake process\n')
            Path(options['resources']).write_text('')
            if fail:
                raise MemoryError('synthetic launch failure')
            if '--worker' in command:
                worker_out = Path(command[command.index('--out')+1]);worker_out.mkdir()
                (worker_out/'result.json').write_text(json.dumps(dict(status='ok', lowered_adapter_contract={})))
            return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        with ExitStack() as stack:
            stack.enter_context(patch.object(paths, 'RESULTS_DIR', self.root))
            stack.enter_context(patch.object(heir_common_model_pipeline, 'sources', return_value=self.source))
            stack.enter_context(patch.object(entry, 'source_gate', return_value=checked))
            stack.enter_context(patch.object(entry, '_snapshot', return_value=[]))
            stack.enter_context(patch.object(entry, '_available_ram_bytes', return_value=16*entry.GIB))
            stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*entry.GIB)))
            stack.enter_context(patch.object(entry, '_latest_rss_sample', return_value=None))
            stack.enter_context(patch.object(entry, 'run_process', side_effect=process))
            stack.enter_context(patch.object(entry, 'check_worker', return_value={'passed': True}))
            stack.enter_context(patch.object(entry, 'native_command', return_value=['/fixture/native-compiler']))
            stack.enter_context(patch.object(entry, 'check_native', return_value={'static_bootstrap_operations': 2}))
            result = entry.execute(args)
        return result, calls

    def test24_is_used_for_both_outer_processes_and_stored_limits(self):
        result, calls = self.fake_outer()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(calls), 2)
        self.assertEqual([call['options']['mem_gib'] for call in calls], [24, 24])
        self.assertEqual([call['options']['fsize_gib'] for call in calls], [16, 16])
        self.assertEqual(calls[0]['command'][calls[0]['command'].index('--memory-gib')+1], '24')
        self.assertEqual(result['limits'], dict(total_seconds=9600, worker_seconds=1800,
            nix_seconds=1860, worker_outer_seconds=1920, native_seconds=7200, AS_gib=24,
            subtree_RSS_gib=14, minimum_available_gib=1.25, case_gib=32, file_gib=16,
            minimum_free_disk_gib=10, concurrency=1))
        saved = json.loads((self.root/'case/result.json').read_text())
        self.assertEqual(saved['limits'], result['limits'])
        self.assertEqual(saved['completed'], dict(export_math=1, native_compile=1))
        self.assertEqual(json.loads((self.root/'case/completion.json').read_text())['status'], 'ok')

    def test24_failure_retains_same_limits_completion_and_unstarted_native(self):
        result, calls = self.fake_outer(fail=True)
        self.assertEqual(result['status'], 'error');self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['options']['mem_gib'], 24)
        self.assertIn('MemoryError', result['error'])
        self.assertEqual(result['limits']['AS_gib'], 24)
        self.assertEqual(result['limits']['subtree_RSS_gib'], 14)
        self.assertEqual(result['limits']['minimum_available_gib'], 1.25)
        self.assertEqual(result['attempted'], dict(export_math=1, native_compile=0))
        self.assertEqual(json.loads((self.root/'case/result.json').read_text())['limits'], result['limits'])
        self.assertEqual(json.loads((self.root/'case/completion.json').read_text())['status'], 'error')

    def test_old_outer_namespace_without_new_field_keeps12_for_both_processes(self):
        result, calls = self.fake_outer(omit_allowance=True)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['limits']['AS_gib'], 12)
        self.assertEqual([call['options']['mem_gib'] for call in calls], [12, 12])

    def test_invalid_direct_namespace_is_rejected_before_case_creation(self):
        with self.assertRaisesRegex(ValueError, '12 or24'):
            entry.execute(Namespace(out=self.root/'unused', memory_gib=16))
        self.assertFalse((self.root/'unused').exists())

    def test_physical_memory_guards_stay14_rss_and1point25_available(self):
        sample = dict(subtree_rss_bytes=14*entry.GIB, sample_ts_ns=1)
        self.assertFalse(entry.physical_observation(sample, 1.25*entry.GIB, 10*entry.GIB)['stop'])
        self.assertTrue(entry.physical_observation(dict(sample, subtree_rss_bytes=14*entry.GIB+1),
                                                 16*entry.GIB, 100*entry.GIB)['stop'])
        self.assertTrue(entry.physical_observation(sample, 1.25*entry.GIB-1, 100*entry.GIB)['stop'])


if __name__ == '__main__':
    unittest.main()
