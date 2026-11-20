"""Own metadata/fake-process checks only; no Go, model or API execution."""
from argparse import Namespace
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_component_library_cpu_overlap as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n' if isinstance(value, dict) else value)


def clean(**changes):
    row = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
    row.update(changes)
    return row


class LibraryOverlapChecks(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.args = Namespace(out=self.root/'qwen-component-library-cpu-overlap-fixture',
            prepared=self.root/'original-prepare', go_sdk=self.root/'sdk', registry=self.root/'registry.jsonl',
            bert_case=self.root/'bert-source', bert_run_case=self.root/'bert-run', batch_case=self.root/'batch',
            bert_pid=64170, bert_starttime=1157046, batch_pid=36762, batch_starttime=961036, run=True)
        self.calls = []; self.names = {}; self.table = self.process_table()
        lattigo = self.root/'existing-lattigo'; lattigo.mkdir()
        put(self.args.go_sdk/'VERSION', 'go1.25.13\n')
        put(self.args.go_sdk/'bin/go', 'fake executable metadata only\n')
        (self.args.go_sdk/'bin/go').chmod(0o700)
        put(self.args.prepared/'result.json', dict(round='R31', phase='prepare', status='ok', lattigo_source=str(lattigo)))
        put(self.args.prepared/'work/module/go.mod', 'module pt-agent-r31-plan\n\ngo 1.25.0\n\n'
            'require github.com/tuneinsight/lattigo/v6 v6.0.0\n\n'
            'replace github.com/tuneinsight/lattigo/v6 => '+str(lattigo)+'\n')
        put(self.args.prepared/'work/module/go.sum', 'existing public module metadata\n')
        for name in ('module-cache', 'build-cache'):
            put(self.args.prepared/'work'/name/'public.fixture', 'tiny synthetic cache\n')
        put(self.args.bert_case/'session/SESSION.json', dict(phase='tune', iteration=3, calls=3, failed=None))
        put(self.args.bert_run_case/'result.json', dict(round='BERT-SERIAL-CONTINUATION', model='bert-tiny-sst2',
            status='running', termination=None, source=dict(source_case=str(self.args.bert_case)),
            attempted=dict(step=1), completed=dict(step=0)))
        put(self.args.batch_case/'supervisor.json', dict(pid=self.args.batch_pid,
            process=dict(pid=self.args.batch_pid, starttime=self.args.batch_starttime)))
        put(self.args.batch_case/'active.json', dict(status='running', id='original-bert', out=str(self.args.bert_run_case),
            argv=['python3', '-B', '-m', 'pt_agent.experiments.bert_serial_continuation',
                  '--source-case', str(self.args.bert_case)]))
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(entry.paths, 'RESULTS_DIR', self.root))
        self.stack.enter_context(patch.object(entry.overlap, '_scan', side_effect=lambda: self.table))
        self.stack.enter_context(patch.object(entry.overlap, '_lease_owners', return_value=[self.args.batch_pid]))
        self.stack.enter_context(patch.object(entry.overlap, '_process_name', side_effect=lambda pid: self.names.get(pid, 'python3')))
        self.stack.enter_context(patch.object(entry.original, '_available_ram_bytes', return_value=12*entry.GIB))
        self.stack.enter_context(patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=40*entry.GIB)))

    def process_table(self):
        return {36762: dict(pid=36762, ppid=1, starttime=961036, state='S'),
            40000: dict(pid=40000, ppid=36762, starttime=7, state='S'),
            64170: dict(pid=64170, ppid=40000, starttime=1157046, state='R'),
            os.getpid(): dict(pid=os.getpid(), ppid=1, starttime=9, state='R', vmrss_kib=2048)}

    def fake(self, after=None, bad=None, archive=True):
        def build(command, **options):
            self.calls.append((command, options))
            Path(options['log']).write_text('fake Go build observation only\n')
            Path(options['resources']).write_text('')
            if archive:
                put(Path(command[command.index('-o')+1]), 'fake archive metadata\n')
            if after:
                after()
            return clean(**(bad or {}))
        with patch.object(entry.original, 'run_process', side_effect=build):
            return entry.execute(self.args)

    def observe(self, sample=None, available=None):
        self.args.out.mkdir(exist_ok=True)
        return entry._namespace(self.args)['physical'](sample, 12*entry.GIB if available is None else available)

    def test_full_fake_build_retains_original_command_env_resources_and_completion(self):
        result = self.fake()
        self.assertEqual(result['status'], 'ok'); self.assertEqual(len(self.calls), 1)
        command, options = self.calls[0]
        self.assertEqual(command, [str(self.args.go_sdk/'bin/go'), 'build', '-p=1', '-mod=readonly',
            '-buildvcs=false', '-buildmode=archive', '-o', str(self.args.out/'work/qwen-generation.a'), '.'])
        self.assertEqual(options['mem_gib'], 4); self.assertEqual(options['fsize_gib'], 8)
        self.assertLessEqual(options['timeout'], 600)
        for name, value in {'GOENV':'off','GOWORK':'off','GOTOOLCHAIN':'local','GOPROXY':'off',
                'GOSUMDB':'off','CGO_ENABLED':'0','GOMAXPROCS':'1','GOMEMLIMIT':'2GiB'}.items():
            self.assertEqual(options['env'][name], value)
        self.assertEqual(result['attempted_builds'], 1); self.assertEqual(result['completed_builds'], 1)
        self.assertTrue(result['generic_library_build_passed']); self.assertTrue(result['generic_library_compile_accepted'])
        self.assertTrue(all(result[name] == 0 for name in entry.original.ZERO))
        for flag in ('linked_native_program_verified','full_qwen_compile_verified','whole_generation_verified'):
            self.assertIs(result[flag], False)
        self.assertEqual(result['actual_component_artifacts_admitted'], 0)
        self.assertIsNone(result['actual_component_manifest']); self.assertIsNone(result['security_acceptance'])
        self.assertIsNone(result['numerical_acceptance']); self.assertIsNone(result['CKKS_latency'])
        self.assertEqual(result['limits']['monitor_poll_seconds'], .25)
        self.assertFalse(result['global_heavy_lease_acquired_by_this_entry']); self.assertEqual(result['bert_signal_actions'], 0)
        self.assertTrue(result['existing_initializer_lock_retained'])
        row = json.loads(self.args.registry.read_text())
        self.assertEqual(set(row), {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        self.assertEqual(row['round'], entry.ROUND); self.assertEqual(row['mode'], entry.MODE)
        marker = entry.original.read(self.args.out/'completion.json')
        self.assertTrue(marker['registry_persisted']); self.assertTrue(marker['required_artifacts_persisted'])
        self.assertEqual(entry.original.read(self.args.out/'build-process-return.json'), clean())
        self.assertTrue((self.args.out/'source/underlying-library-builder.py').is_file())
        self.assertTrue((self.args.out/'source/shared-cpu-overlap-guard.py').is_file())

    def test_guard_delegation_adds_explicit_free_args_and_preserves_observations(self):
        self.args.out.mkdir()
        raw = dict(stop=False, reasons=[], scope='exact shared guard scope')
        sample = dict(subtree_rss_bytes=1234, sample_ts_ns=8)
        with patch.object(entry.overlap, 'parallel_observation', return_value=raw.copy()) as delegated:
            row = entry._namespace(self.args)['physical'](sample, 11*entry.GIB)
        delegated.assert_called_once_with(self.args, sample, 11*entry.GIB, 40*entry.GIB)
        self.assertEqual(row['shared_guard_scope'], raw['scope']); self.assertEqual(row['scope'], entry.SCOPE)
        self.assertEqual(entry.original.read(self.args.out/'parallel-observations.jsonl'), row)

    def test_original_sdk_gate_single_attempt_and_scope(self):
        checked = entry.source_gate(self.args)
        self.assertEqual(checked['go_version'], '1.25.13'); self.assertEqual(checked['lattigo_version'], '6.0.0')
        self.assertFalse(checked['parameters_selected']); self.assertEqual(checked['actual_component_artifacts_admitted'], 0)
        self.assertEqual(checked['identified_bert']['worker_starttime'], self.args.bert_starttime)
        put(self.args.registry, dict(round=entry.ROUND, mode=entry.MODE, metrics=dict(attempted_builds=1)))
        with self.assertRaisesRegex(ValueError, 'one build'):
            entry.source_gate(self.args)

    def test_prepare_only_copies_no_cache_and_runs_no_subprocess(self):
        self.args.run = False
        result = self.fake()
        self.assertEqual(result['status'], 'prepared'); self.assertEqual(self.calls, [])
        self.assertEqual(result['attempted_builds'], 0); self.assertFalse(result['generic_library_build_passed'])
        self.assertFalse((self.args.out/'work/module-cache').exists())

    def test_metadata_transition_stops_before_any_build_and_preserves_result(self):
        put(self.args.bert_case/'session/SESSION.json', dict(phase='confirm', iteration=3, calls=3, failed=None))
        result = self.fake()
        self.assertEqual(result['status'], 'error'); self.assertEqual(self.calls, [])
        self.assertEqual(result['attempted_builds'], 0); self.assertFalse(result['generic_library_build_passed'])
        self.assertTrue(result['resource_guard']['stop']); self.assertTrue((self.args.out/'completion.json').is_file())

    def test_changed_birth_lease_foreign_compiler_and_private_descendants(self):
        self.assertFalse(self.observe()['stop'])
        self.table[self.args.bert_pid]['starttime'] += 1
        self.assertTrue(self.observe()['stop'])
        self.table = self.process_table()
        with patch.object(entry.overlap, '_lease_owners', return_value=[999]):
            self.assertTrue(self.observe()['stop'])
        self.table[77] = dict(pid=77, ppid=os.getpid(), starttime=3, state='R')
        self.names[77] = 'compile'
        self.assertFalse(self.observe()['stop'])
        self.table[78] = dict(pid=78, ppid=1, starttime=4, state='R'); self.names[78] = 'compile'
        self.assertEqual(self.observe()['foreign_compilers'], [dict(pid=78, starttime=4, name='compile')])

    def test_rss_supervisor_available_and_disk_guards_are_independent(self):
        # Include 2MiB of own supervisor RSS in the exact 4GiB boundary.
        self.assertFalse(self.observe(dict(subtree_rss_bytes=4*entry.GIB-2*1024**2), 2*entry.GIB)['stop'])
        self.assertTrue(self.observe(dict(subtree_rss_bytes=4*entry.GIB), 2*entry.GIB)['stop'])
        self.assertTrue(self.observe(available=2*entry.GIB-1)['stop'])
        with patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*entry.GIB-1)):
            self.assertTrue(self.observe()['stop'])

    def test_clean_real_return_after_bert_transition_preserves_counts_but_revokes_success(self):
        def transition():
            put(self.args.bert_case/'session/SESSION.json', dict(phase='done', iteration=3, calls=3, failed=None))
        result = self.fake(after=transition)
        self.assertEqual(result['status'], 'error'); self.assertEqual(result['completed_builds'], 1)
        self.assertFalse(result['generic_library_build_passed']); self.assertFalse(result['generic_library_compile_accepted'])
        self.assertEqual(result['observation'], clean())
        self.assertEqual(entry.original.read(self.args.out/'build-process-return.json'), clean())
        self.assertFalse(json.loads(self.args.registry.read_text())['metrics']['generic_library_build_passed'])

    def test_nonclean_child_return_is_saved_once_without_retry(self):
        result = self.fake(bad=dict(exit_code=1))
        self.assertEqual(result['status'], 'error'); self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['completed_builds'], 0); self.assertEqual(result['observation']['exit_code'], 1)
        self.assertFalse(result['generic_library_compile_accepted'])
        self.assertEqual(entry.original.read(self.args.out/'build-process-return.json')['exit_code'], 1)

    def monitor_fixture(self, failed_log=False):
        signals = []; thread_errors = []
        real_durable = entry.original.durable_json
        def persist(path, value):
            if failed_log and Path(path).name == 'guard-failure.json':
                raise OSError('synthetic guard-log storage failure')
            return real_durable(path, value)
        def transition():
            put(self.args.bert_case/'session/SESSION.json', dict(phase='done', iteration=3, calls=3, failed=None))
            until = time.monotonic()+2
            while not signals and time.monotonic() < until:
                time.sleep(.01)
        old = signal.getsignal(signal.SIGTERM)
        with patch.object(entry.os, 'kill', side_effect=lambda pid, sig: signals.append((pid, sig))), \
                patch.object(entry.original, 'durable_json', side_effect=persist), \
                patch.object(threading, 'excepthook', side_effect=lambda args: thread_errors.append(args.exc_value)):
            result = self.fake(after=transition)
        self.assertEqual(signals, [(os.getpid(), signal.SIGTERM)])
        self.assertIs(signal.getsignal(signal.SIGTERM), old)
        self.assertEqual(result['status'], 'error'); self.assertEqual(result['bert_signal_actions'], 0)
        self.assertFalse(result['generic_library_build_passed']); self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.args.out/'completion.json').is_file())
        return result, thread_errors

    def test_continuous_quarter_second_monitor_stops_only_own_supervisor(self):
        result, errors = self.monitor_fixture()
        self.assertEqual(errors, [])
        self.assertTrue((self.args.out/'guard-failure.json').is_file())
        self.assertTrue(result['latest_parallel_observation']['stop'])

    def test_monitor_guard_log_failure_still_stops_own_supervisor(self):
        result, errors = self.monitor_fixture(failed_log=True)
        self.assertEqual(len(errors), 1); self.assertIsInstance(errors[0], OSError)
        self.assertIn('synthetic guard-log', result['latest_parallel_observation']['guard_persistence_error'])

    def test_namespace_retains_initializer_lock_original_globals_and_code(self):
        namespace = entry._namespace(self.args)
        self.assertIs(namespace['serial_lock'], entry.original.serial_lock)
        self.assertIs(namespace['execute'].__code__, entry.original.execute.__code__)
        self.assertIs(namespace['close'].__code__, entry.original.close.__code__)
        self.assertIs(namespace['GO_SOURCE'], entry.original.GO_SOURCE)
        self.assertIsNot(namespace['execute'].__globals__, entry.original.execute.__globals__)
        self.assertIs(entry.original.execute.__globals__['physical'], entry.original.physical)
        self.assertIs(entry.original.execute.__globals__['source_gate'], entry.original.source_gate)
        with self.assertRaises(TypeError):
            entry.overlap._clone_plain(entry.original.serial_lock, {})

    def test_case_identity_positive_cli_and_resource_contract_are_explicit(self):
        for value in ('0','-1','nan', True, 1.5):
            with self.subTest(value=value), self.assertRaises(Exception):
                entry.positive(value)
        self.assertEqual(entry.positive('64170'), 64170)
        self.args.bert_pid = True
        with self.assertRaises(ValueError):
            entry.execute(self.args)
        self.args.bert_pid = 64170; self.args.out = self.args.bert_case
        with self.assertRaises(ValueError):
            entry.execute(self.args)
        self.args.out = self.root/'qwen-component-library-cpu-overlap-fixture'
        with self.assertRaisesRegex(ValueError, 'AS4/file8'):
            entry._namespace(self.args)['run_process']([], mem_gib=12, fsize_gib=8)


if __name__ == '__main__':
    unittest.main()
