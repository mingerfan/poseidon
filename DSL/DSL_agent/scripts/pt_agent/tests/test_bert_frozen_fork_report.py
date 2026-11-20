"""Isolated report metadata/protocol tests; no model, array or report read."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import bert_frozen_fork_report as entry
from pt_agent.tests import test_bert_budget_fork_compile as fixtures


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def clean():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])


class ReportChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ForkCompileTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.binding = self.fixture.root, self.fixture.args
        self.binding.out.mkdir()
        scope = entry.compiled.namespace(self.binding)
        self.source = scope['metadata_source'](self.binding.source_case, self.binding.checkpoint_audit)
        math_path = self.binding.out/'math/math-result.json'
        put(math_path, self.fixture.proof(scope))
        expected = scope['compile_source'](self.binding.source_case, self.binding.checkpoint_audit, math_path)
        self.master = dict(round=entry.compiled.ROUND, mode=entry.compiled.MODE, model=entry.MODEL,
            status='ok', source=self.source, math_passed=True, compile_completed=True, compile_accepted=True,
            attempted=dict(math=1, compile=1), completed=dict(math=1, compile=1), report_deferred=True,
            compiled_bootstraps=9180, math_result=str(math_path), compile_case=str(self.binding.compile_out),
            processes=[dict(phase=p, observation=clean()) for p in ('math', 'compile')], metrics={'status':'ok'},
            **dict.fromkeys(entry.compiled.base.ZERO, 0))
        self.native = dict(round=entry.compiled.ROUND, mode=entry.compiled.MODE, model=entry.MODEL,
            status='ok', source=expected, compiled_bootstraps=9180, attempted=dict(emit=1, trace=1, compile=1),
            completed=dict(emit=1, trace=1, compile=1), metrics={'status':'ok'},
            processes=[dict(phase=p, observation=clean()) for p in ('emit', 'trace', 'compile')],
            **dict.fromkeys(entry.compiled.base.original.ZERO_FIELDS, 0))
        for case, result in ((self.binding.out, self.master), (self.binding.compile_out, self.native)):
            put(case/'result.json', result)
            put(case/'completion.json', dict(status='ok', required_artifacts_persisted=True, registry_persisted=True))
            self.add_row(case, result)
        self.args = SimpleNamespace(source_case=self.binding.source_case, compile_case=self.binding.out,
            out=self.root/'isolated-report', metadata_template=self.binding.metadata_template,
            intake_python=self.binding.intake_python, data_python=self.fixture.f.args.data_python,
            ledger=self.fixture.f.args.ledger, allowance=self.fixture.f.args.allowance)
        # The existing synthetic SST-2 source has the same independent reader.
        self.source['sample_source']['reader_python'] = str(self.args.data_python)
        for model in (self.fixture.f.case/'model', self.args.source_case/'model'):
            sample = entry.read(model/'samples/source.json'); sample['reader_python'] = str(self.args.data_python)
            put(model/'samples/source.json', sample)
        # Source snapshots and existing mathematical proof retain the updated
        # synthetic reader metadata, as a real preparation would have done.
        self.source = scope['metadata_source'](self.binding.source_case, self.binding.checkpoint_audit)
        put(math_path, self.fixture.proof(scope))
        self.master['source'] = self.source
        self.native['source'] = scope['compile_source'](self.binding.source_case, self.binding.checkpoint_audit, math_path)
        put(self.binding.out/'result.json', self.master); put(self.binding.compile_out/'result.json', self.native)
        (self.args.source_case/'model/samples/confirm.npz').write_text('synthetic confirm cache')
        self.work = self.args.out/'worker'

    def add_row(self, case, result):
        row = dict(run_id=case.name, round=result['round'], model=result['model'], mode=result['mode'],
            metrics=result['metrics'], artifacts_dir=str(case), ts='synthetic', git_rev='synthetic')
        with (self.root/'registry.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')

    def gate(self):
        return entry.admission(self.args)

    def accepted_worker(self, policy=True):
        checks = [dict(passed=True, criterion=entry.compiled.base.original.CRITERION,
                       max_absolute_error=0., max_tolerance_ratio=0.) for _ in range(3)]
        worker = dict(status='ok', phase='report', model=entry.MODEL, frozen=True,
            report_completed=True, report_samples=872, llm_calls=0, task_policy={'acc_drop_pp':.5},
            task_policy_passed=policy, all_finite=True, validation_metrics={'acc_drop_pp':0. if policy else 1.},
            frozen_math=[dict(sample=i, checks=copy.deepcopy(checks)) for i in range(3)],
            **dict.fromkeys(('native_test_reads', 'compiler_runs', 'keygen_runs', 'real_ckks_attempts', 'real_ckks_completed'), 0),
            **dict.fromkeys(original_formal(), None))
        return worker

    def fake_report(self, *, policy=True, leave_marker=False, **changes):
        def run(argv, **kwargs):
            self.assertIn('--delegate-worker', argv); self.assertEqual(argv[argv.index('--phase')+1], 'report')
            worker = self.accepted_worker(policy); worker.update(changes)
            put(self.work/'worker_result.json', worker)
            (self.work/'report-outputs.npz').write_text('synthetic placeholder; no NPZ values')
            put(self.args.out/'session/REPORT_ACCESS.json', dict(complete=not leave_marker, report_samples=872,
                program=str(self.args.out/'session/program.ptdsl'), result=str(self.work/'worker_result.json')))
            return clean()
        return run

    def run_fake(self, runner=None):
        with mock.patch.object(entry, 'resources', return_value={'stop':False}), \
             mock.patch.object(entry.serial, 'foreign_heavy_processes', return_value=[]), \
             mock.patch.object(entry, 'run_process', side_effect=runner or self.fake_report()), \
             mock.patch('pt_agent.registry.git_rev', return_value='synthetic'):
            return entry.execute(self.args)

    def test_actual_metadata_gates_reuse_original_C2_checkpoint_and_compiler_proof(self):
        source = self.gate()
        self.assertEqual(source['source_case'], str(self.args.source_case))
        self.assertEqual(source['compiled_bootstraps'], 9180)
        self.assertFalse(source['report_feedback_permitted'])
        self.assertFalse(self.args.out.exists())

    def test_incomplete_compile_foreign_source_and_duplicate_registry_reject_before_worker(self):
        for field, value in [('compile_accepted', False), ('completed', dict(math=1, compile=0))]:
            bad = copy.deepcopy(self.master); bad[field] = value; put(self.binding.out/'result.json', bad)
            with self.subTest(field=field), self.assertRaises(ValueError): self.gate()
        put(self.binding.out/'result.json', self.master)
        bad = copy.deepcopy(self.native); bad['source']['selected_source'] = 'foreign'; put(self.binding.compile_out/'result.json', bad)
        with self.assertRaises(ValueError): self.gate()
        put(self.binding.compile_out/'result.json', self.native)
        self.add_row(self.binding.compile_out, self.native)
        with self.assertRaisesRegex(ValueError, 'eight-field'): self.gate()

    def test_independent_shell_preserves_all_originals_and_only_rebases_marker(self):
        source = self.gate(); self.args.out.mkdir()
        original_marker = entry.read(self.args.source_case/'session/FROZEN.json')
        result = entry.prepare_shell(self.args, source)
        self.assertEqual(len(result['aliases']), 4)
        self.assertEqual(entry.read(self.args.source_case/'session/FROZEN.json'), original_marker)
        self.assertNotEqual(entry.read(self.args.out/'session/FROZEN.json')['program'], original_marker['program'])
        self.assertEqual(entry.serial.frozen_state(self.args.out)['outcome'], 'frozen')
        for name in ('SESSION.json', 'AGENT.json'):
            self.assertEqual(entry.read(self.args.out/'session'/name), entry.read(self.args.source_case/'session'/name))
        target = self.args.out/'model/samples/heldout.npz'; target.unlink(); target.symlink_to(self.root/'foreign')
        with self.assertRaisesRegex(ValueError, 'reference changed'): entry.shell_bindings(self.args)

    def test_command_uses_explicit_private_shell_and_only_original_report_worker(self):
        argv = entry.command(self.args, self.gate(), 1500.9)
        for flag, path in [('--model-dir', self.args.out/'model'), ('--session-dir', self.args.out/'session'), ('--work-dir', self.work)]:
            self.assertEqual(argv[argv.index(flag)+1], str(path))
        self.assertEqual(argv[argv.index('--seconds')+1], '1500')
        self.assertIn('pt_agent.experiments.bert_sst2_recorded', argv)
        self.assertIn('--delegate-worker', argv); self.assertNotIn('--out', argv)
        self.assertEqual(argv[argv.index('--phase')+1], 'report')

    def test_complete_report_has_durable_registry_and_original_cases_remain_unchanged(self):
        old = {p:entry.read(p) for p in [self.args.source_case/'result.json', self.args.source_case/'completion.json',
                                       self.binding.out/'result.json', self.binding.compile_out/'result.json']}
        result = self.run_fake()
        self.assertEqual(result['status'], 'ok', result['error']); self.assertTrue(result['report_accepted'])
        self.assertEqual(result['completed_report'], 1); self.assertEqual(result['report_samples'], 872)
        self.assertEqual(result['api_calls'], 0); self.assertEqual(result['new_confirmation_commands'], 0)
        for path, record in old.items(): self.assertEqual(entry.read(path), record)
        self.assertFalse((self.args.source_case/'session/REPORT_ACCESS.json').exists())
        self.assertTrue((self.args.out/'session/REPORT_ACCESS.json').is_file())
        marker = entry.read(self.args.out/'completion.json'); self.assertEqual(marker['status'], 'ok')
        rows = [json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        row = [r for r in rows if r['run_id'] == self.args.out.name]
        self.assertEqual(len(row), 1); self.assertEqual(set(row[0]), entry.serial.REGISTRY_FIELDS)
        self.assertLessEqual(entry.artifact_bytes(self.args.out), entry.CASE_BYTES)

    def test_normal_task_policy_rejection_preserves_completed_report_and_frozen_program(self):
        result = self.run_fake(self.fake_report(policy=False))
        self.assertEqual(result['status'], 'ok'); self.assertFalse(result['task_policy_passed'])
        self.assertTrue(result['report_accepted']); self.assertEqual(result['new_agent_steps'], 0)
        self.assertIsNone(result['accuracy_acceptance'])

    def test_incomplete_marker_and_strict_math_failure_are_retained_without_retry(self):
        result = self.run_fake(self.fake_report(leave_marker=True))
        self.assertEqual(result['status'], 'error'); self.assertFalse(result['report_accepted'])
        self.assertFalse(entry.read(self.args.out/'session/REPORT_ACCESS.json')['complete'])
        with self.assertRaises(FileExistsError): entry.execute(self.args)
        self.assertTrue((self.root/'bert-frozen-report-claims'/(self.args.source_case.name+'.json')).is_file())

    def test_wrong_report_count_or_original_policy_and_math_are_rejected(self):
        self.args.out.mkdir(); entry.prepare_shell(self.args, self.gate()); self.work.mkdir()
        put(self.args.out/'session/REPORT_ACCESS.json', dict(complete=True, report_samples=872,
            program=str(self.args.out/'session/program.ptdsl'), result=str(self.work/'worker_result.json')))
        (self.work/'report-outputs.npz').write_text('synthetic')
        for change in ('count', 'policy', 'math'):
            worker = self.accepted_worker()
            if change == 'count': worker['report_samples'] = 871
            elif change == 'policy': worker['task_policy'] = {'acc_drop_pp':1.}
            else: worker['frozen_math'][0]['checks'][0]['max_tolerance_ratio'] = 1.01
            with self.subTest(change=change), self.assertRaises(ValueError): entry.check_report(self.args, worker)

    def test_claim_once_prevents_another_output_case_from_reading_same_report(self):
        self.args.out.mkdir(); entry.claim(self.args)
        other = copy.copy(self.args); other.out = self.root/'another-report'
        with self.assertRaises(FileExistsError): entry.claim(other)

    def test_physical_guard_has_original_four_GiB_and_host_reserve(self):
        self.args.out.mkdir()
        with mock.patch.object(entry, '_scan', return_value={os.getpid():dict(ppid=0, vmrss_kib=4*1024**2+1)}), \
             mock.patch.object(entry, '_available_ram_bytes', return_value=2*entry.GIB), \
             mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)):
            self.assertTrue(entry.resources(self.args)['stop'])
        with mock.patch.object(entry, '_scan', return_value={}), \
             mock.patch.object(entry, '_available_ram_bytes', return_value=int(1.24*entry.GIB)), \
             mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)):
            self.assertTrue(entry.resources(self.args)['stop'])

    def test_incoming_child_signal_and_timeout_keep_partial_process_truth(self):
        def stop(*a, **k): return dict(clean(), exit_code=143, signal=15, timed_out=True)
        result = self.run_fake(stop)
        self.assertEqual(result['status'], 'error'); self.assertTrue(result['wall_timeout_observed'])
        self.assertEqual(result['process']['signal'], 15); self.assertFalse(result['report_accepted'])
        self.assertEqual(result['attempted_report'], 1); self.assertEqual(result['completed_report'], 0)

    def test_terminal_reserve_retracts_acceptance_and_keeps_completed_report_fact(self):
        with mock.patch.object(entry, 'CASE_BYTES', 1): result = self.run_fake()
        self.assertEqual(result['status'], 'error'); self.assertFalse(result['report_accepted'])
        self.assertEqual(result['completed_report'], 1)
        self.assertGreater(result['terminal_budget']['bytes_before_completion'], 1)
        self.assertFalse(entry.read(self.args.out/'result.json')['metrics']['report_accepted'])

    def test_original_report_marker_or_foreign_heavy_process_stops_before_any_child(self):
        marker = self.args.source_case/'session/REPORT_ACCESS.json'; marker.write_text('unread report marker')
        with self.assertRaises(ValueError): self.gate()
        marker.unlink()
        with mock.patch.object(entry.serial, 'foreign_heavy_processes', return_value=[{'pid':123}]), \
             mock.patch.object(entry, 'run_process') as runner, \
             mock.patch('pt_agent.registry.git_rev', return_value='synthetic'):
            result = entry.execute(self.args)
        self.assertEqual(result['status'], 'error'); self.assertEqual(result['attempted_report'], 0)
        runner.assert_not_called()
        self.assertFalse((self.root/'bert-frozen-report-claims').exists())

    def test_real_recorded_inner_entry_has_no_global_registry_or_config_reads(self):
        from pt_agent.experiments import bert_sst2_recorded as recorded
        work = self.root/'inner-recorded'; fake_source = self.root/'fake-worker.py'; fake_source.write_text('# synthetic hook only')
        observed = []
        def numeric_unstarted(argv):
            observed.append(argv)
            self.assertEqual(argv[argv.index('--phase')+1], 'report')
            self.assertEqual(argv[argv.index('--session-dir')+1], str(self.args.out/'session'))
            return 0
        argv = entry.command(self.args, self.gate(), 1200)
        rest = argv[argv.index('--delegate-worker'):]
        rest[rest.index('--work-dir')+1] = str(work)
        template = self.root/'valid-recorded-template.json'
        put(template, ['{hecate_opt}', '{input}', '--pars', '--waterline=43',
            '--runtime-plan-context-id=qwen-bootstrap48', '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64, '-o', '{output_dir}/lowered.mlir'])
        rest[rest.index('--metadata-template')+1] = str(template)
        original_open = Path.open
        def local_only(path, *args, **kwargs):
            if Path(path).name == 'registry.jsonl' or Path(path).is_relative_to('/forbidden-global'):
                raise AssertionError('Inner global filesystem access is forbidden')
            return original_open(path, *args, **kwargs)
        from contextlib import nullcontext
        with mock.patch.object(recorded.paths, 'RESULTS_DIR', Path('/forbidden-global')), \
             mock.patch.object(Path, 'open', local_only), \
             mock.patch('pt_agent.config.load', side_effect=AssertionError('Inner config.load is forbidden')), \
             mock.patch.object(recorded, 'native_metadata_commands', side_effect=lambda *a:nullcontext()), \
             mock.patch.object(recorded.importlib, 'import_module', return_value=SimpleNamespace(__file__=str(fake_source), main=numeric_unstarted)):
            self.assertEqual(recorded.main(rest), 0)
        self.assertEqual(len(observed), 1)
        delegation = entry.read(work/'recorded-command-delegation.json')
        self.assertTrue(delegation['early_context_installed_before_worker_import'])

    def test_foreign_existing_interpreter_or_ledger_path_rejects_source_binding(self):
        foreign = self.root/'foreign-existing-file'; foreign.touch()
        for name in ('intake_python', 'data_python', 'ledger', 'allowance', 'metadata_template'):
            original = getattr(self.args, name); setattr(self.args, name, foreign)
            with self.subTest(field=name), self.assertRaises(ValueError): self.gate()
            setattr(self.args, name, original)


def original_formal(): return entry.original.FORMAL


if __name__ == '__main__': unittest.main()
