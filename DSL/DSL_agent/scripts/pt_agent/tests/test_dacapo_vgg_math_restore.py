import argparse
import copy
from contextlib import ExitStack
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from pt_agent import experiments as experiment_package
from pt_agent.experiments import dacapo_vgg_math_restore as subject


class DacapoVGGMathRestoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.target = self.root/'original-case/emit/runtime.bundle'
        self.target.parent.mkdir(parents=True); self.stage = self.root/'staging'
        self.name = self.target.name; self.archive = self.target.with_name(self.name+'.tar.gz')
        self.payloads = {'a.bin': b'abc', 'sub/b.bin': b'defgh'}
        self.inventory = [dict(path=self.name, kind='directory', bytes=0),
            dict(path=self.name+'/a.bin', kind='file', bytes=3),
            dict(path=self.name+'/sub', kind='directory', bytes=0),
            dict(path=self.name+'/sub/b.bin', kind='file', bytes=5)]
        self.source = dict(bundle=str(self.target), archive=str(self.archive), inventory=self.inventory,
            regular_count=2, regular_bytes=8)
        for name in ('subprocess.Popen', 'subprocess.run', 'os.system', 'socket.getaddrinfo',
                     'socket.create_connection', 'urllib.request.urlopen'):
            blocker = patch(name, side_effect=AssertionError('Only tiny synthetic archives and fake runners are permitted'))
            blocker.start(); self.addCleanup(blocker.stop)

    def tar(self, members=None):
        if members is None:
            members = [(self.name, 'directory', None), (self.name+'/a.bin', 'file', b'abc'),
                (self.name+'/sub', 'directory', None), (self.name+'/sub/b.bin', 'file', b'defgh')]
        with tarfile.open(self.archive, 'w:gz') as stream:
            for name, kind, value in members:
                item = tarfile.TarInfo(name)
                if kind == 'directory':
                    item.type = tarfile.DIRTYPE; stream.addfile(item)
                elif kind in ('symlink', 'hardlink'):
                    item.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
                    item.linkname = value; stream.addfile(item)
                else:
                    item.size = len(value); stream.addfile(item, io.BytesIO(value))
        return self.archive.stat()

    def target_tree(self):
        self.target.mkdir(); (self.target/'sub').mkdir()
        for name, data in self.payloads.items():
            path = self.target/name; path.write_bytes(b'x'*len(data))

    def metadata(self, path):
        stat = Path(path).stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def assert_source_preserved(self, before):
        self.assertEqual(self.metadata(self.archive), (before.st_ino, before.st_size, before.st_mtime_ns))

    def restore(self, **changes):
        return subject.restore_bundle(self.source, self.stage, **changes)

    def test_inventory_normalized_root_and_file_metadata(self):
        checked = subject.validate_inventory(copy.deepcopy(self.inventory), self.name)
        self.assertEqual(set(checked), {x['path'] for x in self.inventory})
        self.assertEqual(checked[self.name+'/a.bin']['bytes'], 3)
        self.assertEqual(checked[self.name]['kind'], 'directory')

    def test_inventory_requires_root_directory_and_rejects_absolute_traversal_links(self):
        bad_sets = [self.inventory[1:]]
        for path in ('/'+self.name+'/a.bin', '../'+self.name+'/a.bin', self.name+'/../escaped', self.name+'/sub/../../escaped'):
            rows = copy.deepcopy(self.inventory); rows[1]['path'] = path; bad_sets.append(rows)
        for kind in ('symlink', 'hardlink'):
            rows = copy.deepcopy(self.inventory); rows[1]['kind'] = kind; bad_sets.append(rows)
        for rows in bad_sets:
            with self.subTest(rows=rows), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_inventory(rows, self.name)

    def test_inventory_rejects_duplicate_paths_and_size_metadata(self):
        for rows in (self.inventory+[copy.deepcopy(self.inventory[1])],
                     self.inventory+[dict(path=self.name+'/./a.bin', kind='file', bytes=3)]):
            with self.subTest(rows=rows), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_inventory(rows, self.name)
        for amount in (-1, True, 1.5):
            rows = copy.deepcopy(self.inventory); rows[1]['bytes'] = amount
            with self.subTest(amount=amount), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_inventory(rows, self.name)

    def test_legal_stream_archive_restores_complete_metadata_and_preserves_archive(self):
        before = self.tar(); result = self.restore()
        self.assertEqual(result['status'], 'restored')
        self.assertTrue(self.target.is_dir()); self.assertEqual((self.target/'a.bin').stat().st_size, 3)
        self.assertEqual((self.target/'sub/b.bin').stat().st_size, 5)
        self.assert_source_preserved(before)

    def test_existing_matching_target_returns_without_reading_unavailable_archive(self):
        self.target_tree(); before = self.metadata(self.target/'a.bin')
        self.source['archive'] = str(self.root/'intentionally-unavailable.tar.gz')
        result = self.restore()
        self.assertEqual(result['status'], 'already_present')
        self.assertEqual(self.metadata(self.target/'a.bin'), before)

    def test_existing_mismatched_target_is_preserved_without_overwrite(self):
        before_archive = self.tar(); self.target_tree(); path = self.target/'a.bin'; path.write_text('different-size')
        before_file = self.metadata(path)
        with self.assertRaises((ValueError, RuntimeError)): self.restore()
        self.assertEqual(self.metadata(path), before_file); self.assert_source_preserved(before_archive)

    def test_existing_target_symlink_is_refused(self):
        before = self.tar(); external = self.root/'external'; external.mkdir()
        self.target.symlink_to(external, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)): self.restore()
        self.assertTrue(self.target.is_symlink()); self.assertEqual(list(external.iterdir()), [])
        self.assert_source_preserved(before)

    def test_tar_absolute_and_parent_traversal_are_refused_before_promotion(self):
        for number, malicious in enumerate((str(self.root/'escaped'), self.name+'/../escaped', self.name+'/../../escaped')):
            with self.subTest(member=malicious):
                self.stage = self.root/('staging'+str(number))
                before = self.tar([(self.name, 'directory', None), (malicious, 'file', b'x')])
                with self.assertRaises((ValueError, RuntimeError)): self.restore()
                self.assertFalse(self.target.exists()); self.assertFalse((self.root/'escaped').exists())
                self.assert_source_preserved(before)

    def test_tar_symlink_and_hardlink_are_refused_even_for_expected_file_name(self):
        for number, kind in enumerate(('symlink', 'hardlink')):
            with self.subTest(kind=kind):
                self.stage = self.root/('stage-link'+str(number))
                before = self.tar([(self.name, 'directory', None), (self.name+'/a.bin', kind, self.name+'/sub/b.bin')])
                with self.assertRaises((ValueError, RuntimeError)): self.restore()
                self.assertFalse(self.target.exists()); self.assert_source_preserved(before)

    def test_duplicate_archive_member_is_refused_even_if_final_sizes_could_match(self):
        before = self.tar([(self.name, 'directory', None), (self.name+'/a.bin', 'file', b'abc'),
            (self.name+'/a.bin', 'file', b'xyz'), (self.name+'/sub', 'directory', None),
            (self.name+'/sub/b.bin', 'file', b'defgh')])
        with self.assertRaises((ValueError, RuntimeError)): self.restore()
        self.assertFalse(self.target.exists()); self.assert_source_preserved(before)

    def test_missing_archive_member_keeps_partial_stage_and_source(self):
        before = self.tar([(self.name, 'directory', None), (self.name+'/a.bin', 'file', b'abc')])
        with self.assertRaises((ValueError, RuntimeError)): self.restore()
        self.assertFalse(self.target.exists()); self.assertTrue(self.stage.is_dir())
        self.assertTrue(any(p.is_file() for p in self.stage.rglob('*')))
        self.assert_source_preserved(before)

    def test_size_mismatch_and_unexpected_extra_file_keep_failed_stage(self):
        cases = [
            [(self.name, 'directory', None), (self.name+'/a.bin', 'file', b'too-long')],
            [(self.name, 'directory', None), (self.name+'/a.bin', 'file', b'abc'),
             (self.name+'/sub', 'directory', None), (self.name+'/sub/b.bin', 'file', b'defgh'),
             (self.name+'/extra', 'file', b'x')]]
        for number, members in enumerate(cases):
            with self.subTest(case=number):
                self.stage = self.root/('failed-size'+str(number)); before = self.tar(members)
                with self.assertRaises((ValueError, RuntimeError)): self.restore()
                self.assertFalse(self.target.exists()); self.assertTrue(self.stage.is_dir())
                self.assert_source_preserved(before)

    def test_timeout_after_progress_keeps_partial_stage_and_archive(self):
        before = self.tar(); clock = [0.]; observed = []
        def progress(*args, **kwargs):
            observed.append(True); clock[0] = 301.
        with self.assertRaises(TimeoutError): self.restore(seconds=300, clock=lambda: clock[0], progress=progress)
        self.assertTrue(observed); self.assertTrue(self.stage.is_dir()); self.assertFalse(self.target.exists())
        self.assert_source_preserved(before)

    def test_progress_or_resource_check_failure_keeps_partial_and_no_target(self):
        before = self.tar()
        def fail(*args, **kwargs): raise RuntimeError('Synthetic physical resource stop')
        with self.assertRaisesRegex(RuntimeError, 'Synthetic physical resource stop'): self.restore(progress=fail)
        self.assertTrue(self.stage.is_dir()); self.assertFalse(self.target.exists()); self.assert_source_preserved(before)

    def test_existing_staging_root_is_preserved(self):
        self.tar(); self.stage.mkdir(); marker = self.stage/'preexisting'; marker.write_text('keep')
        before = self.metadata(marker)
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)): self.restore()
        self.assertEqual(self.metadata(marker), before); self.assertFalse(self.target.exists())

    def test_atomic_promotion_refuses_target_created_after_initial_check(self):
        before = self.tar(); competing = []
        def progress(*args, **kwargs):
            if not competing:
                self.target_tree(); competing.append(self.metadata(self.target/'a.bin'))
        with self.assertRaises((OSError, RuntimeError)): self.restore(progress=progress)
        self.assertEqual(self.metadata(self.target/'a.bin'), competing[0])
        self.assertTrue((self.stage/self.name).is_dir()); self.assert_source_preserved(before)

    def test_source_regular_count_and_total_size_must_match_inventory(self):
        self.tar()
        for key, value in (('regular_count', 3), ('regular_bytes', 9)):
            with self.subTest(field=key):
                source = copy.deepcopy(self.source); source[key] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    subject.restore_bundle(source, self.root/('invalid-'+key))
                self.assertFalse(self.target.exists())

    def test_math_command_reuses_original_entry_and_fixed_math_resource_contract(self):
        args = argparse.Namespace(compile_case=self.root/'compiled', archive_case=self.root/'archived', math_out=self.root/'math')
        command = subject.math_command(args)
        self.assertEqual(command, [sys.executable, '-B', '-m', 'pt_agent.experiments.dacapo_vgg16_plan_math',
            '--compile-case', str(args.compile_case), '--archive-case', str(args.archive_case),
            '--out', str(args.math_out), '--seconds', '1800', '--memory-gib', '4',
            '--case-gib', '0.5', '--minimum-free-gib', '10'])
        self.assertNotIn('--report', command); self.assertNotIn('--bias-corrected', command)

    def resource_fixture(self, *, own=None, staging=None, free=None, available=None, subtree=None, supervisor=None, now=0):
        gib = subject.GIB; out = self.root/'metadata'; stage = out/'work/staging'
        values = dict(own=gib//4 if own is None else own, staging=5*gib if staging is None else staging,
            free=20*gib if free is None else free, available=2*gib if available is None else available,
            subtree=12*gib if subtree is None else subtree, supervisor=gib if supervisor is None else supervisor)
        def tree(path, **kwargs):
            if Path(path) == out:
                self.assertEqual(kwargs.get('excluded'), stage); return values['own']
            self.assertEqual(Path(path), stage); return values['staging']
        with patch.object(subject, '_tree_bytes', side_effect=tree), patch.object(subject, '_latest_rss',
                return_value=dict(subtree_rss_bytes=values['subtree'], supervisor_rss_bytes=values['supervisor'])):
            return subject.resource_check(out, stage, self.root/'resource.jsonl', 100,
                clock=lambda: now, disk_usage=lambda _: argparse.Namespace(free=values['free']), available_ram=lambda: values['available'])

    def test_own_staging_and_physical_resource_observations_remain_separate(self):
        observation = self.resource_fixture()
        self.assertEqual(observation['own_artifact_bytes'], subject.GIB//4)
        self.assertEqual(observation['staging_bytes'], 5*subject.GIB)
        self.assertEqual(observation['rss']['subtree_rss_bytes'], 12*subject.GIB)
        self.assertEqual(observation['rss']['supervisor_rss_bytes'], subject.GIB)

    def test_each_independent_resource_limit_retains_observation_on_failure(self):
        gib = subject.GIB
        changes = [dict(own=gib//2+1), dict(staging=6*gib+1), dict(free=10*gib-1),
            dict(available=5*gib//4-1), dict(subtree=14*gib, supervisor=1), dict(now=100)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(RuntimeError) as raised:
                self.resource_fixture(**change)
            self.assertTrue(raised.exception.observation['reasons'])

    def fake_execute(self, *, restore_exit=0, worker_status='ok', admission_failure=False):
        results = self.root/'results'; results.mkdir(exist_ok=True)
        args = argparse.Namespace(out=results/'restore-case', math_out=results/'math-case',
            compile_case=self.root/'compiled', archive_case=self.root/'archived', registry=results/'registry.jsonl')
        calls = []; accepted = []
        def runner(argv, **options):
            calls.append(dict(argv=argv, **options))
            if '--restore-worker' in argv:
                target = Path(argv[argv.index('--worker-result')+1]); target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(dict(status=worker_status, elapsed_ns=1, archive_restores=1,
                    archive_restore_attempts=1, restore=dict(status='restored'))))
                if restore_exit == 0 and worker_status == 'ok': self.target_tree()
                return dict(exit_code=restore_exit, timed_out=False, signal=None, remaining_pids=[], signals_sent=[])
            return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[], signals_sent=[])
        def resources(*args, **kwargs):
            if admission_failure:
                error = RuntimeError('Synthetic physical stop'); error.observation = dict(reasons=['fixture RSS'], disk_free_bytes=30*subject.GIB)
                raise error
            return dict(own_artifact_bytes=0, staging_bytes=0, disk_free_bytes=30*subject.GIB,
                available_ram_bytes=2*subject.GIB, rss=None)
        def accept(*args):
            accepted.append(True); return dict(status='ok', result=str(args[0].math_out/'result.json'))
        with ExitStack() as stack:
            stack.enter_context(patch.object(subject.paths, 'RESULTS_DIR', results))
            stack.enter_context(patch.object(subject, 'source_gate', return_value=copy.deepcopy(self.source)))
            stack.enter_context(patch.object(subject, '_parent_limits', return_value=dict(fixture=True)))
            stack.enter_context(patch.object(subject, 'resource_check', side_effect=resources))
            stack.enter_context(patch.object(subject, '_register'))
            stack.enter_context(patch.object(subject, '_accept_math', side_effect=accept))
            stack.enter_context(patch.object(subject, 'run_process', side_effect=runner))
            stack.enter_context(patch.object(subject.shutil, 'copy2'))
            thread = stack.enter_context(patch.object(subject.threading, 'Thread'))
            thread.return_value.is_alive.return_value = False
            result = subject.execute(args)
        return args, result, calls, accepted

    def test_fake_execute_restore_and_original_math_use_distinct_parent_resources(self):
        args, result, calls, accepted = self.fake_execute()
        self.assertEqual(result['status'], 'ok'); self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['mem_gib'], 1); self.assertEqual(calls[0]['timeout'], 300)
        self.assertEqual(calls[1]['mem_gib'], 24); self.assertLessEqual(calls[1]['timeout'], 2040)
        self.assertGreater(calls[1]['timeout'], 0); self.assertEqual([c['fsize_gib'] for c in calls], [1, 1])
        self.assertEqual(calls[1]['argv'], subject.math_command(args)); self.assertEqual(len(accepted), 1)
        self.assertEqual(result['math_attempts'], 1); self.assertEqual(result['math_completed'], 1)
        self.assertTrue((args.out/'completion.json').is_file())

    def test_restore_process_failure_stops_before_any_math_and_preserves_closure(self):
        args, result, calls, accepted = self.fake_execute(restore_exit=1, worker_status='error')
        self.assertEqual(result['status'], 'error'); self.assertEqual(len(calls), 1)
        self.assertEqual(result['math_attempts'], 0); self.assertEqual(result['math_completed'], 0)
        self.assertEqual(accepted, []); self.assertTrue((args.out/'completion.json').is_file())

    def test_clean_restore_process_with_failed_worker_stops_before_math(self):
        _, result, calls, accepted = self.fake_execute(worker_status='error')
        self.assertEqual(result['status'], 'error'); self.assertEqual(len(calls), 1)
        self.assertEqual(result['math_attempts'], 0); self.assertEqual(accepted, [])

    def test_synchronous_resource_stop_preserves_classification_and_launches_nothing(self):
        args, result, calls, accepted = self.fake_execute(admission_failure=True)
        self.assertEqual(result['status'], 'resource_stopped'); self.assertEqual(calls, [])
        self.assertEqual(accepted, []); self.assertEqual(result['restore_worker_attempts'], 0)
        self.assertEqual(result['math_attempts'], 0); self.assertEqual(result['guard_trigger']['reasons'], ['fixture RSS'])
        self.assertTrue((args.out/'completion.json').is_file())

    def archive_gate_fixture(self):
        self.tar(); case = self.root/'closed-archive'; case.mkdir()
        (case/'manifest.json').write_text(json.dumps(dict(files=[dict(kind='runtime_bundle',
            path=str(self.target), inventory=self.inventory)])))
        record = dict(round='R34-completed', status='ok', elapsed_ns=123456, ended_utc='2026-11-16T00:00:00Z',
            guard_errors=[], in_progress=None, source_result_unchanged=True, selected_files=1, completed_files=1,
            files=[dict(kind='runtime_bundle', status='compressed', source=str(self.target), archive=str(self.archive),
                raw_bytes=8, compressed_bytes=self.archive.stat().st_size)])
        (case/'result.json').write_text(json.dumps(record))
        row = dict(run_id=case.name, round='R34-completed', model='fixture', mode='reversible-completed-case-archive',
            git_rev=None, metrics=dict(status='ok', completed_files=1), artifacts_dir=str(case), ts='fixture')
        registry = self.root/'archive-registry.jsonl'; registry.write_text(json.dumps(row)+'\n')
        return argparse.Namespace(archive_case=case, compile_case=self.root/'compiled', registry=registry), row

    def test_source_gate_accepts_real_archive_terminal_protocol_without_completion_marker(self):
        args, _ = self.archive_gate_fixture(); name = 'pt_agent.experiments.dacapo_vgg16_plan_math'
        existing = types.ModuleType(name); checks = []
        def original_gate(args, *, require_restored):
            checks.append(require_restored); return dict(bundle=str(self.target), restored_bundle_available=False), {}
        existing.source_gate = original_gate
        with patch.dict(sys.modules, {name: existing}), patch.object(experiment_package, 'dacapo_vgg16_plan_math', existing, create=True), \
                patch.object(subject, 'EXPECTED_FILES', 2), patch.object(subject, 'EXPECTED_BYTES', 8), \
                patch.object(subject.tarfile, 'open', side_effect=AssertionError('Source gate must not read archive payload')):
            checked = subject.source_gate(args)
        self.assertEqual(checks, [False]); self.assertFalse((args.archive_case/'completion.json').exists())
        self.assertEqual(checked['regular_count'], 2); self.assertEqual(checked['regular_bytes'], 8)
        self.assertEqual(checked['payload_reads_in_source_gate'], 0)

    def test_source_archive_registry_must_retain_unique_original_eight_field_row(self):
        args, row = self.archive_gate_fixture(); name = 'pt_agent.experiments.dacapo_vgg16_plan_math'
        existing = types.ModuleType(name); existing.source_gate = lambda *args, **kwargs: (dict(bundle=str(self.target)), {})
        with patch.dict(sys.modules, {name: existing}), patch.object(experiment_package, 'dacapo_vgg16_plan_math', existing, create=True), \
                patch.object(subject, 'EXPECTED_FILES', 2), patch.object(subject, 'EXPECTED_BYTES', 8):
            args.registry.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError): subject.source_gate(args)
            del row['git_rev']; args.registry.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): subject.source_gate(args)

    def test_math_source_readback_changes_only_restored_availability(self):
        math = self.root/'math-readback'; math.mkdir(); name = 'pt_agent.experiments.dacapo_vgg16_plan_math'
        existing = types.ModuleType(name); existing.ROUND = 'synthetic-existing-math'; accepted = []
        existing.accepted_worker = lambda row: accepted.append(row)
        source = dict(official_source=dict(bundle=str(self.target), restored_bundle_available=False, original_reference='unchanged'))
        observed = dict(source['official_source'], restored_bundle_available=True)
        record = dict(round=existing.ROUND, model=subject.MODEL, status='ok', worker_attempts=1, worker_completed=1,
            source=observed, worker=dict(synthetic_math_passed=True))
        (math/'result.json').write_text(json.dumps(record))
        (math/'completion.json').write_text(json.dumps(dict(status='ok', required_artifacts_persisted=True,
            registry_persisted=True, complete_case_elapsed_seconds=.01)))
        with patch.dict(sys.modules, {name: existing}), patch.object(experiment_package, 'dacapo_vgg16_plan_math', existing, create=True):
            self.assertEqual(subject._accept_math(argparse.Namespace(math_out=math), source)['status'], 'ok')
            self.assertFalse(source['official_source']['restored_bundle_available'])
            record['source']['original_reference'] = 'changed'; (math/'result.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError): subject._accept_math(argparse.Namespace(math_out=math), source)
        self.assertEqual(len(accepted), 1)

    def test_running_restore_zero_checkpoint_keeps_actual_total_unknown(self):
        record = dict(status='running', archive_restore_attempts=1, archive_restores=0)
        observed = subject._restore_observation(record)
        self.assertEqual(observed['archive_restore_attempts'], 1)
        self.assertIsNone(observed['archive_restores'])
        self.assertFalse(observed['restore_terminal_counters_available'])
        record['status'] = 'error'
        self.assertIsNone(subject._restore_observation(record)['archive_restores'])

    def test_recorded_promotion_one_and_terminal_zero_have_distinct_evidence(self):
        promoted = subject._restore_observation(dict(status='running', archive_restore_attempts=1,
            archive_restores=0, restore=dict(promotion_succeeded=True)))
        self.assertEqual(promoted['archive_restores'], 1)
        self.assertFalse(promoted['restore_terminal_counters_available'])
        terminal = subject._restore_observation(dict(status='error', elapsed_ns=1,
            archive_restore_attempts=1, archive_restores=0, restore=dict(promotion_succeeded=False)))
        self.assertEqual(terminal['archive_restores'], 0)
        self.assertTrue(terminal['restore_terminal_counters_available'])


if __name__ == '__main__': unittest.main()
