import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import queued_compile_delivery as subject


class QueuedCompileDeliveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.results = self.root/'results'; self.results.mkdir()
        self.paper = self.root/'paper'; self.paper.mkdir(); self.out = self.paper/'delivery'
        self.now = datetime(2026, 11, 16, tzinfo=timezone.utc).timestamp()
        self.base_ns = 10**18+7
        for target in ('subprocess.Popen', 'subprocess.run', 'os.system', 'socket.getaddrinfo',
                       'socket.create_connection', 'urllib.request.urlopen'):
            blocker = patch(target, side_effect=AssertionError('No subprocess or network in metadata export'))
            blocker.start(); self.addCleanup(blocker.stop)

    def write(self, path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')

    def entry(self, kind='qwen_compile', name='case'):
        return dict(id=name, case=str(self.results/name), kind=kind, model='fixture', parameter_group=kind)

    def process(self, ns=316419312689, **changes):
        value = dict(exit_code=0, timed_out=False, signal=None, started_ns=self.base_ns,
            ended_ns=self.base_ns+ns, signals_sent=[], remaining_pids=[], peak_rss_kib_by_pid={'9': 1234},
            limits=dict(memory_gib=32, file_gib=64, wall_seconds=7200))
        value.update(changes); return value

    def close(self, entry, record, status=None):
        case = Path(entry['case']); self.write(case/'result.json', record)
        elapsed = 318578592073
        self.write(case/'completion.json', dict(status=status or record['status'],
            required_artifacts_persisted=True, registry_persisted=True, timing_status='ok',
            started_ns=self.base_ns, ended_ns=self.base_ns+elapsed, complete_case_elapsed_ns=elapsed,
            complete_case_elapsed_seconds=elapsed/1e9,
            phase_spans=[dict(event_id='parent', stage_id='complete', parent_id=None,
                inclusive_ns=elapsed, exclusive_ns=2159279384, status=record['status']),
                dict(event_id='child', stage_id='compile', parent_id='parent',
                inclusive_ns=316419312689, exclusive_ns=316419312689, status=record['status'])]))

    def compile_record(self, kind='qwen_compile', **changes):
        phases = ('placement', 'emit', 'trace', 'compile') if kind == 'qwen_compile' else ('emit', 'trace', 'compile')
        workers = {p: dict(status='ok', phase=p) for p in phases}
        workers['trace']['process'] = self.process(ns=100)
        workers['compile'].update(process=self.process(), compiled_bootstraps=123,
            runtime_plan='/metadata/only.runtime-plan.json')
        value = dict(status='ok', attempted=dict.fromkeys(phases, 1), completed=dict.fromkeys(phases, 1),
            phase_results=workers, processes=[dict(phase=p, observation=self.process(ns=30)) for p in phases],
            compiled_bootstraps=123, runtime_plan='/metadata/only.runtime-plan.json',
            limits=dict(address_space_gib=32, subtree_rss_gib=14, minimum_available_gib=1.25),
            source=dict(profile='/original/qwen48/compiler-config.json'),
            scope='Saved frozen prefill source; no prior Agent clocks')
        if kind == 'bert_compile': value['phases'] = value.pop('phase_results')
        value.update(changes); return value

    def manifest(self, entries):
        return dict(schema=subject.MANIFEST_SCHEMA, results_root=str(self.results), cases=entries)

    def args(self, entries, **changes):
        manifest = self.root/'manifest.json'; self.write(manifest, self.manifest(entries))
        value = dict(manifest=manifest, results_root=self.results, paper_dir=self.paper, publish_dir=self.out,
            poll_seconds=60, until='2026-11-16T00:02:00Z', once=True)
        value.update(changes); return argparse.Namespace(**value)

    def test_missing_and_prepared_keep_compile_and_unstarted_counters_empty(self):
        entry = self.entry(); row = subject.observe(entry)
        self.assertEqual(row['result_status'], 'unobserved'); self.assertIsNone(row['model_native_compile_success'])
        self.assertIsNone(row['compiled_bootstraps']); self.assertIsNone(row['API_new_attempts'])
        self.close(entry, dict(status='prepared', compiled_bootstraps=999, api_calls=6))
        row = subject.observe(entry)
        self.assertFalse(row['closed']); self.assertFalse(row['admitted_closed_stage_success'])
        self.assertIsNone(row['reported_compile_count']); self.assertIsNone(row['API_new_attempts'])

    def test_result_without_completion_never_admits_compile_success(self):
        entry = self.entry(); self.write(Path(entry['case'])/'result.json', self.compile_record())
        row = subject.observe(entry)
        self.assertFalse(row['closure_verified']); self.assertIsNone(row['model_native_compile_success'])
        self.assertIsNone(row['compiled_bootstraps']); self.assertIsNone(row['native_compile_success_seconds'])

    def test_clean_closed_prefill_preserves_native_integer_ns_and_scope(self):
        entry = self.entry(); self.close(entry, self.compile_record()); row = subject.observe(entry)
        self.assertTrue(row['closure_verified']); self.assertTrue(row['model_native_compile_success'])
        self.assertEqual(row['compiled_bootstraps'], 123)
        self.assertEqual(row['native_compile_attempt_ns'], 316419312689)
        self.assertEqual(row['native_compile_success_ns'], 316419312689)
        self.assertEqual(row['elapsed_ns'], 318578592073)
        self.assertEqual(row['native_compile_attempt_seconds'], 316.419312689)
        self.assertIn('single-prefill', row['elapsed_scope'])
        self.assertEqual(row['native_parameters'][0]['profile_path'], '/original/qwen48/compiler-config.json')
        self.assertFalse(row['prediction_transferred_from_other_plan'])

    def test_closed_error_keeps_native_attempt_clock_and_success_clock_empty(self):
        entry = self.entry(); record = self.compile_record(status='error', error='RuntimeError: FAILED CANDIDATE TEXT')
        record['phase_results']['compile'].update(status='error', process=self.process(exit_code=-6, signal=6))
        record['completed']['compile'] = 0; self.close(entry, record); row = subject.observe(entry)
        self.assertFalse(row['model_native_compile_success']); self.assertIsNone(row['compiled_bootstraps'])
        self.assertEqual(row['native_compile_attempt_seconds'], 316.419312689)
        self.assertIsNone(row['native_compile_success_seconds'])
        self.assertEqual(row['failure']['error_class'], 'RuntimeError')
        self.assertEqual(row['failure']['category'], 'process_signal_origin_unresolved')
        self.assertNotIn('CANDIDATE', json.dumps(row))

    def test_timeout_guard_and_external_signal_remain_distinct(self):
        entry = self.entry()
        samples = (
            (dict(wall_timeout_observed=True, timeout_signal=14), 'recorded_timeout', None),
            (dict(guard_trigger=dict(available_ram_bytes=1), guard_stop_signal=15), 'recorded_resource_or_budget_guard', None),
            (dict(termination=dict(cause='incoming_signal', signal=15)), 'recorded_incoming_signal', 15))
        for changes, expected, incoming in samples:
            with self.subTest(expected=expected):
                self.close(entry, self.compile_record(status='error', **changes)); row = subject.observe(entry)
                self.assertEqual(row['failure']['category'], expected)
                self.assertEqual(row['failure']['incoming_signal'], incoming)
                self.assertFalse(row['model_native_compile_success'])

    def test_cleanup_actions_and_remaining_children_prevent_success(self):
        entry = self.entry()
        for changes in (dict(signals_sent=[dict(pid=9, signal=15)]), dict(remaining_pids=[9]), dict(timed_out=True)):
            with self.subTest(changes=changes):
                record = self.compile_record(); record['processes'][0]['observation'] = self.process(**changes)
                self.close(entry, record); row = subject.observe(entry)
                self.assertFalse(row['clean_children']); self.assertFalse(row['model_native_compile_success'])
                self.assertIsNone(row['native_compile_success_ns'])

    def test_completion_mismatch_and_incomplete_phase_counts_prevent_success(self):
        entry = self.entry(); record = self.compile_record(); record['completed']['emit'] = 0
        self.close(entry, record); self.assertFalse(subject.observe(entry)['model_native_compile_success'])
        self.close(entry, self.compile_record(), status='error')
        self.assertFalse(subject.observe(entry)['closure_verified'])

    def test_generic_interpreter_build_never_fills_model_compile_or_boots(self):
        entry = self.entry('orbit_interpreter'); record = dict(status='ok', process=self.process(),
            go_build_attempts=1, go_build_completed=1, runtime_interpreter_compiled=True,
            compiled_bootstraps=999, model_backend_compiled=True)
        self.close(entry, record); row = subject.observe(entry)
        self.assertTrue(row['runtime_interpreter_build_success']); self.assertIsNone(row['model_native_compile_success'])
        self.assertIsNone(row['compiled_bootstraps']); self.assertIsNone(row['reported_compile_count'])
        self.assertIsNone(row['native_compile_attempt_ns'])
        self.assertEqual(row['generic_interpreter_build_elapsed_ns'], 316419312689)

    def test_orbit_optimizer_placement_is_separate_from_compiled_count(self):
        entry = self.entry('orbit_optimizer'); record = dict(status='ok', native_optimizer_completed=True,
            placement_artifact_checked=True, attempted=dict(optimize=1, validate=1), completed=dict(optimize=1, validate=1),
            phase_results=dict(optimize=dict(status='ok', params=dict(poly_deg=131072, max_slot=65536)),
                validate=dict(status='ok', placement_bootstrap_nodes=20)),
            processes=[dict(phase=p, observation=self.process()) for p in ('optimize', 'validate')])
        self.close(entry, record); row = subject.observe(entry)
        self.assertTrue(row['admitted_closed_stage_success']); self.assertEqual(row['optimizer_placement_bootstrap_nodes'], 20)
        self.assertIsNone(row['model_native_compile_success']); self.assertIsNone(row['compiled_bootstraps'])
        self.assertEqual(row['native_parameters'][0]['values']['max_slot'], 65536)

    def test_bert_failure_recovers_saved_native_process_metadata_only(self):
        entry = self.entry('bert_compile'); record = self.compile_record('bert_compile', status='error')
        record['phases']['compile'] = dict(status='error'); record['completed']['compile'] = 0
        self.close(entry, record)
        self.write(Path(entry['case'])/'work/compile-process.json', self.process(ns=11930723115, exit_code=-6))
        row = subject.observe(entry)
        self.assertEqual(row['native_compile_attempt_ns'], 11930723115)
        self.assertIsNone(row['native_compile_success_ns']); self.assertFalse(row['model_native_compile_success'])

    def test_recovery_attempt_history_and_frozen_flag_do_not_claim_compile(self):
        entry = self.entry('qwen_recovery'); record = dict(status='ok', candidate_frozen=True, api_calls=4,
            session=dict(status='ok', done=True, frozen=True, calls=13, inherited_attempts=9, new_attempts=4),
            phases=[dict(phase='step', process=self.process(), worker=dict(status='ok', session=dict(calls=13)))])
        self.close(entry, record); row = subject.observe(entry)
        self.assertTrue(row['candidate_frozen']); self.assertEqual(row['API_new_attempts'], 4)
        self.assertEqual(row['session_attempts_after'], 13); self.assertIsNone(row['model_native_compile_success'])
        self.assertIn('inherited', row['elapsed_scope'])

    def test_bert_session_attempt_delta_is_not_an_http_counter(self):
        entry = self.entry('bert_continuation'); record = dict(status='error', frozen=False,
            source=dict(initial_session=dict(calls=3)), processes=[dict(phase='step', observation=self.process(exit_code=124),
                source_state_after=dict(phase='tune', calls=4, iteration=4))])
        self.close(entry, record); row = subject.observe(entry)
        self.assertEqual(row['session_attempt_delta'], 1); self.assertIsNone(row['API_new_attempts'])
        self.assertIsNone(row['model_native_compile_success']); self.assertFalse(row['candidate_frozen'])

    def test_actual_qwen_outer_phase_worker_shape_retains_last_state_and_counter_scopes(self):
        entry = self.entry('qwen_recovery')
        phases = [dict(phase='step', process=self.process(), worker=dict(status='ok', api_calls=1,
            candidate_frozen=False, session=dict(done=False, frozen=False, calls=10, inherited_attempts=9, new_attempts=1))),
            dict(phase='step', process=self.process(), worker=dict(status='ok', api_calls=1,
            candidate_frozen=True, session=dict(done=True, frozen=True, calls=11, inherited_attempts=9, new_attempts=2)))]
        record = dict(status='error', api_calls=2, candidate_frozen=False, phases=phases,
            attempted=dict(calibration=1, prompt=1, step=2), completed=dict(calibration=1, prompt=1, step=2))
        self.close(entry, record); row = subject.observe(entry)
        self.assertEqual(row['API_new_attempts'], 2); self.assertEqual(row['API_worker_reported_attempts'], 2)
        self.assertEqual(row['session_attempts_before'], 9); self.assertEqual(row['session_attempts_after'], 11)
        self.assertEqual(row['session_attempt_delta'], 2); self.assertEqual(row['session_reported_new_attempts'], 2)
        self.assertFalse(row['outer_candidate_frozen']); self.assertTrue(row['last_saved_worker_candidate_frozen'])
        self.assertTrue(row['frozen_observation_disagreement']); self.assertFalse(row['candidate_frozen'])
        del record['candidate_frozen']; del record['api_calls']; self.close(entry, record)
        row = subject.observe(entry); self.assertTrue(row['candidate_frozen'])
        self.assertIsNone(row['API_new_attempts']); self.assertEqual(row['API_worker_reported_attempts'], 2)

    def test_dacapo_native_clock_comes_from_compile_worker_not_outer_compile_process(self):
        entry = self.entry('dacapo_compile')
        record = dict(status='ok', bootstrap_count=20, compile_attempts=1, compile_completed=1,
            native_compile_attempts=1, native_compile_completed=1, compile_process=self.process(ns=999999999999),
            compile_worker=dict(status='ok', compile_attempts=1, compile_completed=1, process=self.process(ns=316419312689)))
        self.close(entry, record); row = subject.observe(entry)
        self.assertTrue(row['model_native_compile_success']); self.assertEqual(row['compiled_bootstraps'], 20)
        self.assertEqual(row['native_compile_attempt_ns'], 316419312689)
        self.assertEqual(row['native_compile_success_ns'], 316419312689)

    def test_source_gate_error_and_sensitive_fields_are_not_published(self):
        entry = self.entry(); record = dict(status='error', error_type='ValueError',
            error='ValueError: '+('FORBIDDEN SECRET PROGRAM '*10000), messages='FORBIDDEN messages',
            conversation={'raw': 'FORBIDDEN conversation'}, samples=[12345]*1000, source={'weights': 'FORBIDDEN'},
            limits={'address_space_gib': 32, 'unknown_payload': 'FORBIDDEN limits'})
        self.close(entry, record); row = subject.observe(entry)
        self.assertEqual(row['failure']['error_class'], 'ValueError')
        self.assertFalse(row['model_native_compile_success']); self.assertIsNone(row['native_compile_attempt_ns'])
        self.assertNotIn('FORBIDDEN', json.dumps(row)); self.assertNotIn('samples', row)

    def test_stage_integer_spans_are_preserved_without_summation(self):
        entry = self.entry(); self.close(entry, self.compile_record()); row = subject.observe(entry)
        self.assertEqual([s['inclusive_ns'] for s in row['stage_clocks']], [318578592073, 316419312689])
        self.assertEqual(row['stage_clocks'][1]['parent_id'], 'parent')
        self.assertEqual(row['elapsed_ns'], 318578592073)

    def test_only_dedicated_outputs_are_written_and_readiness_can_embed_manifest(self):
        entry = self.entry(); self.close(entry, self.compile_record()); args = self.args([entry])
        self.out.mkdir(); protected = self.out/'THREE_CNN_FOUR_SYSTEMS_EXPERIMENTS.json'; protected.write_text('unchanged')
        self.write(args.manifest, dict(schema=subject.READINESS_SCHEMA, manifest=self.manifest([entry]), note='ignored'))
        self.assertEqual(subject.execute(args, now=lambda: self.now), 1)
        self.assertEqual(protected.read_text(), 'unchanged')
        value = json.loads((self.out/(subject.BASE+'.json')).read_text()); self.assertTrue(value['cases'][0]['model_native_compile_success'])
        with (self.out/(subject.BASE+'.csv')).open() as stream: row = next(csv.DictReader(stream))
        self.assertEqual(row['native_compile_success_ns'], '316419312689')
        self.assertEqual(row['elapsed_ns'], '318578592073')

    def test_case_cap_duplicates_root_and_publication_paths_are_rejected(self):
        entry = self.entry()
        for entries in ([self.entry(name=str(i)) for i in range(9)], [entry, entry], [dict(entry, case=str(self.root/'foreign'))]):
            with self.subTest(entries=len(entries)), self.assertRaises(ValueError): subject.manifest_value(self.manifest(entries), self.results)
        with self.assertRaises(ValueError): subject.execute(self.args([entry], publish_dir=self.results/'export'), now=lambda: self.now)

    def test_finite_polling_stops_at_cutoff_and_expired_once_writes_nothing(self):
        entry = self.entry(); args = self.args([entry], once=False); clock = [self.now]; delays = []
        def sleep(seconds): delays.append(seconds); clock[0] += seconds
        with patch.object(subject, 'publish') as publish:
            self.assertEqual(subject.execute(args, now=lambda: clock[0], sleep=sleep), 2)
        self.assertEqual(publish.call_count, 2); self.assertEqual(delays, [60, 60])
        args = self.args([entry]); self.assertEqual(subject.execute(args, now=lambda: self.now+120), 0)
        for changes in (dict(poll_seconds=True), dict(poll_seconds=30), dict(until='2026-11-18T00:00:00Z')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                subject.execute(self.args([entry], **changes), now=lambda: self.now)

    def test_publication_symlink_is_refused(self):
        entry = self.entry(); args = self.args([entry]); self.out.mkdir()
        protected = self.paper/'protected'; protected.write_text('original')
        (self.out/(subject.BASE+'.json')).symlink_to(protected)
        with self.assertRaises(ValueError): subject.execute(args, now=lambda: self.now)
        self.assertEqual(protected.read_text(), 'original')


if __name__ == '__main__': unittest.main()
