"""Bounded source/worker/closure fixtures; no models, solvers or native tools."""
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths, registry
from pt_agent.experiments import orbit_rule63_compile as d
from pt_agent.experiments import cold_rule63_whole_model as cold
from pt_agent.experiments import dacapo_compile_continuation as limits
from pt_agent.experiments import closed_dacapo_trace_archive as lease


def process(**changes):
    value = dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[], signals_sent=[],
                 started_ns=400, ended_ns=1100)
    value.update(changes)
    return value


class OrbitCompile(unittest.TestCase):
    def setUp(self):
        self.deadline = patch.object(d, 'RESNET_DEADLINE', datetime(2099, 1, 1, tzinfo=timezone.utc))
        self.deadline.start(); self.addCleanup(self.deadline.stop)

    def fixture(self, root, model='resnet20', archived=False):
        source = root/('a2b-'+model+'-rule63-fresh-cold-depth-20261117')
        (source/'source').mkdir(parents=True)
        (source/'work/emit').mkdir(parents=True)
        raw = source/'work/emit/candidate_trace.mlir'
        if archived:
            raw.with_name(raw.name+'.gz').write_bytes(gzip.compress(b'module { /* own fixture */ }\n'))
        else: raw.write_text('module { /* own fixture */ }\n')
        profile = root/'profile.json'
        profile.write_text(json.dumps(dict(polynomialDegree=65536, rescalingFactor=43,
            levelLowerBound=1, levelUpperBound=8, bootstrapLevelLowerBound=1,
            bootstrapLevelUpperBound=5, runtime='HEAAN-HEVM', latencyTable={})))
        (source/'source/compiler-profile.json').write_text(profile.read_text())
        orbit_root = root/'orbit'; orbit_root.mkdir()
        python = root/'orbit-python'; python.write_text('fixture executable'); python.chmod(0o700)
        license_file = root/'license'; license_file.touch()
        args = d.arguments(['--model', model, '--source-case', str(source), '--out', str(root/'new-case'),
            '--profile', str(profile), '--paper-runs', str(root/'paper.jsonl'),
            '--orbit-root', str(orbit_root), '--orbit-python', str(python), '--license-file', str(license_file)])
        phases = {name: dict(status='ok') for name in cold.PHASES}
        phases['generation'].update(rule_options=dict(d.shared.OPTIONS), raw_Count=dict(stats=dict(counts=dict(boot=0))),
            training_math=dict(samples=3, checks=[dict(passed=True) for _ in range(3)]))
        stats = dict(counts=dict(boot=0, add=1), ops_total=1, distinct_consts=0)
        phases['generation'].update(raw_Count=dict(stats=stats), signature=dict(passed=True))
        phases['emit']['emit_stats'] = dict(stats)
        phases['trace']['trace_bytes'] = 28
        phases['compile']['native_process'] = process(started_ns=500, ended_ns=1000)
        source_result = dict(round=cold.ROUND, mode=cold.MODE, model=args.model, run_id=source.name,
            placement='depth', status='ok', monitor_stopped=True, guard_error=None, monitor_error=None,
            git_rev='fixture', expected_bootstraps_match=True,
            attempted=dict.fromkeys(cold.PHASES, 1), completed=dict.fromkeys(cold.PHASES, 1),
            phases=phases, processes=[dict(phase=p, observation=process()) for p in cold.PHASES])
        completion = dict(round=cold.ROUND, status='ok', registry_persisted=True,
            required_artifacts_persisted=True, started_ns=100, ended_ns=1200)
        row = dict(run_id=source.name, round=cold.ROUND, mode=cold.MODE, model=args.model,
            git_rev='fixture', artifacts_dir=str(source), metrics=dict(status='ok'), ts='fixture')
        (source/'result.json').write_text(json.dumps(source_result))
        (source/'completion.json').write_text(json.dumps(completion))
        (root/'registry.jsonl').write_text(json.dumps(row)+'\n')
        if archived:
            self.archive_fixture(root, source, raw, phases['trace']['trace_bytes'])
        return args, source_result

    def archive_fixture(self, root, source, raw, projection):
        case = root/(source.name+'-raw-npz-resource-archive'); case.mkdir()
        archive = raw.with_name(raw.name+'.gz')
        value = dict(round='A-GROUP-COLD-CLOSED-RESOURCE-ARCHIVE', status='ok', source_kind=source.name,
            source_case=str(source), scientific_status='ok', resource_status='archived_or_kept',
            in_progress=None, files=[dict(source=str(raw), archive=str(archive), status='compressed',
                raw_bytes=projection, compressed_bytes=archive.stat().st_size)])
        marker = dict(round=value['round'], status='ok', registry_persisted=True, required_artifacts_persisted=True)
        row = dict(run_id=case.name, round=value['round'], model=source.name,
            mode='resource-only-reversible-gzip', git_rev='fixture', metrics=dict(status='ok'),
            artifacts_dir=str(case), ts='fixture')
        (case/'result.json').write_text(json.dumps(value)); (case/'completion.json').write_text(json.dumps(marker))
        with (root/'registry.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

    def gate(self, args):
        with patch.object(cold, 'admission', return_value=dict(compiler='/fixture/hecate-opt')), \
             patch.object(d.closed, 'producer_births', return_value=9), \
             patch.object(d.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*d.GIB)), \
             patch.object(lease, 'lease_observation', return_value=dict(owner=dict(pid=7, starttime=99))):
            return d.source_gate(args)

    def test_strict_source_accepts_raw_and_archived_forms(self):
        for archived in (False, True):
            with self.subTest(archived=archived), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                args, _ = self.fixture(Path(tmp), archived=archived)
                proof = self.gate(args)
                self.assertEqual(proof['source_input_kind'], 'gzip' if archived else 'raw')
                self.assertEqual(proof['shared_generation_prefix_ns'], 400)
                self.assertEqual(proof['source_parameter_group'], 'ours-N16-boot5')
                self.assertEqual(proof['source_payload_reads'], 0)

    def test_source_rejects_changed_identity_math_profile_clock_and_registry(self):
        for failure in ('model', 'math', 'profile', 'clock', 'registry', 'phase'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                root = Path(tmp); args, value = self.fixture(root)
                if failure == 'model': value['model'] = 'alexnet_silu_avgpool'
                elif failure == 'math': value['phases']['generation']['training_math']['checks'][0]['passed'] = False
                elif failure == 'profile': (args.source_case/'source/compiler-profile.json').write_text('{}')
                elif failure == 'clock': value['phases']['compile']['native_process']['started_ns'] = 20
                elif failure == 'registry':
                    with (root/'registry.jsonl').open('a') as f: f.write((root/'registry.jsonl').read_text())
                elif failure == 'phase': value['completed']['emit'] = 0
                (args.source_case/'result.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError): self.gate(args)

    def test_source_rejects_live_birth_or_foreign_lease_before_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp))
            with patch.object(cold, 'admission', return_value={}), \
                 patch.object(d.closed, 'producer_births', side_effect=ValueError('live producer')):
                with self.assertRaisesRegex(ValueError, 'live producer'): d.source_gate(args)
            with patch.object(cold, 'admission', return_value={}), \
                 patch.object(d.closed, 'producer_births', return_value=2), \
                 patch.object(lease, 'lease_observation', side_effect=ValueError('foreign owner')):
                with self.assertRaisesRegex(ValueError, 'foreign owner'): d.source_gate(args)
            self.assertFalse(args.out.exists())

    def test_snapshot_restores_only_new_copy_with_zero_footer_checks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp), archived=True); proof = self.gate(args)
            original = Path(proof['source_input']); before = d.raw_stat(original)
            args.out.mkdir(); (args.out/'work').mkdir()
            with patch.object(d.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*d.GIB)):
                d.snapshot(args, proof)
            self.assertFalse(Path(proof['worker_input']).is_symlink())
            self.assertEqual(proof['input_preparation']['CRC_checks'], 0)
            self.assertEqual(proof['input_preparation']['ISIZE_checks'], 0)
            self.assertGreaterEqual(proof['input_preparation_s'], 0)
            self.assertGreaterEqual(proof['input_preparation_ended_ns'], proof['input_preparation_started_ns'])
            self.assertEqual(d.raw_stat(original), before)
            self.assertFalse((args.source_case/'work/emit/candidate_trace.mlir').exists())
            self.assertFalse((args.out/'work/emit/constants.npz').exists())

    def test_snapshot_raw_link_leaves_original_identity_and_nlink(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp)); proof = self.gate(args)
            before = d.raw_stat(proof['source_input'])
            args.out.mkdir(); (args.out/'work').mkdir(); d.snapshot(args, proof)
            self.assertTrue(Path(proof['worker_input']).is_symlink())
            self.assertEqual(d.raw_stat(proof['source_input']), before)

    def test_worker_argv_only_explicit_paths_and_preserves_venv_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            command, seconds = d.phase_command(args, 'compile', 7500)
            self.assertEqual(command[0], str(args.orbit_python))
            self.assertEqual(command[1:4], ['-B', '-m', d.MODULE])
            self.assertIn('--source-proof', command); self.assertNotIn('--source-case', command)
            self.assertNotIn('--paper-runs', command); self.assertEqual(seconds, 7320)
            self.assertEqual(command[command.index('--native-seconds')+1], '7200')
            with self.assertRaises(ValueError): d.phase_command(args, 'trace', 7500)

    def test_adapter_acceptance_requires_actual_verified_artifacts_and_finite_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); artifact = work/'optimized.mlir'; artifact.write_text('fixture')
            value = dict(status='ok', compiled_bootstraps=2, execution_steps=12, compile_verified=True,
                parameter_contract_verified=True, bootstrap_weighted_matches_serialized=True,
                source_operator_constant_bijection_verified=True, official_decoded_graph_check_passed=True,
                artifacts=dict(optimized_mlir=str(artifact)), prediction=dict(pred_latency_s=None,
                pred_latency_without_upscale_s=3, known_cost_partial_s=2))
            self.assertIs(d.accept_adapter(value, work), value)
            for change in ('compile_verified', 'artifact', 'infinite', 'bool_count'):
                copy = json.loads(json.dumps(value))
                if change == 'compile_verified': copy['compile_verified'] = False
                if change == 'artifact': copy['artifacts']['optimized_mlir'] = '/outside/path'
                if change == 'infinite': copy['prediction']['pred_latency_s'] = float('inf')
                if change == 'bool_count': copy['compiled_bootstraps'] = True
                with self.assertRaises(ValueError): d.accept_adapter(copy, work)

    def output(self, work):
        artifact = work/'orbit-adapter/optimized.mlir'; artifact.parent.mkdir()
        artifact.write_text('own tiny serialized output')
        return dict(status='ok', compiled_bootstraps=2, execution_steps=12, compile_verified=True,
            parameter_contract_verified=True, bootstrap_weighted_matches_serialized=True,
            source_operator_constant_bijection_verified=True, official_decoded_graph_check_passed=True,
            artifacts=dict(optimized_mlir=str(artifact)), prediction=dict(pred_latency_s=4,
                pred_latency_without_upscale_s=3, known_cost_partial_s=4, missing=[]),
            adapter_scope=dict(fixed_bootstrap='1->5', upscale='mul proxy'))

    def test_real_worker_dispatch_keeps_explicit_source_and_new_adapter_directory(self):
        from pt_agent.experiments import orbit_rule63_adapter as adapter
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp)); proof = self.gate(args)
            args.out.mkdir(); (args.out/'work').mkdir(); d.snapshot(args, proof)
            work = args.out/'work/compile'; work.mkdir()
            worker_args = d.arguments(['--worker', '--model', 'resnet20', '--work-dir', str(work),
                '--profile', str(args.profile), '--orbit-root', str(args.orbit_root),
                '--source-proof', str(args.out/'source/worker-source-proof.json')])
            seen = []
            def compile(**kw):
                seen.append(kw); self.assertFalse(kw['work_dir'].exists())
                partial = d.read(work/'phase_result.json')
                self.assertEqual(partial['status'], 'running')
                self.assertIsInstance(partial['compile_started_ns'], int)
                self.assertEqual(kw['active_slots'], 16384); self.assertEqual(kw['threads'], 1)
                kw['checkpoint'](); return self.output(work)
            with patch.object(adapter, 'compile_rule63', side_effect=compile), \
                 patch.object(d, 'registry_rows', side_effect=AssertionError('worker registry read')):
                self.assertEqual(d.worker(worker_args), 0)
            result = d.read(work/'phase_result.json')
            self.assertEqual(result['compiled_bootstraps'], 2); self.assertEqual(len(seen), 1)
            self.assertEqual(result['runtime_plan'], None)

    def fake_outer(self, root, *, failed=False, rejected=False):
        args, source = self.fixture(root)
        if rejected:
            source['monitor_stopped'] = False
            (args.source_case/'result.json').write_text(json.dumps(source))
        calls = []
        def runner(argv, **kw):
            calls.append(argv)
            self.assertEqual(kw['mem_gib'], 24); self.assertEqual(kw['fsize_gib'], 16)
            work = args.out/'work/compile'
            output = self.output(work)
            value = dict(round=d.ROUND, mode=d.MODE, phase='compile', model=args.model,
                status='error' if failed else 'ok', error='own fixture failure' if failed else None,
                adapter=output, compile_started_ns=2000, compile_ended_ns=3000,
                compiled_bootstraps=2, execution_steps=12, prediction=output['prediction'],
                predicted_latency_us=4e6, runtime_plan=None)
            (work/'phase_result.json').write_text(json.dumps(value))
            return process(exit_code=1 if failed else 0)
        with patch.object(cold, 'admission', return_value=dict(compiler='/fixture/current')), \
             patch.object(d.closed, 'producer_births', return_value=9), \
             patch.object(lease, 'lease_observation', return_value=dict(owner=dict(pid=7, starttime=99))), \
             patch.object(cold, 'run_process', side_effect=runner), \
             patch.object(registry, 'git_rev', return_value='fixture'), \
             patch.object(limits, 'available_ram', return_value=20*d.GIB), \
             patch.object(cold.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*d.GIB)):
            result = d.execute(args)
        return args, result, calls

    def test_actual_private_outer_closure_success_and_failure_rows(self):
        previous = (cold.ROUND, cold.MODE, cold.PHASES, cold.execute.__code__)
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                args, result, calls = self.fake_outer(Path(tmp), failed=failed)
                self.assertEqual(result['status'], 'error' if failed else 'ok')
                self.assertEqual(result['compile_accepted'], not failed)
                self.assertEqual(len(calls), 1)
                self.assertTrue(d.read(args.out/'completion.json')['registry_persisted'])
                rows = [json.loads(t) for t in (Path(tmp)/'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2); self.assertEqual(set(rows[-1]), d.closed.REGISTRY_FIELDS)
                row = d.read(args.out/'paper-record.json')
                self.assertEqual(row['compile_s'], 1e-6)
                self.assertAlmostEqual(row['generation_e2e_s'], 1.4e-6)
                self.assertEqual(row['pred_latency_without_upscale_s'], 3)
                self.assertFalse(row['cold']); self.assertTrue(row['estimated_from_shared_generation'])
                self.assertEqual(len((Path(tmp)/'paper.jsonl').read_text().splitlines()), 1)
        self.assertEqual(previous, (cold.ROUND, cold.MODE, cold.PHASES, cold.execute.__code__))

    def test_source_rejection_persists_zero_attempts_and_no_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, result, calls = self.fake_outer(Path(tmp), rejected=True)
            self.assertEqual(calls, []); self.assertEqual(result['attempted'], {})
            self.assertFalse(result['compile_accepted'])
            row = d.read(args.out/'paper-record.json')
            self.assertIsNone(row['compile_s']); self.assertIsNone(row['generation_e2e_s'])
            self.assertEqual(row['status'], 'driver_error')

    def test_resnet_deadline_and_later_models_require_accepted_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            with patch.object(d, 'RESNET_DEADLINE', datetime(2000, 1, 1, tzinfo=timezone.utc)):
                with self.assertRaisesRegex(ValueError, 'deadline'): d.prerequisite_resnet(args)
            args.short_model = 'alexnet'
            with self.assertRaisesRegex(ValueError, 'explicit successful ResNet'): d.prerequisite_resnet(args)

    def test_preserves_timeout_resource_and_compiler_signal_categories(self):
        base = dict(status='error', guard_error=None, phases={}, processes=[], error='own failure')
        self.assertEqual(d.classification(base), 'driver_error')
        self.assertEqual(d.classification(dict(base, phases={'compile': dict(error=None)})), 'driver_error')
        self.assertEqual(d.classification(dict(base, phases={'compile': dict(error='TimeoutError: allowance')})), 'timeout')
        self.assertEqual(d.classification(dict(base, guard_error='RSS14')), 'resource_limit')
        self.assertEqual(d.classification(dict(base, processes=[dict(observation=process(exit_code=-11))])), 'compiler_crash')

    def test_later_model_requires_real_closed_on_time_resnet_artifacts(self):
        for failure in (None, 'late', 'worker', 'proof'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                root = Path(tmp); args, _ = self.fixture(root, model='alexnet')
                case = root/'resnet-accepted'; args.resnet_case = case
                (case/'source').mkdir(parents=True); (case/'work/compile').mkdir(parents=True)
                (case/'source/compiler-profile.json').write_text(args.profile.read_text())
                output = self.output(case/'work/compile')
                value = dict(round=d.ROUND, mode=d.MODE, model=d.MODELS['resnet20'], status='ok',
                    compile_accepted=True, monitor_stopped=True, attempted={'compile': 1}, completed={'compile': 1},
                    accepted_utc='2098-12-31T00:00:00+00:00', phases={'compile': dict(adapter=output)},
                    processes=[dict(phase='compile', observation=process())],
                    source=dict(source_parameter_group='ours-N16-boot5'))
                completion = dict(round=d.ROUND, status='ok', registry_persisted=True,
                    required_artifacts_persisted=True)
                row = dict(run_id=case.name, round=d.ROUND, model=d.MODELS['resnet20'], mode=d.MODE,
                    git_rev='fixture', metrics=dict(status='ok'), artifacts_dir=str(case), ts='fixture')
                if failure == 'late': value['accepted_utc'] = '2099-01-02T00:00:00+00:00'
                if failure == 'worker': value['processes'][0]['observation']['remaining_pids'] = [42]
                if failure == 'proof': value['phases']['compile']['adapter']['official_decoded_graph_check_passed'] = False
                (case/'result.json').write_text(json.dumps(value)); (case/'completion.json').write_text(json.dumps(completion))
                with (root/'registry.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                with patch.object(d.closed, 'producer_births', return_value=4):
                    if failure:
                        with self.assertRaises(ValueError): d.prerequisite_resnet(args)
                    else: self.assertEqual(d.prerequisite_resnet(args)['case'], str(case))

    def test_archive_provenance_closed_births_projection_and_math_inventory_reject(self):
        for failure in ('archive_status', 'archive_record', 'projection', 'signature', 'emit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                root = Path(tmp); args, value = self.fixture(root, archived=True)
                archive_case = root/(args.source_case.name+'-raw-npz-resource-archive')
                archive = d.read(archive_case/'result.json')
                if failure == 'archive_status': archive['status'] = 'running'
                if failure == 'archive_record': archive['files'][0]['archive'] = '/foreign.gz'
                if failure == 'projection': value['phases']['trace']['trace_bytes'] = 17*d.GIB
                if failure == 'signature': value['phases']['generation']['signature']['passed'] = False
                if failure == 'emit': value['phases']['emit']['emit_stats']['ops_total'] = 2
                (archive_case/'result.json').write_text(json.dumps(archive))
                (args.source_case/'result.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError): self.gate(args)

    def test_interrupted_compile_retains_observed_failed_clock_and_no_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            value = dict(status='error', guard_error=None, phases=dict(compile=dict(
                status='running', error=None, compile_started_ns=2000)),
                processes=[dict(phase='compile', observation=process(ended_ns=4000, exit_code=-15, timed_out=True))],
                error='worker timeout', source=dict(shared_generation_prefix_ns=400), compile_accepted=False)
            row = d.paper_record(args, value, dict(complete_case_elapsed_seconds=3), {})
            self.assertEqual(row['status'], 'timeout'); self.assertEqual(row['compile_s'], 2e-6)
            self.assertIn('observed worker exit', row['compile_clock_scope'])
            self.assertFalse(row['compile_function_completed']); self.assertFalse(row['compile_accepted'])
            self.assertIsNone(row['boots_compiled'])

    def test_accepted_early_but_completion_persisted_late_rejects_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            root = Path(tmp); args, _ = self.fixture(root, model='alexnet')
            case = root/'resnet-accepted'; args.resnet_case = case
            (case/'source').mkdir(parents=True); (case/'work/compile').mkdir(parents=True)
            (case/'source/compiler-profile.json').write_text(args.profile.read_text())
            value = dict(round=d.ROUND, mode=d.MODE, model=d.MODELS['resnet20'], status='ok',
                compile_accepted=True, monitor_stopped=True, attempted={'compile': 1}, completed={'compile': 1},
                accepted_utc='2098-12-31T00:00:00+00:00',
                phases={'compile': dict(adapter=self.output(case/'work/compile'))},
                processes=[dict(phase='compile', observation=process())],
                source=dict(source_parameter_group='ours-N16-boot5'))
            marker = case/'completion.json'
            marker.write_text(json.dumps(dict(round=d.ROUND, status='ok', registry_persisted=True,
                required_artifacts_persisted=True)))
            (case/'result.json').write_text(json.dumps(value))
            row = dict(run_id=case.name, round=d.ROUND, model=d.MODELS['resnet20'], mode=d.MODE,
                git_rev='fixture', metrics=dict(status='ok'), artifacts_dir=str(case), ts='fixture')
            with (root/'registry.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            cutoff = int(d.RESNET_DEADLINE.timestamp())*10**9
            with patch.object(d.closed, 'producer_births', return_value=4):
                os.utime(marker, ns=(cutoff-1, cutoff-1))
                self.assertEqual(d.prerequisite_resnet(args)['case'], str(case))
                os.utime(marker, ns=(cutoff+1, cutoff+1))
                with self.assertRaisesRegex(ValueError, 'Complete ResNet persistence'):
                    d.prerequisite_resnet(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
