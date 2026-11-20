"""Small source, command, supervision and paper-record fixtures; no native run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import a5_a6_shared_rule63_compile as d
from pt_agent.experiments import cold_rule63_whole_model as cold
from pt_agent.experiments import dacapo_compile_continuation as resource
from pt_agent.experiments import a_group_runs


def observation(start=400, end=1100, **changes):
    row = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
               started_ns=start, ended_ns=end, peak_rss_kib_by_pid={})
    row.update(changes)
    return row


class SharedCompile(unittest.TestCase):
    def fixture(self, root, model='vgg16'):
        source = root/('a2b-'+model+'-rule63-fresh-cold-depth-20261117')
        (source/'source').mkdir(parents=True)
        (source/'work/emit').mkdir(parents=True)
        (source/'work/emit/candidate_trace.mlir').write_text('synthetic raw fixture only\n')
        profile = root/'profile.json'
        recipe = root/'recipe.json'
        profile.write_text(json.dumps(dict(polynomialDegree=65536, rescalingFactor=43,
            bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5, levelLowerBound=1,
            levelUpperBound=8, runtime='HEAAN-HEVM', latencyTable={})))
        recipe.write_text(json.dumps(['{hecate_opt}', '{input}', '--pars', '-o', '{output_dir}/lowered.mlir',
            '--compiler-config={package}/config/compiler-config.json', '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5', '--waterline=43',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64]))
        (source/'source/compiler-profile.json').write_text(profile.read_text())
        (source/'source/compile-metadata-template.json').write_text(recipe.read_text())
        args = d.arguments(['--model', model, '--source-case', str(source), '--out', str(root/'new-case'),
            '--paper-runs', str(root/'paper.jsonl'), '--profile', str(profile), '--metadata-template', str(recipe),
            '--diagnostic-elide-final-ir'])
        phases = {phase: dict(status='ok') for phase in cold.PHASES}
        phases['generation'].update(rule_options=dict(d.OPTIONS), raw_Count=dict(stats=dict(counts=dict(boot=0))),
            training_math=dict(samples=3, checks=[dict(passed=True) for _ in range(3)]))
        phases['compile']['native_process'] = observation(start=500, end=1000)
        result = dict(round=cold.ROUND, mode=cold.MODE, model=args.model, run_id=source.name,
            placement='depth', git_rev='fixture', status='ok', monitor_stopped=True,
            expected_bootstraps_match=True, guard_error=None, monitor_error=None,
            attempted=dict.fromkeys(cold.PHASES, 1), completed=dict.fromkeys(cold.PHASES, 1), phases=phases,
            processes=[dict(phase=p, observation=observation()) for p in cold.PHASES])
        completion = dict(round=cold.ROUND, status='ok', registry_persisted=True,
            required_artifacts_persisted=True, started_ns=100, ended_ns=1200)
        ledger = dict(run_id=source.name, round=cold.ROUND, model=args.model, mode=cold.MODE,
            git_rev='fixture', metrics=dict(status='ok'), artifacts_dir=str(source), ts='fixture')
        (source/'result.json').write_text(json.dumps(result))
        (source/'completion.json').write_text(json.dumps(completion))
        (root/'registry.jsonl').write_text(json.dumps(ledger)+'\n')
        return args, result

    def gate(self, args):
        with patch.object(cold, 'admission', return_value={'compiler': '/fixture/hecate-opt'}), \
             patch.object(d.closed_source, 'producer_births', return_value=7):
            return d.source_gate(args)

    def test_source_gate_binds_exact_closed_A2b_and_monotonic_prefix(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp))
            proof = self.gate(args)
            self.assertEqual(proof['shared_generation_prefix_ns'], 400)
            self.assertEqual(proof['closed_producer_births'], 7)
            self.assertEqual(proof['source_payload_reads'], 0)
            self.assertEqual(proof['clock_evidence']['source_native_compile_ended_ns'], 1000)

    def test_rejects_unclosed_wrong_source_and_duplicate_registry(self):
        for change in ('status', 'model', 'phase', 'registry', 'monitor'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                root = Path(tmp)
                args, source = self.fixture(root)
                if change == 'status': source['status'] = 'running'
                if change == 'model': source['model'] = 'alexnet_silu_avgpool'
                if change == 'phase': source['completed']['trace'] = 0
                if change == 'monitor': source['monitor_stopped'] = False
                (args.source_case/'result.json').write_text(json.dumps(source))
                if change == 'registry':
                    with (root/'registry.jsonl').open('a') as stream:
                        stream.write((root/'registry.jsonl').read_text())
                with self.assertRaises(ValueError): self.gate(args)

    def test_rejects_missing_raw_changed_profile_math_or_clock(self):
        for change in ('raw', 'profile', 'math', 'clock', 'contained'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(paths, 'RESULTS_DIR', Path(tmp)):
                args, source = self.fixture(Path(tmp))
                if change == 'raw': (args.source_case/'work/emit/candidate_trace.mlir').unlink()
                if change == 'profile':
                    (args.source_case/'source/compiler-profile.json').write_text('{}')
                if change == 'math': source['phases']['generation']['training_math']['checks'][1]['passed'] = False
                if change == 'clock': source['phases']['compile']['native_process']['started_ns'] = 50
                if change == 'contained': source['phases']['compile']['native_process']['started_ns'] = 300
                (args.source_case/'result.json').write_text(json.dumps(source))
                with self.assertRaises(ValueError): self.gate(args)

    def test_live_producer_failure_propagates_and_never_prepares_link(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp))
            with patch.object(cold, 'admission', return_value={}), \
                 patch.object(d.closed_source, 'producer_births', side_effect=ValueError('producer alive')):
                with self.assertRaisesRegex(ValueError, 'producer alive'): d.source_gate(args)
            self.assertFalse(args.out.exists())

    def test_prepare_uses_only_raw_symlink_and_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp))
            proof = self.gate(args)
            args.out.mkdir(); (args.out/'work').mkdir()
            raw = Path(proof['source_raw']); before = raw.stat()
            d.prepare_source(args, proof)
            link = args.out/'work/emit/candidate_trace.mlir'
            self.assertTrue(link.is_symlink()); self.assertEqual(link.resolve(), raw)
            after = raw.stat()
            self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns),
                             (after.st_ino, after.st_size, after.st_mtime_ns))
            self.assertFalse((link.parent/'constants.npz').exists())

    def test_command_has_one_pure_interpreter_and_only_compile_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            argv, seconds = d.phase_command(args, 'compile', 7500)
            inner = argv[argv.index('--')+1:]
            self.assertEqual(inner[:3], ['-B', '-m', d.__spec__.name])
            self.assertIn('--worker', inner)
            self.assertEqual(inner[inner.index('--native-seconds')+1], '7200')
            self.assertEqual(inner[inner.index('--model')+1], 'vgg16')
            self.assertNotIn('--source-case', inner)
            self.assertEqual(seconds, 7320)
            with self.assertRaises(ValueError): d.phase_command(args, 'trace', 7500)

    def fake_execute(self, root, *, failed=False, source_rejected=False):
        from pt_agent import registry
        args, source = self.fixture(root)
        if source_rejected:
            source['monitor_stopped'] = False
            (args.source_case/'result.json').write_text(json.dumps(source))
        calls = []
        def process(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(kwargs['mem_gib'], 24)
            self.assertEqual(kwargs['fsize_gib'], 16)
            self.assertEqual(kwargs['timeout'], 7320)
            worker = dict(status='error' if failed else 'ok', model=args.model, phase='compile',
                native_process=observation(2000, 3200, exit_code=1 if failed else 0),
                error='Synthetic native failure' if failed else None,
                compiled_bootstraps=None if failed else 17, execution_steps=None if failed else 123,
                execution_ops=None if failed else dict(boot=17, relinearize=5),
                prediction=None if failed else dict(table_known_sum_us=12, missing=[dict(op='relinearize')]),
                runtime_plan=None if failed else 'synthetic-runtime-plan-metadata')
            expected = d.read(args.out/'source/expected-native-argv.json')
            (args.out/'work/compile/native-argv.json').write_text(json.dumps(expected))
            (args.out/'work/compile/phase_result.json').write_text(json.dumps(worker))
            return observation(1900, 3300)
        with patch.object(paths, 'RESULTS_DIR', root), \
             patch.object(cold, 'admission', return_value={'compiler': '/fixture/hecate-opt'}), \
             patch.object(d.closed_source, 'producer_births', return_value=7), \
             patch.object(cold, 'run_process', side_effect=process), \
             patch.object(registry, 'git_rev', return_value='fixture'), \
             patch.object(resource, 'available_ram', return_value=20*cold.GIB), \
             patch.object(cold.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 100*cold.GIB})()):
            result = d.execute(args)
        return args, result, calls

    def test_real_private_execute_success_persists_compile_only_and_derived_row(self):
        previous = (cold.ROUND, cold.MODE, cold.PHASES, cold.execute.__code__)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args, result, calls = self.fake_execute(root)
            self.assertEqual(result['status'], 'ok'); self.assertEqual(len(calls), 1)
            self.assertEqual(result['completed'], {'compile': 1}); self.assertFalse(result['cold'])
            row = d.read(args.out/'paper-record.json')
            self.assertEqual(row['compile_s'], .0000012)
            self.assertAlmostEqual(row['generation_e2e_s'], .0000016)
            self.assertEqual(row['pred_latency_s'], .000012)
            self.assertTrue(row['estimated_from_shared_generation']); self.assertFalse(row['cold'])
            self.assertEqual(row['boots_compiled'], 17)
            self.assertEqual(row['actual_compile_only_case_seconds'], d.read(args.out/'completion.json')['complete_case_elapsed_seconds'])
            self.assertEqual(len((root/'paper.jsonl').read_text().splitlines()), 1)
            rows = [json.loads(line) for line in (root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2); self.assertTrue(all(len(r) == 8 for r in rows))
            self.assertEqual(rows[-1]['round'], d.ROUND)
        self.assertEqual(previous, (cold.ROUND, cold.MODE, cold.PHASES, cold.execute.__code__))

    def test_native_failure_still_appends_actual_elapsed_and_terminal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, result, calls = self.fake_execute(Path(tmp), failed=True)
            row = d.read(args.out/'paper-record.json')
            self.assertEqual(result['status'], 'error'); self.assertEqual(len(calls), 1)
            self.assertEqual(row['status'], 'compiler_crash'); self.assertEqual(row['compile_s'], .0000012)
            self.assertFalse(row['compile_accepted']); self.assertIsNone(row['boots_compiled'])
            self.assertTrue((args.source_case/'work/emit/candidate_trace.mlir').is_file())
            self.assertEqual(d.read(args.out/'completion.json')['status'], 'error')

    def test_source_rejection_has_zero_workers_and_no_derived_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, result, calls = self.fake_execute(Path(tmp), source_rejected=True)
            row = d.read(args.out/'paper-record.json')
            self.assertEqual(result['status'], 'error'); self.assertEqual(calls, [])
            self.assertEqual(result['attempted'], {}); self.assertIsNone(row['generation_e2e_s'])
            self.assertIsNone(row['compile_s']); self.assertFalse(row['prediction_policy']['available'])

    def test_shared_price_policy_omits_only_relinearize(self):
        policy = d.profile_policy(dict(table_known_sum_us=22, missing=[dict(op='relinearize'), dict(op='new_op')]))
        self.assertIsNone(policy['pred_latency_s'])
        self.assertEqual(policy['known_cost_partial_s'], .000022)
        self.assertEqual(policy['excluded_counts'], dict(relinearize=1))
        self.assertEqual(policy['missing_remaining'], [dict(op='new_op')])

    def test_terminal_stat_and_original_native_argv_gate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp)); proof = self.gate(args)
            args.out.mkdir(); (args.out/'work').mkdir(); d.prepare_source(args, proof)
            work = args.out/'work/compile'; work.mkdir()
            command = d.expected_native_argv(args, proof)
            (work/'native-argv.json').write_text(json.dumps(command))
            result = dict(status='ok', error=None)
            d.terminal_source_observation(args, result, proof)
            self.assertEqual(result['status'], 'ok')
            self.assertTrue(result['source_raw_stat_preserved']); self.assertTrue(result['native_argv_verified'])
            self.assertEqual(Path(proof['source_raw']).stat().st_nlink, proof['source_raw_stat']['links'])
            Path(proof['source_raw']).write_text('changed tiny fixture')
            d.terminal_source_observation(args, result, proof)
            self.assertEqual(result['status'], 'error'); self.assertFalse(result['source_raw_stat_preserved'])

    def test_terminal_rejects_old_output_or_missing_actual_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'RESULTS_DIR', Path(tmp)):
            args, _ = self.fixture(Path(tmp)); proof = self.gate(args)
            args.out.mkdir(); (args.out/'work').mkdir(); d.prepare_source(args, proof)
            work = args.out/'work/compile'; work.mkdir()
            result = dict(status='ok', error=None)
            d.terminal_source_observation(args, result, proof)
            self.assertEqual(result['status'], 'error')
            command = d.expected_native_argv(args, proof)
            command[command.index('-o')+1] = str(args.source_case/'work/emit/lowered.mlir')
            (work/'native-argv.json').write_text(json.dumps(command))
            result = dict(status='ok', error=None)
            d.terminal_source_observation(args, result, proof)
            self.assertEqual(result['status'], 'error'); self.assertFalse(result['native_argv_verified'])
            self.assertTrue(result['source_raw_stat_preserved'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
