"""New tiny metadata/fake-native fixtures; no original Qwen or compiler run."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent import paths, registry, toolchain
from pt_agent.experiments import qwen_rule63_shared_optimized_compile as own
from pt_agent.experiments import cold_rule63_whole_model as supervisor
from pt_agent.experiments import dacapo_compile_continuation as resources


def observation(start=100, end=1000, **changes):
    value = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
                 started_ns=start, ended_ns=end)
    value.update(changes)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SharedQwenCompile(unittest.TestCase):
    def fixture(self, root):
        profile = root / 'compiler-config.json'
        template = root / 'compile_argv.json'
        binary = root / 'existing-hecate-opt'; binary.write_text('synthetic fixture only')
        write(profile, dict(polynomialDegree=65536, rescalingFactor=43,
            bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=5, levelLowerBound=1,
            levelUpperBound=8, runtime='HEAAN-HEVM', latencyTable={}))
        write(template, ['{hecate_opt}', '{input}', '--pars', '-o', '{output_dir}/lowered.mlir',
            '--compiler-config={package}/config/compiler-config.json', '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5', '--waterline=43',
            '--runtime-plan-operator-spec-sha256=sha256:' + '0' * 64])
        source = root / own.SOURCE_ID
        emit = source / 'work/emit'; emit.mkdir(parents=True)
        (emit / 'candidate_trace.mlir').write_text('synthetic original raw fixture only\n')
        write(emit / 'lowered.synthetic.runtime-plan.json', {'synthetic_metadata_fixture': True})
        write(source / 'source' / profile.name, own.read(profile))
        write(source / 'source' / template.name, own.read(template))
        phases = {p: dict(status='ok') for p in own.depth.PHASES}
        phases['prepare'].update(training_math=dict(passed=True, sample_id='train:0', sequence_capacity=32,
            criterion='1e-12+1e-12*abs(reference)'), activation_degree_changes=[dict(fn='silu',
            original_degree=127, degree=63)], program_ast='synthetic unchanged AST',
            raw_count=dict(counts=dict(boot=0, mul_cc=3), ops_total=7, distinct_consts=2))
        phases['emit']['emit_stats'] = dict(phases['prepare']['raw_count'])
        phases['trace']['native_process'] = observation()
        phases['compile'].update(native_process=observation(), compiled_bootstraps=12,
            runtime_plan=str(emit / 'lowered.synthetic.runtime-plan.json'))
        source_result = dict(status='ok', round=own.depth.ROUND, model=own.MODEL, run_id=source.name,
            monitor_stopped=True, guard_error=None, monitor_error=None, phases=phases,
            source=dict(C2_required=False, Agent_source_used=False, bridge_policy_used=False),
            processes=[dict(phase=p, observation=observation()) for p in own.depth.PHASES])
        write(source / 'result.json', source_result)
        write(source / 'completion.json', dict(round=own.depth.ROUND, status='ok',
            registry_persisted=True, required_artifacts_persisted=True))
        record = dict(run_id=source.name, round=own.depth.ROUND, model=own.MODEL, mode=own.SOURCE_MODE,
            git_rev='fixture', metrics=dict(status='ok'), artifacts_dir=str(source), ts='fixture')
        (root / 'registry.jsonl').write_text(json.dumps(record) + '\n')
        args = own.arguments(['--source-case', str(source), '--out', str(root / 'new-optimized'),
            '--paper-runs', str(root / 'paper.jsonl'), '--host-pid-namespace', os.readlink('/proc/self/ns/pid'),
            '--profile', str(profile), '--metadata-template', str(template), '--diagnostic-elide-final-ir'])
        return args, source_result, binary

    def test_closed_source_exact_math_count_profile_birth_and_raw_contract(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(paths, 'RESULTS_DIR', Path(temporary)):
            root = Path(temporary); args, source, binary = self.fixture(root)
            with patch.object(toolchain, 'host_paths', return_value={'hecate_opt': str(binary)}), \
                    patch.object(own.shared.closed_source, 'producer_births', return_value=7):
                proof = own.source_gate(args)
                self.assertEqual(proof['closed_producer_births'], 7)
                self.assertEqual(proof['source_payload_reads'], 0)
                self.assertFalse(proof['C2_required'])
                source['phases']['prepare']['training_math']['passed'] = False
                write(args.source_case / 'result.json', source)
                with self.assertRaisesRegex(ValueError, 'mathematics'): own.source_gate(args)
                source['phases']['prepare']['training_math']['passed'] = True
                source['phases']['emit']['emit_stats'] = dict(counts=dict(boot=1), ops_total=7, distinct_consts=2)
                write(args.source_case / 'result.json', source)
                with self.assertRaisesRegex(ValueError, 'Count/Emit'): own.source_gate(args)

    def test_pure_cli_has_one_interpreter_native7200_and_no_model_or_generation_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = self.fixture(Path(temporary))
            command, seconds = own.phase_command(args, 'compile', 7500)
            self.assertEqual(command.count('--python'), 1)
            self.assertEqual(command.count(own.MODULE), 1)
            self.assertEqual(seconds, 7320)
            self.assertEqual(command[command.index('--native-seconds') + 1], '7200')
            self.assertNotIn('--source-case', command)
            self.assertNotIn('--model-info', command)
            self.assertNotIn('--calibration', command)
            with self.assertRaises(ValueError): own.phase_command(args, 'trace', 7500)

    def test_fake_native_or_rejected_source_closes_one_case_without_generation_e2e(self):
        before = (supervisor.ROUND, supervisor.MODE, supervisor.PHASES, supervisor.execute.__code__)
        for reject in (False, True):
            with self.subTest(reject=reject), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(paths, 'RESULTS_DIR', Path(temporary)):
                root = Path(temporary); args, source, binary = self.fixture(root)
                if reject:
                    source['status'] = 'error'; write(args.source_case / 'result.json', source)
                calls = []
                def native(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(kwargs['mem_gib'], 24)
                    self.assertEqual(kwargs['fsize_gib'], 16)
                    value = dict(status='ok', error=None, native_process=observation(1500, 2500),
                        compiled_bootstraps=9, execution_steps=77, runtime_plan='synthetic-compiled-plan',
                        execution_ops=dict(boot=9, relinearize=4),
                        prediction=dict(table_known_sum_us=1_250_000, missing=[dict(op='relinearize')]))
                    write(args.out / 'work/compile/phase_result.json', value)
                    write(args.out / 'work/compile/native-argv.json',
                          own.read(args.out / 'source/expected-native-argv.json'))
                    return observation(1400, 2600)
                with patch.object(toolchain, 'host_paths', return_value={'hecate_opt': str(binary)}), \
                        patch.object(own.shared.closed_source, 'producer_births', return_value=7), \
                        patch.object(supervisor, 'run_process', side_effect=native), \
                        patch.object(registry, 'git_rev', return_value='fixture'), \
                        patch.object(resources, 'available_ram', return_value=20 * supervisor.GIB):
                    result = own.execute(args)
                self.assertEqual(len(calls), 0 if reject else 1)
                self.assertEqual(result['status'], 'error' if reject else 'ok')
                self.assertEqual(result['attempted'], {} if reject else {'compile': 1})
                self.assertEqual(own.read(args.out / 'completion.json')['status'], result['status'])
                row = own.read(args.out / 'paper-record.json')
                self.assertEqual(row['status'], 'driver_error' if reject else 'ok')
                self.assertIsNone(row['generation_e2e_s'])
                self.assertFalse(row['estimated_from_shared_generation'])
                self.assertFalse(row['full_model']); self.assertFalse(row['cold'])
                self.assertEqual(row['pred_latency_s'], None if reject else 1.25)
                self.assertAlmostEqual(sum(v for v in row['stages_s'].values() if v is not None),
                                       row['actual_compile_only_case_seconds'])
                self.assertEqual(len((root / 'registry.jsonl').read_text().splitlines()), 2)
                self.assertTrue(all(len(json.loads(line)) == 8
                                    for line in (root / 'registry.jsonl').read_text().splitlines()))
                self.assertEqual(len((root / 'paper.jsonl').read_text().splitlines()), 1)
                if not reject:
                    self.assertTrue(result['source_raw_stat_preserved']); self.assertTrue(result['native_argv_verified'])
                    self.assertTrue((args.out / 'work/emit/candidate_trace.mlir').is_symlink())
                    self.assertTrue((args.source_case / 'work/emit/candidate_trace.mlir').is_file())
                    self.assertFalse((args.out / 'work/emit/constants.npz').exists())
        self.assertEqual(before, (supervisor.ROUND, supervisor.MODE, supervisor.PHASES, supervisor.execute.__code__))


if __name__ == '__main__':
    unittest.main()
