"""Synthetic build protocol fixtures: never invoke Go or a model."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_backend_compile_only as entry


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class OrbitBackendCompileOnlyChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self):
        case = self.root/'optimizer'; case.mkdir()
        mlir = case/'work/optimized.mlir'; mlir.parent.mkdir(); mlir.write_text('synthetic metadata fixture\n')
        constant = self.root/'original.cst'; constant.write_text('synthetic unused constant payload')
        frontend = self.root/'frontend/result.json'
        save(frontend, dict(round='R22', status='ok', model=entry.MODEL, phase_results=dict(trace=dict(
            constant_files=[dict(path=str(constant), size_bytes=constant.stat().st_size)]))))
        params = dict(poly_deg=131072, max_slot=65536, Sf=51, Sw=40, Csw=40, lvl_ub=16,
                      bts_lb=3, bts_ub=16, dacapo_mlir_in=True, dacapo_mlir_out=True)
        worker = dict(model=entry.MODEL, status='ok', source_checked=True, environment_restored=True,
                      output_mlir=dict(path=str(mlir), size_bytes=mlir.stat().st_size), params=params)
        observed = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        record = dict(round='R23', model=entry.MODEL, status='ok', native_optimizer_completed=True,
                      placement_artifact_checked=True, placement_bootstrap_nodes=19, placement_bootstrap_weight_sum=19,
                      attempted=dict(optimize=1, validate=1), completed=dict(optimize=1, validate=1),
                      source_records=dict(r22_result=str(frontend)), phase_results=dict(optimize=worker,
                      validate=dict(worker, placement_artifact_checked=True, check_tdag_calls=1, check_tdag_completed=1)),
                      processes=[dict(stage=name, observation=deepcopy(observed)) for name in ('model-optimize', 'model-validate')])
        save(case/'result.json', record)
        orbit = self.root/'orbit'; package = orbit/'backend/lattigo/lowering/fhe'; package.mkdir(parents=True)
        for name in ('main.go', 'parser.go', 'scheme.go', 'evaluator.go'):
            (package/name).write_text('package main\n')
        for name in ('go.mod', 'go.sum'):
            (orbit/'backend/lattigo'/name).write_text('synthetic module metadata\n')
        sdk = self.root/'sdk'; (sdk/'bin').mkdir(parents=True)
        (sdk/'bin/go').write_text('never executed'); (sdk/'VERSION').write_text('synthetic-go-version\n')
        cache = self.root/'cache'; cache.mkdir(); (cache/'metadata.txt').write_text('synthetic offline cache\n')
        return Namespace(source_case=case, out=self.root/'build', orbit_root=orbit,
                         go_sdk=sdk, module_cache=cache, seconds=1200, case_gib=8), record

    def test_source_gate_preserves_native_profile_and_placement_scope(self):
        args, _ = self.fixture(); gate = entry.source_gate(args.source_case)
        self.assertEqual(gate['optimizer_params']['Sw'], 40)
        self.assertEqual(gate['optimizer_params']['Sf'], 51)
        self.assertEqual(gate['placement_bootstrap_nodes'], 19)
        self.assertFalse(gate['prior_optimizer_times_included']); self.assertFalse(gate['model_payload_or_inputs_read'])

    def test_source_drift_dirty_process_and_incomplete_validation_rejected(self):
        args, original = self.fixture()
        for changed in ('path', 'scale', 'remaining', 'validation'):
            record = deepcopy(original)
            if changed == 'path': record['phase_results']['validate']['output_mlir']['path'] = str(self.root/'wrong.mlir')
            elif changed == 'scale': record['phase_results']['validate']['params']['Sf'] = 43
            elif changed == 'remaining': record['processes'][1]['observation']['remaining_pids'] = [123]
            else: record['phase_results']['validate']['check_tdag_completed'] = 0
            save(args.source_case/'result.json', record)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                entry.source_gate(args.source_case)

    def test_original_build_command_has_no_execution_or_model_argument(self):
        command = entry.build_command(Path('/synthetic/sdk'), Path('/synthetic/out/orbit-fhe'))
        self.assertEqual(command[1], 'build'); self.assertEqual(command[-1], './lowering/fhe')
        self.assertIn('-mod=readonly', command); self.assertIn('-buildvcs=false', command)
        self.assertNotIn('run', command); self.assertFalse(any(x.startswith(('-mlir', '-getLog', '-heMode')) for x in command))

    def test_native_environment_isolated_and_parent_unchanged(self):
        original = dict(HOME='/original-home', KEEP='safe', GOROOT='old', GOPROXY='network',
                        GOFLAGS='old-flags', LD_PRELOAD='old', NIX_PATH='old', HECATE='old')
        snapshot = deepcopy(original); env = entry.native_environment(Path('/new/work'), Path('/new/sdk'), original)
        self.assertEqual(original, snapshot); self.assertEqual(env['HOME'], '/original-home')
        self.assertEqual(env['KEEP'], 'safe'); self.assertEqual(env['GOROOT'], '/new/sdk')
        self.assertEqual(env['GOPROXY'], 'off'); self.assertNotIn('GOFLAGS', env)
        self.assertNotIn('LD_PRELOAD', env); self.assertNotIn('NIX_PATH', env); self.assertNotIn('HECATE', env)
        self.assertEqual(env['TEST_TELEMETRY_DIR'], '/new/work/telemetry'); self.assertEqual(env['CGO_ENABLED'], '0')

    def test_source_preparation_does_not_copy_or_open_constants(self):
        args, _ = self.fixture(); gate = entry.source_gate(args.source_case); args.out.mkdir()
        work, manifest = entry.prepare_sources(args, args.out, gate)
        self.assertEqual((work/'telemetry/mode').read_text(), 'off\n')
        self.assertFalse(manifest['constant_payload_copied']); self.assertTrue(manifest['original_constant_file_required'])
        self.assertFalse(manifest['model_specific_native_compiler_available'])
        self.assertFalse(manifest['runtime_execution_scheduled'])
        self.assertFalse(list(args.out.rglob('*.cst')))
        self.assertTrue((args.out/'model/optimized.mlir').is_file())
        self.assertEqual(manifest['constants'], gate['constants'])

    def fake_execute(self, *, exit_code=0, raising=False):
        args, _ = self.fixture(); calls = []
        class NoThread:
            def __init__(self, **kwargs): pass
            def start(self): pass
            def is_alive(self): return False
        def run(command, **kwargs):
            calls.append((command, kwargs))
            self.assertEqual((args.out/'work/telemetry/mode').read_text(), 'off\n')
            if raising: raise RuntimeError('synthetic supervisor failure')
            if exit_code == 0: Path(command[command.index('-o')+1]).write_text('synthetic non-executed binary')
            return dict(exit_code=exit_code, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
                        started_ns=1, ended_ns=2)
        with patch.object(entry.paths, 'RESULTS_DIR', self.root), patch('pt_agent.registry.git_rev', return_value='fixture'), \
                patch.object(entry.threading, 'Thread', NoThread), patch.object(entry, 'run_process', side_effect=run), \
                patch.object(entry, '_resource_observation', return_value=dict(stop=False, reasons=[])):
            result = entry.execute(args)
        return args, result, calls

    def test_one_build_with_complete_registry_and_honest_model_fields(self):
        args, result, calls = self.fake_execute()
        self.assertEqual(result['status'], 'ok'); self.assertEqual(len(calls), 1)
        self.assertTrue(result['runtime_interpreter_compiled']); self.assertIsNone(result['model_backend_compiled'])
        self.assertIsNone(result['compiled_bootstraps']); self.assertIsNone(result['model_backend_compile_seconds'])
        self.assertEqual(result['source']['placement_bootstrap_nodes'], 19)
        self.assertEqual(calls[0][1]['cwd'], args.orbit_root/'backend/lattigo')
        self.assertEqual(calls[0][1]['mem_gib'], 12); self.assertEqual(calls[0][1]['timeout'], 900)
        registry = [json.loads(x) for x in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(registry), 1); self.assertEqual(len(registry[0]), 8)
        completed = entry.read(args.out/'completion.json')
        self.assertTrue(completed['registry_persisted']); self.assertTrue(completed['required_artifacts_persisted'])
        self.assertEqual(completed['case_artifact_bytes'], entry.artifact_bytes(args.out))
        self.assertTrue(all(result[key] == 0 for key in entry.ZERO_FIELDS))

    def test_build_failure_stops_and_keeps_terminal_case(self):
        args, result, calls = self.fake_execute(exit_code=1)
        self.assertEqual(len(calls), 1); self.assertEqual(result['status'], 'error')
        self.assertEqual(result['go_build_attempts'], 1); self.assertEqual(result['go_build_completed'], 0)
        self.assertFalse(result['runtime_interpreter_compiled']); self.assertFalse(result['cleanup_unknown'])
        self.assertEqual(entry.read(args.out/'completion.json')['status'], 'error')

    def test_supervisor_exception_keeps_cleanup_unknown_and_never_retries(self):
        args, result, calls = self.fake_execute(raising=True)
        self.assertEqual(len(calls), 1); self.assertTrue(result['cleanup_unknown'])
        self.assertIsNone(result['process']); self.assertEqual(result['status'], 'error')
        self.assertTrue((args.out/'completion.json').is_file())

    def test_guard_signal_preserves_its_reason_and_external_cancellation_is_separate(self):
        for key, reason in (('resource_guard', dict(stop=True, reasons=['host reserve'])),
                            ('resource_monitor_error', 'synthetic missing resource metadata')):
            record = dict(incoming_signal=None, guard_stop_signal=None, **{key: reason})
            with self.subTest(key=key), self.assertRaises(InterruptedError):
                entry.receive_signal(record, signal.SIGUSR1)
            save(self.root/'signal-result.json', record); saved = entry.read(self.root/'signal-result.json')
            self.assertIsNone(saved['incoming_signal']); self.assertEqual(saved['guard_stop_signal'], signal.SIGUSR1)
            self.assertEqual(saved[key], reason)
        record = dict(incoming_signal=None, guard_stop_signal=None)
        with self.assertRaises(InterruptedError):
            entry.receive_signal(record, signal.SIGTERM)
        self.assertEqual(record['incoming_signal'], signal.SIGTERM); self.assertIsNone(record['guard_stop_signal'])

    def test_resource_observation_keeps_physical_and_virtual_limits_distinct(self):
        args, _ = self.fixture()
        with patch('pt_agent.experiments.r27_orbit_lazy_backend._latest_rss_sample', return_value=None), \
                patch('pt_agent.experiments.r27_orbit_lazy_backend._available_ram_bytes', return_value=entry.GIB), \
                patch.object(entry, 'artifact_bytes', return_value=0), \
                patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)):
            observation = entry._resource_observation(args, self.root, self.root/'samples', __import__('time').perf_counter_ns())
        self.assertTrue(observation['stop']); self.assertIsNone(observation['subtree_rss_bytes'])
        self.assertEqual(observation['rss_allowance_bytes'], 4*entry.GIB)
        self.assertEqual(observation['minimum_available_bytes'], int(1.25*entry.GIB))

    def test_cli_is_finite_and_requires_explicit_existing_cache(self):
        with self.assertRaises(SystemExit):
            entry.parser().parse_args(['--source-case', 'source', '--out', 'output'])
        for seconds in (0, 1201):
            with self.subTest(seconds=seconds), self.assertRaises(SystemExit), patch.object(entry, 'execute') as execute:
                entry.main(['--source-case', 'source', '--out', 'output', '--module-cache', 'cache', '--seconds', str(seconds)])
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
