"""Small synthetic checks for the original-source compile orchestration."""

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_original_compile as subject


def fixtures():
    compiler = dict(runtime='HEAAN-HEVM', polynomialDegree=131072, rescalingFactor=51,
                    levelLowerBound=1, levelUpperBound=29,
                    bootstrapLevelLowerBound=3, bootstrapLevelUpperBound=16,
                    latencyTable={'earth.bootstrap_single': [0, 1]})
    profile = dict(spec_format_version=2, spec_id='dacapo-heaan-cpu-v1', target_id='dacapo-heaan-cpu',
        context=dict(poly_degree=131072, rns_moduli_log2=[51] * 30, default_scale_log2=40),
        levels=dict(lower_bound=1, upper_bound=29), boot_profiles=[dict(
            profile_id='dacapo-heaan-cpu-boot-v1', implementation='native', input_level_min=3,
            input_level_max=16, input_components=2, output_level=16, output_scale_log2=40,
            output_components=2, latency_us_by_input_level=[0] * 30, noise_by_input_level=None,
            host_requirements=dict(needs_secret_key=False, needs_host_compute=False))])
    return compiler, profile


def plan_fixture():
    return dict(format_version=1, values=[
        dict(id='0', context=subject.CONTEXT, level=3, scale_log2=48, components=2),
        dict(id='1', context=subject.CONTEXT, level=16, scale_log2=51, components=2)],
        execution=[dict(kind='compute', op='boot', inputs=['0'], output='1',
            attrs=dict(operator_profile=subject.BOOT_ID, implementation='native',
                       target_level=16, target_scale_log2=51, target_components=2))])


class OriginalCompileChecks(unittest.TestCase):
    def test_export_schema_keeps_original_profile_and_separates_scales(self):
        compiler, original = fixtures()
        before = copy.deepcopy(original)
        contract, notes = subject.make_export_contract(original, compiler)
        self.assertEqual(original, before)
        self.assertEqual(contract['context']['default_scale_log2'], 40)
        self.assertEqual(contract['boot_profiles'][0]['output_scale_log2'], 51)
        self.assertEqual(notes['compiler_rescaling_factor_log2'], 51)
        self.assertIs(notes['concrete_primes_verified'], False)
        self.assertIs(notes['native_lattigo_binding_pending'], True)
        self.assertEqual(notes['source_lowering_initial_remaining_level'], 16)

    def test_changed_native_profile_rejected_before_use(self):
        compiler, imported = fixtures()
        compiler['polynomialDegree'] = 65536
        with self.assertRaisesRegex(ValueError, 'original HEAAN'):
            subject.make_export_contract(imported, compiler)
        compiler, imported = fixtures()
        imported['boot_profiles'][0]['implementation'] = 'decrypt_reencrypt'
        with self.assertRaisesRegex(ValueError, 'native Boot profile'):
            subject.make_export_contract(imported, compiler)

    def test_command_preserves_existing_marker_without_legacy_builders(self):
        marker = '--runtime-plan-operator-spec-sha256=sha256:' + 'a' * 64
        template = ['unused', '--runtime-plan-format=json', '--runtime-plan-bundle-format=files',
                    marker, '--pars', '--enable-debug-printer']
        command = subject.compiler_argv(template, binary='/bin/compiler', source='/work/input.mlir',
            profile='/work/profile.json', output='/work/lowered.mlir')
        self.assertEqual(command[:2], ['/bin/compiler', '/work/input.mlir'])
        self.assertIn(marker, command)
        self.assertIn('--dacapo', command)
        self.assertIn('--waterline=40', command)
        self.assertIn('--runtime-plan-boot-profile=' + subject.BOOT_ID, command)
        self.assertNotIn('--pars', command)
        self.assertNotIn('--enable-debug-printer', command)
        with self.assertRaisesRegex(ValueError, 'one existing'):
            subject.compiler_argv(template + [marker], binary='c', source='i', profile='p', output='o')

    def test_saved_original_model_paths_match_without_loading_checkpoint(self):
        upstream = Path('/original/dacapo')
        checkpoint = upstream / 'examples/data/resnet20.silu.model'
        factory = 'pt_agent.experiments.r13_common_models:resnet20_silu'
        record = dict(round='R13', status='ok', model=factory, worker=dict(model_source=dict(
            factory=factory, checkpoint=str(checkpoint),
            original_source=str(upstream / 'python/poly/poly/models/ResNet.py'),
            training_preprocessing_verified=False)))
        observed = subject.validate_source_record(record, upstream=upstream, checkpoint=checkpoint)
        self.assertEqual(observed['factory'], factory)
        self.assertFalse(observed['training_preprocessing_verified'])
        record['worker']['model_source']['checkpoint'] = '/different/weights'
        with self.assertRaisesRegex(ValueError, 'Saved original ModelInfo'):
            subject.validate_source_record(record, upstream=upstream, checkpoint=checkpoint)

    def test_format1_single_output_boot_observation(self):
        compiler, imported = fixtures()
        contract, _ = subject.make_export_contract(imported, compiler)
        observed = subject.observe_plan(plan_fixture(), contract)
        self.assertEqual(observed['op_counts'], {'boot': 1})
        self.assertEqual(observed['bootstrap_positions'][0]['output'], '1')
        self.assertEqual(observed['bootstrap_positions'][0]['input_scale_log2'], 48)
        self.assertTrue(observed['export_contract_compatible'])
        self.assertFalse(observed['concrete_prime_binding_verified'])
        self.assertEqual(observed['value_levels'], [3, 16])
        self.assertTrue(observed['all_value_levels_within_r27_max16'])

    def test_export_mismatch_retains_actual_boot_count(self):
        compiler, imported = fixtures()
        contract, _ = subject.make_export_contract(imported, compiler)
        plan = plan_fixture()
        plan['values'][1]['scale_log2'] = 40
        plan['execution'][0]['attrs']['target_scale_log2'] = 40
        observed = subject.observe_plan(plan, contract)
        self.assertEqual(observed['bootstrap_count'], 1)
        self.assertEqual(observed['contract_mismatch_ordinals'], [0])
        self.assertFalse(observed['export_contract_compatible'])

    def test_non_compute_execution_instructions_remain_in_observation(self):
        compiler, imported = fixtures()
        contract, _ = subject.make_export_contract(imported, compiler)
        plan = plan_fixture()
        plan['execution'].insert(0, dict(kind='encode', output='2'))
        observed = subject.observe_plan(plan, contract)
        self.assertEqual(observed['op_counts'], {'encode': 1, 'boot': 1})
        self.assertEqual(observed['bootstrap_positions'][0]['instruction_ordinal'], 1)
        self.assertTrue(observed['export_contract_compatible'])

    def test_both_nix_phases_have_independent_native_wall_limits(self):
        args = SimpleNamespace(trace_seconds=1800, compile_seconds=7200,
            compiler=Path('/bin/compiler'), memory_gib=12, intake_python=Path('/venv/python'),
            source=Path('/original/ResNet.py'), frontend=Path('/frontend'),
            poly_root=Path('/poly'), einops=Path('/einops'))
        trace = subject._nix_worker(args, 'trace', Path('/work/trace'))
        compile_ = subject._nix_worker(args, 'compile', Path('/work/compile'), command_file=Path('/work/argv.json'))
        self.assertEqual(trace[trace.index('--seconds') + 1], '1860')
        self.assertEqual(compile_[compile_.index('--seconds') + 1], '7260')
        self.assertEqual(compile_[compile_.index('--worker-seconds') + 1], '7200')
        self.assertIn('--worker-compile', compile_)
        self.assertIn('/work/argv.json', compile_)

    def test_budget_signal_between_children_unwinds_to_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = signal.getsignal(signal.SIGTERM)
            with patch.object(subject.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 40)):
                with self.assertRaisesRegex(RuntimeError, 'Case supervisor'):
                    with subject._budget_guard(Path(directory), case_gib=1, minimum_free_gib=10):
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_preflight_failure_saves_result_unique_registry_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            compiler, imported = fixtures()
            compiler['rescalingFactor'] = 43
            (base / 'compiler.json').write_text(json.dumps(compiler))
            (base / 'imported.json').write_text(json.dumps(imported))
            args = SimpleNamespace(out=base / 'case', compiler_profile=base / 'compiler.json',
                imported_spec=base / 'imported.json', waterline=40, registry=base / 'registry.jsonl', case_gib=1)
            with patch.object(subject, 'run_process', side_effect=AssertionError('No child allowed')):
                self.assertEqual(subject.run(args), 1)
            result = json.loads((args.out / 'result.json').read_text())
            rows = [json.loads(line) for line in args.registry.read_text().splitlines()]
            endpoint = json.loads((args.out / 'completion.json').read_text())
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['compile_attempts'], 0)
            self.assertIsNone(result['actual_ckks_latency_seconds'])
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            self.assertTrue(endpoint['registry_persisted'])
            self.assertGreaterEqual(endpoint['ended_ns'], endpoint['started_ns'])

    def test_compile_worker_retains_native_failure_without_attempting_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            command = [str(Path('/bin/compiler').resolve()), '--dacapo', '--boot-placement=optimized',
                       '--runtime-plan-boot-profile=' + subject.BOOT_ID]
            (base / 'argv.json').write_text(json.dumps(command))
            args = SimpleNamespace(worker_dir=base / 'worker', command_file=base / 'argv.json',
                                   compiler=Path('/bin/compiler'), worker_seconds=7200, memory_gib=12)
            obs = dict(exit_code=-11, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
            with patch.dict(os.environ, {'HECATE_PYTHON_LIBRARY_PATH': '/local/lib'}):
                with patch.object(subject, 'run_process', return_value=obs) as run:
                    self.assertEqual(subject.compile_worker(args), 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['timeout'], 7200)
            result = json.loads((args.worker_dir / 'worker_result.json').read_text())
            self.assertEqual(result['compile_attempts'], 1)
            self.assertEqual(result['compile_completed'], 0)
            self.assertEqual(result['process']['exit_code'], -11)

    def test_runpy_source_and_original_save_filename_preserved(self):
        from pt_agent.models import dacapo_builders
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'upstream/ResNet.py'
            source.parent.mkdir()
            source.write_text('# trusted fixture only\n')
            args = SimpleNamespace(worker_dir=base / 'work', source=source,
                frontend=base / 'frontend', compiler=base / 'build/bin/compiler',
                poly_root=base / 'poly', einops=base / 'einops', worker_seconds=3)
            seen, finalized = [], []
            module = SimpleNamespace(ctxt=1, lt=SimpleNamespace(finalize=lambda value: finalized.append(value)))
            def trusted_run(path, run_name):
                seen.append((path, run_name, list(sys.argv), Path.cwd()))
                Path('traced/ResNet.mlir').write_text('module {}')
                sys.modules['hecate'] = module
            with patch.object(dacapo_builders, '_torchvision_imports', return_value=nullcontext()):
                with patch.object(subject.runpy, 'run_path', side_effect=trusted_run):
                    with patch.dict(sys.modules):
                        self.assertEqual(subject.trace_worker(args), 0)
            self.assertEqual(seen[0][:3], (str(source), '__main__', [str(source)]))
            self.assertEqual(seen[0][3], args.worker_dir)
            self.assertEqual(finalized, [1])
            result = json.loads((args.worker_dir / 'worker_result.json').read_text())
            self.assertEqual(result['mlir']['path'], str(args.worker_dir / 'traced/ResNet.mlir'))
            self.assertEqual(result['raw_cifar_reads'], 0)


if __name__ == '__main__':
    unittest.main()
