"""Small saved-metadata/fake-process checks; no HEIR/Go/model is invoked."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_common_go_compile as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def good_process():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[], elapsed_seconds=.001)


class CommonGoCompileCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.case = self.root/'source-case';self.cache = self.root/'backend'
        self.tool = self.root/'tools';self.tool.mkdir();self.translate = self.tool/'heir-translate'
        self.translate.write_text('existing executable');self.translate.chmod(0o700)
        (self.tool/'heir-opt').write_text('existing executable')
        self.go = self.cache/'toolchain/go/bin/go';put(self.go, 'existing sdk executable');self.go.chmod(0o700)
        put(self.cache/'toolchain/go/VERSION', 'go1.26.0\ntime irrelevant\n')
        put(self.cache/'module/go.mod', 'module old\n\ngo 1.26.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.1.0\n')
        put(self.cache/'module/go.sum', 'existing opaque native metadata\n')
        (self.cache/'module-cache/github.com/tuneinsight/lattigo/v6@v6.1.0').mkdir(parents=True)
        (self.cache/'build-cache').mkdir()
        self.backend_case = self.root/'r20'
        put(self.backend_case/'result.json', dict(round='R20', mode='heir-go-backend-build-preflight', status='ok',
            backend_library_build_passed=True, approved_go_version='1.26.0', lattigo_version='6.1.0',
            go_sdk_version_observed='go version go1.26.0 linux/arm64', work_directory=str(self.cache)))
        self.args = SimpleNamespace(model='vgg16', source_case=self.case, backend_case=self.backend_case,
            cache_source=self.cache, translate=self.translate, go=self.go, out=self.root/'new-case',
            registry=self.root/'registry.jsonl', run=True)
        fields = ('intake_result', 'graph_dir', 'program', 'frontend_result', 'probes', 'probe_source', 'pipeline', 'native_parameters')
        self.parameters = dict(logN=16, Q=[101, 103], P=[107], logDefaultScale=45)
        self.saved_source = dict(paths={key: str(self.root/key) for key in fields},
            contract=dict(model='vgg16_silu_avgpool', original_nodes=50), native_parameters=self.parameters,
            pipeline='builtin.module(synthetic-original-pass)')
        self.math = dict(passed=True, training_samples=3, activation_pairs=45, logit_elements=30, emitted_constants=61)
        self.worker = dict(status='ok', source=self.saved_source, lowered_adapter_contract=dict(nodes=46))
        self.artifact = dict(parameters=[dict(logN=16, Q=[101, 103], P=[107], log_default_scale=45)],
            static_bootstrap_operations=102, runtime_bootstraps=None, backend_executed=False, security_validated=False)
        put(self.case/'worker/model.mlir', 'synthetic source, never executed')
        put(Path(self.saved_source['paths']['native_parameters']), self.parameters)
        from pt_agent.experiments.heir_vgg_fused_pipeline import native_command
        # Exercise the real ordinary command builder, without invoking the compiler.
        original_command = native_command(self.tool/'heir-opt', self.case/'worker/model.mlir',
            self.case/'native', self.saved_source)
        put(self.case/'native/model-lattigo.mlir', 'synthetic native source, never read by admission')
        self.fake_pipeline = SimpleNamespace(ROUND='FIXTURE', MODE='fixture-complete-native', VARIANT='explicit-fusion',
            SOURCE_FIELDS=fields, ZERO_FIELDS=('go_codegen_runs', 'keygen_runs'), FORMAL=('accuracy_acceptance',),
            __file__=entry.__file__,
            source_gate=lambda sources: copy.deepcopy(self.saved_source),
            check_worker=lambda worker, source: copy.deepcopy(self.math))
        self.record = dict(status='ok', round='FIXTURE', mode='fixture-complete-native', variant='explicit-fusion',
            model='vgg16_silu_avgpool', model_compiled=True, strict_math_passed=True, quality_selection=False,
            original_model_changed=False, guard_trigger=None, incoming_signal=None, timeout_signal=None,
            wall_timeout_observed=False, go_codegen_runs=0, keygen_runs=0, accuracy_acceptance=None,
            attempted=dict(export_math=1, native_compile=1), completed=dict(export_math=1, native_compile=1),
            source=self.saved_source, worker=self.worker, static_native_bootstraps=102,
            native_request=dict(parameters=self.parameters, source_mlir=str(self.case/'worker/model.mlir'),
                pipeline=str(self.case/'native/explicit.pipeline'), original_model_contract=self.saved_source['contract'],
                lowered_adapter_contract=self.worker['lowered_adapter_contract']),
            phases=dict(export_math=dict(status='ok', observation=good_process(), audit=self.math),
                native_compile=dict(status='ok', observation=good_process(), artifact=self.artifact,
                    argv=original_command)))
        self.save_source()
        put(self.case/'completion.json', dict(status='ok', required_artifacts_persisted=True, registry_persisted=True))
        self.pipeline_patch = patch.object(entry, '_pipeline', return_value=self.fake_pipeline);self.pipeline_patch.start()
        self.addCleanup(self.pipeline_patch.stop)

    def save_source(self):
        put(self.case/'result.json', self.record);put(self.case/'source.json', self.saved_source)
        put(self.case/'worker/result.json', self.worker);put(self.case/'native/model-native-artifact.json', self.artifact)

    def fake_process(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if '--emit-lattigo-preprocessing' in argv:
            put(Path(argv[argv.index('-o')+1]), 'package heircnn_utils\nfunc init() {}\n')
        elif '--emit-lattigo-preprocessed' in argv:
            put(Path(argv[argv.index('-o')+1]), 'package heircnn\nfunc init() {}\n')
        return good_process()

    def execute_fake(self, runner=None, gate=None):
        self.calls = []
        with patch.object(entry.paths, 'RESULTS_DIR', self.root), \
             patch.object(entry, 'source_gate', gate or entry.source_gate), \
             patch.object(entry, '_available_ram_bytes', return_value=32*entry.GIB), \
             patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*entry.GIB)), \
             patch.object(entry, 'run_process', side_effect=runner or self.fake_process):
            return entry.execute(self.args)

    def test_original_validators_reused(self):
        self.pipeline_patch.stop()
        from pt_agent.experiments import heir_vgg_fused_pipeline as vgg, heir_alexnet_wide_pool as alex
        self.assertIs(entry._pipeline('vgg16'), vgg);self.assertIs(entry._pipeline('alexnet'), alex)

    def test_bound_source_and_no_native_mlir_payload_open(self):
        original = Path.read_text
        def reading(path, *args, **kwargs):
            if path.suffix == '.mlir':self.fail('Admission must stat the native file, not open it')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', reading):checked = entry.source_gate(self.args)
        self.assertEqual(checked['model']['model'], 'vgg16_silu_avgpool')
        self.assertEqual(checked['model']['strict_math'], self.math)

    def test_failure_unfinished_wrong_model_and_missing_math_rejected(self):
        for change in ({'status': 'error'}, {'model_compiled': False}, {'model': 'alexnet_silu_avgpool'},
                       {'strict_math_passed': False}, {'go_codegen_runs': 1}):
            with self.subTest(change=change):
                original = copy.deepcopy(self.record);self.record.update(change);self.save_source()
                with self.assertRaises(ValueError):entry.model_source_gate(self.args)
                self.record = original
        self.save_source()

    def test_native_parameter_or_outer_worker_mismatch_rejected(self):
        self.artifact['parameters'][0]['Q'][0] = 999;self.save_source()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.artifact['parameters'][0]['Q'][0] = 101;self.save_source()
        put(self.case/'worker/result.json', dict(status='error'))
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_closed_clean_source_required(self):
        for key, value in (('exit_code', 1), ('timed_out', True), ('signals_sent', [15]), ('remaining_pids', [123])):
            with self.subTest(key=key):
                original = self.record['phases']['native_compile']['observation']
                self.record['phases']['native_compile']['observation'] = dict(good_process(), **{key: value});self.save_source()
                with self.assertRaises(ValueError):entry.model_source_gate(self.args)
                self.record['phases']['native_compile']['observation'] = original
        self.save_source();put(self.case/'completion.json', dict(status='ok', registry_persisted=False))
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_source_lineage_and_original_compiler_path_required(self):
        for change in ({'source_mlir': str(self.root/'foreign.mlir')}, {'lowered_adapter_contract': dict(nodes=50)}):
            original = copy.deepcopy(self.record['native_request']);self.record['native_request'].update(change);self.save_source()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record['native_request'] = original
        self.save_source();command = self.record['phases']['native_compile']['argv']
        command[command.index('-o')+1] = str(self.root/'foreign.mlir');self.save_source()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_real_original_native_command_is_admitted_foreign_input_rejected(self):
        command = self.record['phases']['native_compile']['argv']
        self.assertEqual(command[1], '--mlir-disable-threading')
        self.assertEqual(command[2], '--pass-pipeline=builtin.module(synthetic-original-pass)')
        entry.model_source_gate(self.args)
        command[3] = str(self.root/'foreign-input.mlir');self.save_source()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_backend_version_and_cache_binding_rejected(self):
        self.assertEqual(entry.backend_gate(self.args)['lattigo_version'], '6.1.0')
        self.args.cache_source = self.root/'foreign'
        with self.assertRaises(ValueError):entry.backend_gate(self.args)
        self.args.cache_source = self.cache
        put(self.cache/'module/go.mod', 'module old\ngo 1.26.0\nrequire github.com/tuneinsight/lattigo/v6 v6.1.0\nreplace a => b\n')
        with self.assertRaises(ValueError):entry.backend_gate(self.args)

    def test_commands_match_original_split_library_only(self):
        command = entry.commands(self.args, self.args.out, entry.source_gate(self.args))
        self.assertEqual(command['library_build'], [str(self.go), 'build', '-p=1', '-mod=readonly', '-buildvcs=false', './...'])
        self.assertIn('--emit-lattigo-preprocessing', command['utils_emission'])
        self.assertIn('--package-name=heircnn_utils', command['utils_emission'])
        self.assertIn('--extra-imports=heircnn/heircnn_utils', command['main_emission'])
        self.assertIn('--package-name=heircnn', command['main_emission'])
        self.assertFalse(any('run' in argv or 'test' in argv for argv in command.values()))

    def test_offline_isolated_environment_preserves_home(self):
        with patch.dict(os.environ, {'HOME': '/original', 'GOPROXY': 'foreign', 'LD_PRELOAD': 'foreign', 'NIX_PATH': 'foreign'}):
            env = entry.environment(self.args.out)
        self.assertEqual(env['HOME'], '/original');self.assertEqual(env['GOPROXY'], 'off');self.assertEqual(env['GOSUMDB'], 'off')
        self.assertEqual(env['GOTOOLCHAIN'], 'local');self.assertNotIn('LD_PRELOAD', env);self.assertNotIn('NIX_PATH', env)
        self.assertEqual(env['GOMODCACHE'], str(self.args.out/'module-cache'))

    def test_full_fake_success_target_registry_and_completion(self):
        result = self.execute_fake()
        self.assertEqual(result['status'], 'ok');self.assertTrue(result['go_library_compile_accepted'])
        self.assertEqual(result['attempted'], {key: 1 for key in entry.PHASE_SECONDS})
        self.assertEqual(result['completed'], result['attempted']);self.assertEqual(len(self.calls), 3)
        for _, keywords in self.calls:
            self.assertEqual(keywords['mem_gib'], 4);self.assertEqual(keywords['fsize_gib'], 8)
        row = json.loads(self.args.registry.read_text())
        self.assertEqual(set(row), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
        self.assertEqual(row['model'], 'vgg16_silu_avgpool')
        marker = entry.read(self.args.out/'completion.json')
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['required_artifacts_persisted'])
        self.assertTrue(all(result[name] == 0 for name in entry.ZERO_FIELDS))
        self.assertTrue(all(result[name] is None for name in entry.FORMAL_FIELDS))

    def test_failed_emission_stops_build_retains_evidence(self):
        def failure(argv, **kw):
            observation = self.fake_process(argv, **kw)
            if '--emit-lattigo-preprocessed' in argv:observation['exit_code'] = 1
            return observation
        result = self.execute_fake(failure)
        self.assertEqual(result['status'], 'error');self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['attempted']['library_build'], 0)
        self.assertEqual(result['completed']['utils_emission'], 1)
        self.assertEqual(result['phases']['main_emission']['observation']['exit_code'], 1)
        self.assertFalse(result['go_library_compile_accepted']);self.assertTrue((self.args.out/'completion.json').is_file())

    def test_alex_registry_uses_target_model_not_shared_res_backend(self):
        self.args.model = 'alexnet';self.record['model'] = 'alexnet_silu_avgpool';self.save_source()
        result = self.execute_fake()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(json.loads(self.args.registry.read_text())['model'], 'alexnet_silu_avgpool')

    def test_host_available_guard_prevents_any_tool_and_closes_case(self):
        with patch.object(entry, 'physical_observation', return_value=dict(stop=True, reasons=['host available RAM below1.25GiB'])):
            result = self.execute_fake()
        self.assertEqual(result['status'], 'resource_stopped');self.assertEqual(self.calls, [])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_post_build_guard_retains_completed_child_and_rejects_acceptance(self):
        def physical(sample, available, free):
            stopped = len(self.calls) == 3
            return dict(stop=stopped, reasons=['host available RAM below1.25GiB'] if stopped else [])
        with patch.object(entry, 'physical_observation', side_effect=physical):result = self.execute_fake()
        self.assertEqual(result['status'], 'resource_stopped');self.assertTrue(result['library_build_passed'])
        self.assertEqual(result['completed']['library_build'], 1);self.assertFalse(result['go_library_compile_accepted'])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_source_failure_is_persisted_without_tool(self):
        def gate(args):raise ValueError('Original lowering has not completed')
        result = self.execute_fake(gate=gate)
        self.assertEqual(result['status'], 'error');self.assertEqual(self.calls, [])
        self.assertTrue((self.args.out/'completion.json').is_file())
        self.assertIn('not completed', result['error'])

    def test_prepare_uses_no_tools_and_keeps_old_dependency_lines(self):
        self.args.run = False;result = self.execute_fake()
        self.assertEqual(result['status'], 'prepared');self.assertFalse(result['library_build_passed'])
        self.assertEqual(self.calls, []);self.assertFalse((self.args.out/'module-cache').exists())
        self.assertEqual((self.args.out/'module/go.mod').read_text(),
            'module heircnn\n'+(self.cache/'module/go.mod').read_text().split('\n', 1)[1])

    def test_terminal_failure_does_not_admit_successful_build(self):
        self.args.out.mkdir()
        result = dict(status='error', library_build_passed=True)
        entry.terminal(self.args.out, result)
        self.assertFalse(result['go_library_compile_accepted']);self.assertTrue(result['library_build_passed'])

    def test_header_check_stops_before_generated_numeric_body(self):
        path = self.root/'synthetic.go';path.write_bytes(b'package heircnn\n'+b'\xff'*10000)
        self.assertEqual(entry.check_package(path, 'heircnn')['package'], 'heircnn')
        with self.assertRaises(ValueError):entry.check_package(path, 'main')

    def test_cli_requires_explicit_model_sources_no_worker_path(self):
        args = entry.parser().parse_args(['--model', 'alexnet', '--source-case', str(self.case), '--out', str(self.args.out),
            '--backend-case', str(self.backend_case), '--cache-source', str(self.cache), '--go', str(self.go),
            '--translate', str(self.translate), '--run'])
        self.assertEqual(args.model, 'alexnet');self.assertTrue(args.run)
        self.assertEqual(entry.TOTAL_SECONDS, 1800);self.assertEqual(entry.PHASE_SECONDS['library_build'], 600)


if __name__ == '__main__':unittest.main()
