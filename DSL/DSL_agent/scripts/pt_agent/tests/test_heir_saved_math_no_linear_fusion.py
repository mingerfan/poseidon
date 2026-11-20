"""Saved-metadata and fake compiler checks for the explicit one-pass variant."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_saved_math_no_linear_fusion as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def observation(exit_code=0):
    return dict(exit_code=exit_code, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
        started_ns=100, ended_ns=200, termination_observed_ns=201)


class SavedMathCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.registry = self.root/'registry.jsonl'
        self.tool = self.root/'tools/heir-opt';put(self.tool, 'opaque existing tool fixture');self.tool.chmod(0o700)
        self.available = 3*entry.GIB;self.calls = []
        self.configure('alex-serial-json')
        self.pipeline_patch = patch.object(entry.host_sources, '_original_pipeline', return_value=self.pipeline)
        self.pipeline_mock = self.pipeline_patch.start();self.addCleanup(self.pipeline_patch.stop)
        self.scan_patch = patch.object(entry, '_scan', return_value={})
        self.scan_patch.start();self.addCleanup(self.scan_patch.stop)

    def configure(self, variant):
        self.descriptor = entry.host_sources._host_pipeline(variant)
        self.index = getattr(self, 'index', 0)+1
        self.case = self.root/(self.descriptor.host.OUT_PREFIX+'fixture'+str(self.index))
        self.out = self.root/('new-native-'+str(self.index))
        self.args = SimpleNamespace(source_case=self.case, source_variant=variant, out=self.out,
            heir_opt=self.tool, registry=self.registry, run=True)
        self.parameters = dict(logN=16, Q=list(range(101, 112)), P=[113, 127, 131],
            logDefaultScale=45, encryptionType='pk', encryptionTechnique='standard')
        source_paths = {key: str(self.root/'original-sources'/key) for key in entry.original.SOURCE_FIELDS}
        put(Path(source_paths['native_parameters']), self.parameters)
        text = ('builtin.module(\n  annotate-module{backend=lattigo scheme=ckks},\n'
            '  linalg-fuse-linear-ops,\n  layout-propagation{min-slot-count=16384},\n'
            '  convert-to-ciphertext-semantics{codegen-strategy=auto min-slot-count=16384 unroll-kernels=false},\n'
            '  generate-param-ckks{parameter-file='+source_paths['native_parameters']+'},\n'
            '  scheme-to-lattigo\n)')
        self.model = entry.host_sources.common.MODELS[entry.host_sources.VARIANTS[variant]]
        self.source = dict(paths=source_paths, pipeline=text, native_parameters=self.parameters,
            contract=dict(model=self.model, source='original graph contract'))
        audit = dict(passed=True, training_samples=3,
            activation_pairs=45 if self.model.startswith('vgg') else 21, logit_elements=30,
            emitted_constants=61 if self.model.startswith('vgg') else 31)
        worker = dict(status='ok', source=self.source, strict_math_passed=True, accepted_fixture=True,
            lowered_adapter_contract=dict(role='explicit original adapter'))
        worker_reference, audit_reference = copy.deepcopy(worker), copy.deepcopy(audit)
        def gate(paths):
            if {k: str(v) for k, v in paths.items()} != source_paths:raise ValueError('source binding')
            return copy.deepcopy(self.source)
        def check_worker(value, source):
            if value != worker_reference or source != self.source or value['accepted_fixture'] is not True:
                raise ValueError('original strict math rejection')
            return copy.deepcopy(audit_reference)
        self.pipeline = SimpleNamespace(VARIANT='original-representation', source_gate=gate, check_worker=check_worker)
        self.runtime = self.descriptor.numeric.host if self.model.startswith('vgg') else self.descriptor.host.host
        interpreter = self.runtime.host_python()
        self.math_command = [str(interpreter), '-B', '-m', self.descriptor.numeric.MODULE,
            '--worker', '--out', str(self.case/'worker'), '--memory-gib', '24']
        for key in entry.original.SOURCE_FIELDS:self.math_command += ['--'+key.replace('_', '-'), source_paths[key]]
        source_mlir = self.case/'worker/model.mlir'
        put(source_mlir, 'opaque original source, never opened by the gate')
        self.old_pipeline = entry._pipeline_at(self.source, self.case/'native')
        self.old_argv = entry._compiler_argv(self.tool, source_mlir, self.case/'native', self.old_pipeline)
        self.record = dict(round=self.descriptor.host.ROUND, mode=self.descriptor.host.MODE, model=self.model,
            variant=self.pipeline.VARIANT, status='error', strict_math_passed=True, quality_selection=False,
            original_model_changed=False, model_compiled=False, model_compiled_accepted=False,
            static_native_bootstraps=None, guard_trigger=None, incoming_signal=None, timeout_signal=None,
            wall_timeout_observed=False, runtime_variant=self.descriptor.runtime,
            timing_variant=self.descriptor.timing, constant_parser_variant=self.descriptor.parser,
            attempted=dict(export_math=1, native_compile=1), completed=dict(export_math=1, native_compile=0),
            source=self.source, worker=worker,
            host_runtime=dict(variant='r16a-existing-orion-host-runtime', interpreter=str(interpreter),
                last_observed_import_versions=self.runtime.OBSERVED_HOST_VERSIONS,
                original_Nix_versions_observed_here=False, original_Nix_runtime_equivalence_claimed=False,
                original_worker_code_unchanged=True, text_or_sample_or_weight_reads_in_import_check=0,
                nix_launcher_used=False, numerical_threads=1, strict_reference_or_tolerance_changed=False),
            phases=dict(export_math=dict(status='ok', observation=observation(), audit=audit, argv=self.math_command),
                native_compile=dict(status='error', observation=observation(1), argv=self.old_argv)),
            native_request=dict(parameters=self.parameters, source_mlir=str(source_mlir),
                pipeline=str(self.case/'native/explicit.pipeline'), original_model_contract=self.source['contract'],
                lowered_adapter_contract=worker['lowered_adapter_contract']),
            **{k: 0 for k in entry.original.ZERO_FIELDS}, **{k: None for k in entry.original.FORMAL})
        self.completion = dict(round=self.record['round'], status='error', required_artifacts_persisted=True,
            registry_persisted=True, complete_case_elapsed_seconds=3.25)
        put(self.case/'source.json', self.source);put(self.case/'worker/result.json', worker)
        put(self.case/'completion.json', self.completion)
        put(self.case/'native/explicit.pipeline', self.old_pipeline+'\n')
        put(self.case/'native/native-parameters.json', self.parameters)
        for phase in ('export_math', 'native_compile'):
            put(self.case/'phases'/phase/'resources.jsonl', json.dumps(dict(
                outer_process=dict(pid=310, starttime=5), processes=[dict(pid=311, starttime=6)]))+'\n')
        put(self.case/'phases/native_compile/worker.log', 'error: unexpected unrealized conversion cast op found\n')
        self.persist()

    def persist(self):
        put(self.case/'result.json', self.record)
        metrics = {k: self.record[k] for k in ('status', 'attempted', 'completed', 'strict_math_passed',
            'model_compiled', 'model_compiled_accepted', 'static_native_bootstraps')}
        self.registry_row = dict(run_id=self.case.name, round=self.record['round'], mode=self.record['mode'],
            model=self.model, git_rev=None, metrics=metrics, artifacts_dir=str(self.case), ts='saved-source')
        put(self.registry, json.dumps(self.registry_row)+'\n')

    def lease(self, expected=None):
        identity = dict(pid=10, starttime=20)
        if expected is not None:self.assertEqual(expected, identity)
        return dict(owner=identity, own_subtree_rss_bytes=32*1024**2, supervisor_rss_bytes=16*1024**2)

    def native_text(self, parameters=None):
        p = parameters or self.parameters
        return ('module { paramsLiteral = <logN = '+str(p['logN'])+', Q = '+str(p['Q'])+', P = '+str(p['P'])+
            ', logDefaultScale = '+str(p['logDefaultScale'])+'>\n%x = lattigo.ckks.bootstrap %y\n}')

    def fake_run(self, command, **options):
        self.calls.append((command, options))
        put(options['log'], 'bounded fake native compiler')
        put(options['resources'], '{}\n')
        put(self.out/'native/model-lattigo.mlir', self.native_text())
        return observation()

    def run_entry(self, runner=None):
        with patch.object(entry.paths, 'RESULTS_DIR', self.root), \
                patch.object(entry.serial, 'lease_observation', side_effect=self.lease), \
                patch.object(entry, '_available_ram_bytes', side_effect=lambda: self.available), \
                patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)), \
                patch.object(entry, 'run_process', side_effect=runner or self.fake_run):
            return entry.execute(self.args)

    def test_pass_removal_retains_every_other_character_and_rejects_ambiguity(self):
        old = 'builtin.module(\n  first,\n  linalg-fuse-linear-ops,\n  last\n)'
        self.assertEqual(entry.remove_linear_fusion(old), 'builtin.module(\n  first,\n  last\n)')
        for bad in [old+old, old.replace('linalg-fuse-linear-ops,', 'another,'),
                old.replace('linalg-fuse-linear-ops,', 'linalg-fuse-linear-ops{option=true},')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):entry.remove_linear_fusion(bad)

    def test_exact_closed_Alex_source_and_original_math_are_reused(self):
        checked = entry.source_gate(self.args)
        self.assertEqual(checked['inherited_strict_math']['activation_pairs'], 21)
        self.assertEqual(checked['original_source'], self.source)
        self.assertEqual(checked['new_math_runs'], 0)
        self.assertFalse(checked['original_MLIR_payload_read'])
        self.assertEqual(checked['original_native_argv'], self.old_argv)

    def test_exact_VGG_host_source_supported_without_live_tune_guard(self):
        self.configure('vgg-rss12-json-overlap')
        self.pipeline_mock.return_value = self.pipeline
        checked = entry.source_gate(self.args)
        self.assertEqual(checked['model'], 'vgg16_silu_avgpool')
        self.assertEqual(checked['inherited_strict_math']['activation_pairs'], 45)

    def test_source_rejects_incomplete_guarded_boolean_or_foreign_identity(self):
        for path, value in [('strict_math_passed', False), ('model_compiled_accepted', True),
                ('guard_trigger', {'reason': 'old guard'}), ('llm_calls', 1), ('llm_calls', False),
                ('round', 'foreign')]:
            original_value = self.record[path];self.record[path] = value;self.persist()
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):entry.source_gate(self.args)
            self.record[path] = original_value
        self.record['attempted']['native_compile'] = True;self.persist()
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_missing_completion_or_live_source_process_rejected(self):
        self.completion['registry_persisted'] = False;put(self.case/'completion.json', self.completion)
        with self.assertRaises(ValueError):entry.source_gate(self.args)
        self.completion['registry_persisted'] = True;put(self.case/'completion.json', self.completion)
        with patch.object(entry, '_scan', return_value={311:dict(starttime=6, state='R')}):
            with self.assertRaises(ValueError):entry.source_gate(self.args)
        with patch.object(entry, '_scan', return_value={311:dict(starttime=7, state='R')}):
            self.assertTrue(entry.source_gate(self.args)['original_closed_identities'])

    def test_original_native_argv_parameters_pipeline_and_math_argv_exact(self):
        mutations = [lambda:self.record['phases']['native_compile']['argv'].__setitem__(3, '/foreign/model.mlir'),
            lambda:self.record['phases']['export_math']['argv'].append('--foreign'),
            lambda:self.record['native_request']['parameters'].__setitem__('logDefaultScale', 50)]
        saved = copy.deepcopy(self.record)
        for mutate in mutations:
            self.record = copy.deepcopy(saved);mutate();self.persist()
            with self.assertRaises(ValueError):entry.source_gate(self.args)
        self.record = saved;self.persist()
        put(self.case/'native/explicit.pipeline', self.old_pipeline.replace('min-slot-count=16384', 'min-slot-count=8192'))
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_strict_worker_and_audit_cannot_be_substituted(self):
        self.record['phases']['export_math']['audit']['training_samples'] = 2;self.persist()
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_registry_unique_eight_fields_and_failed_fact_required(self):
        put(self.registry, json.dumps(self.registry_row)+'\n'+json.dumps(self.registry_row)+'\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)
        self.persist();row = copy.deepcopy(self.registry_row);row['extra'] = 0
        put(self.registry, json.dumps(row)+'\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_original_MLIR_is_not_opened_or_copied_and_argv_delta_is_exact(self):
        checked = entry.source_gate(self.args);self.out.mkdir()
        read_text = Path.read_text
        def bounded_read(path, *args, **kwargs):
            if path == self.case/'worker/model.mlir':raise AssertionError('payload read')
            return read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', bounded_read):command = entry.native_command(self.args, self.out, checked)
        pipeline = (self.out/'native/explicit.pipeline').read_text().strip()
        self.assertEqual(pipeline, entry.remove_linear_fusion(entry._pipeline_at(self.source, self.out/'native')))
        self.assertEqual(command[:-4], entry._compiler_argv(self.tool, self.case/'worker/model.mlir', self.out/'native', pipeline))
        self.assertEqual(tuple(command[-4:]), entry.FAILURE_FLAGS)
        self.assertEqual(json.loads((self.out/'native/native-parameters.json').read_text()), self.parameters)
        self.assertEqual((self.case/'native/explicit.pipeline').read_text(), self.old_pipeline+'\n')

    def test_physical16_and_original_host_reserves_include_supervisor(self):
        lease = self.lease();lease['own_subtree_rss_bytes'] = 16*entry.GIB
        self.assertFalse(entry.physical_observation(None, 2*entry.GIB, 12*entry.GIB, lease)['stop'])
        lease['own_subtree_rss_bytes'] += 1
        self.assertTrue(entry.physical_observation(None, 2*entry.GIB, 12*entry.GIB, lease)['stop'])
        lease = self.lease()
        self.assertTrue(entry.physical_observation(None, int(1.25*entry.GIB)-1, 12*entry.GIB, lease)['stop'])
        self.assertTrue(entry.physical_observation(None, 2*entry.GIB, 10*entry.GIB-1, lease)['stop'])

    def test_installed_tool_symlink_canonical_identity_keeps_recorded_argv(self):
        target = self.root/'installed/heir-opt';put(target, 'existing tool target');target.chmod(0o700)
        self.tool.unlink();self.tool.symlink_to(target)
        self.args.heir_opt = self.tool.resolve()
        checked = entry.source_gate(self.args);self.out.mkdir()
        command = entry.native_command(self.args, self.out, checked)
        self.assertEqual(command[0], str(self.tool));self.assertNotEqual(command[0], str(target))
        foreign = self.root/'foreign/heir-opt';put(foreign, 'foreign same-name tool');foreign.chmod(0o700)
        self.args.heir_opt = foreign
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_fake_success_one_phase_parameters_final_registry_and_scope(self):
        result = self.run_entry()
        self.assertEqual(result['status'], 'ok');self.assertTrue(result['model_compiled_accepted'])
        self.assertEqual(result['attempted'], {'native_compile':1});self.assertEqual(result['completed'], {'native_compile':1})
        self.assertEqual(result['static_native_bootstraps'], 1);self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]['mem_gib'], 24);self.assertEqual(self.calls[0][1]['fsize_gib'], 16)
        self.assertLessEqual(self.calls[0][1]['timeout'], 7200)
        self.assertTrue(result['math_inherited']);self.assertFalse(result['new_math_measured'])
        self.assertTrue(all(result[key] == 0 for key in entry.ZERO_FIELDS))
        self.assertTrue(all(result[key] is None for key in entry.original.FORMAL))
        rows = [json.loads(line) for line in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows), 2);self.assertEqual(set(rows[-1]), entry.host_sources.REGISTRY_FIELDS)
        self.assertEqual(rows[-1]['metrics']['status'], 'ok')
        self.assertTrue(json.loads((self.out/'completion.json').read_text())['registry_persisted'])
        self.assertEqual(json.loads((self.case/'result.json').read_text()), self.record)

    def test_fake_compiler_error_preserves_partial_log_and_durable_failure(self):
        def fail(command, **options):
            put(options['log'], 'original fake compiler error');put(options['resources'], '{}\n')
            return observation(1)
        result = self.run_entry(fail)
        self.assertEqual(result['status'], 'error');self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed']['native_compile'], 0)
        self.assertEqual(result['phases']['native_compile']['observation']['remaining_pids'], [])
        self.assertIn('compiler error', (self.out/'phases/native_compile/worker.log').read_text())
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'], 'error')

    def test_wrong_generated_parameters_rejected_after_clean_child(self):
        def bad(command, **options):
            returned = self.fake_run(command, **options);p = dict(self.parameters, logDefaultScale=50)
            put(self.out/'native/model-lattigo.mlir', self.native_text(p));return returned
        result = self.run_entry(bad)
        self.assertEqual(result['status'], 'error');self.assertFalse(result['model_compiled'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['phases']['native_compile']['observation']['exit_code'], 0)
        self.assertIsNone(result['static_native_bootstraps'])

    def test_signal_exception_restores_handlers_and_keeps_partial_uncertainty(self):
        before = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def interrupted(command, **options):
            put(options['log'], 'partial fake log')
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        result = self.run_entry(interrupted)
        self.assertEqual(result['incoming_signal'], signal.SIGTERM);self.assertEqual(result['status'], 'interrupted')
        self.assertIsNone(result['phases']['native_compile']['observation'])
        self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual({sig:signal.getsignal(sig) for sig in before}, before)
        self.assertTrue((self.out/'completion.json').is_file())
        # The original contextmanager releases the initializer slot on failure.
        with entry.serial_lock(self.root):pass

    def test_terminal_guard_revokes_acceptance_preserves_completed_native_fact(self):
        original_check = entry.original.check_native
        def complete_then_resource(*args):
            artifact = original_check(*args);self.available = entry.GIB;return artifact
        with patch.object(entry.original, 'check_native', side_effect=complete_then_resource):result = self.run_entry()
        self.assertEqual(result['status'], 'resource_stopped');self.assertTrue(result['model_compiled'])
        self.assertEqual(result['completed']['native_compile'], 1);self.assertFalse(result['model_compiled_accepted'])
        self.assertIsNotNone(result['guard_trigger'])

    def test_prepare_only_and_reused_out_do_not_invoke_tool(self):
        self.args.run = False;result = self.run_entry()
        self.assertEqual(result['status'], 'prepared');self.assertEqual(self.calls, [])
        self.assertEqual(result['attempted'], {'native_compile':0})
        self.assertFalse(result['model_compiled_accepted'])
        with self.assertRaises(ValueError):self.run_entry()

    def test_native_timeout_retains_actual_cleanup_and_timeout_category(self):
        def timed_out(command, **options):
            put(options['log'], 'partial timeout log')
            return dict(observation(1), timed_out=True, signal=signal.SIGTERM,
                signals_sent=[dict(pid=999, signal=15)], remaining_pids=[])
        result = self.run_entry(timed_out)
        self.assertEqual(result['status'], 'timeout');self.assertTrue(result['wall_timeout_observed'])
        self.assertIsNone(result['incoming_signal']);self.assertEqual(result['child_timeout_signal'], signal.SIGTERM)
        self.assertEqual(result['phases']['native_compile']['observation']['remaining_pids'], [])
        self.assertFalse(result['model_compiled_accepted'])

    def test_monitor_log_failure_still_requests_only_own_stop_in_finally(self):
        class ImmediateThread:
            def __init__(self, target, **kwargs):self.target = target
            def start(self):self.target()
            def is_alive(self):return False
        durable = entry.durable_json;observations = [0]
        def available():
            observations[0] += 1
            return 3*entry.GIB if observations[0] == 1 else entry.GIB
        def failed_monitor_log(path, value):
            if Path(path).name == 'monitor-stop.json':raise OSError('fake monitor log failure')
            return durable(path, value)
        with patch.object(entry.paths, 'RESULTS_DIR', self.root), \
                patch.object(entry.serial, 'lease_observation', side_effect=self.lease), \
                patch.object(entry, '_available_ram_bytes', side_effect=available), \
                patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*entry.GIB)), \
                patch.object(entry.threading, 'Thread', ImmediateThread), \
                patch.object(entry, 'durable_json', side_effect=failed_monitor_log), \
                patch.object(entry.os, 'kill') as kill, patch.object(entry, 'run_process') as native:
            result = entry.execute(self.args)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM);native.assert_not_called()
        self.assertEqual(result['status'], 'resource_stopped')
        self.assertTrue((self.out/'completion.json').is_file())

    def test_stuck_monitor_withdraws_acceptance_preserves_native_completion(self):
        joined = []
        class StuckThread:
            def __init__(self, **kwargs):pass
            def start(self):pass
            def is_alive(self):return True
            def join(self, timeout):joined.append(timeout)
        with patch.object(entry.threading, 'Thread', StuckThread):result = self.run_entry()
        self.assertEqual(joined, [3]);self.assertIs(result['monitor_stopped'], False)
        self.assertEqual(result['status'], 'error');self.assertIn('monitor did not stop', result['error'])
        self.assertTrue(result['model_compiled']);self.assertEqual(result['completed']['native_compile'], 1)
        self.assertFalse(result['model_compiled_accepted']);self.assertEqual(len(self.calls), 1)
        row = json.loads(self.registry.read_text().splitlines()[-1])
        self.assertEqual(row['metrics']['status'], 'error');self.assertFalse(row['metrics']['model_compiled_accepted'])
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'], 'error')

    def test_parser_explicit_source_and_unknown_options_rejected(self):
        parser = entry.parser()
        args = parser.parse_args(['--source-case', str(self.case), '--source-variant', 'alex-serial-json',
            '--out', str(self.out), '--heir-opt', str(self.tool), '--run'])
        self.assertEqual(args.source_case, self.case);self.assertTrue(args.run)
        with self.assertRaises(SystemExit):parser.parse_args(['--source-variant', 'unknown'])


if __name__ == '__main__':unittest.main()
