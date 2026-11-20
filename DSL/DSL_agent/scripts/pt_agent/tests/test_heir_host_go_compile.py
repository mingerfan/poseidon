"""Bounded saved-metadata and fake-tool checks; no model or tool is executed."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_host_go_compile as entry
from pt_agent.experiments import heir_vgg_fused_pipeline as contract


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def observation():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[], elapsed_seconds=.001)


class HostGoCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.backend = self.root/'backend';self.tools = self.root/'tools'
        self.translate = self.tools/'heir-translate';put(self.translate, 'existing opaque executable');self.translate.chmod(0o700)
        put(self.tools/'heir-opt', 'existing opaque executable')
        self.go = self.backend/'toolchain/go/bin/go';put(self.go, 'existing opaque SDK');self.go.chmod(0o700)
        put(self.backend/'toolchain/go/VERSION', 'go1.26.0\n')
        put(self.backend/'module/go.mod', 'module old\n\ngo 1.26.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.1.0\n')
        put(self.backend/'module/go.sum', 'original opaque module metadata\n')
        (self.backend/'module-cache/github.com/tuneinsight/lattigo/v6@v6.1.0').mkdir(parents=True)
        (self.backend/'build-cache').mkdir()
        self.backend_case = self.root/'r20'
        put(self.backend_case/'result.json', dict(round='R20', mode='heir-go-backend-build-preflight', status='ok',
            backend_library_build_passed=True, approved_go_version='1.26.0', lattigo_version='6.1.0',
            go_sdk_version_observed='go version go1.26.0 linux/arm64', work_directory=str(self.backend)))
        self.registry = self.root/'registry.jsonl';self.calls = []
        self.configure('vgg-rss12-json-overlap')
        self.original_patch = patch.object(entry, '_original_pipeline', side_effect=lambda model: self.pipeline)
        self.original_patch.start();self.addCleanup(self.original_patch.stop)

    def configure(self, variant):
        descriptor = entry._host_pipeline(variant);model = entry.VARIANTS[variant]
        self.fixture_index = getattr(self,'fixture_index',0)+1
        self.case = self.root/(descriptor.host.OUT_PREFIX+'fixture'+str(self.fixture_index))
        self.args = SimpleNamespace(model=model, source_variant=variant, source_case=self.case,
            backend_case=self.backend_case, cache_source=self.backend, go=self.go, translate=self.translate,
            out=self.root/('fresh-'+model+'-Go'), registry=self.registry, run=True)
        self.parameters = dict(logN=16, Q=[101, 103], P=[107], logDefaultScale=45)
        self.source = dict(paths={key: str(self.root/key) for key in entry.SOURCE_FIELDS},
            contract=dict(model=entry.common.MODELS[model]), native_parameters=self.parameters,
            pipeline='builtin.module(synthetic-original-pass)')
        for key in ('pipeline', 'native_parameters'):
            put(Path(self.source['paths'][key]), self.source[key])
        self.math = dict(passed=True, training_samples=3, activation_pairs=45 if model=='vgg16' else 21,
            logit_elements=30, emitted_constants=61 if model=='vgg16' else 31)
        self.worker = dict(status='ok', source=self.source, lowered_adapter_contract=dict(original=True))
        self.artifact = dict(parameters=[dict(logN=16, Q=[101, 103], P=[107], log_default_scale=45)],
            static_bootstrap_operations=12, backend_executed=False, security_validated=False)
        original = entry.common._pipeline(model)
        self.pipeline = SimpleNamespace(ROUND=original.ROUND, MODE=original.MODE, VARIANT=original.VARIANT,
            ZERO_FIELDS=contract.ZERO_FIELDS, FORMAL=contract.FORMAL, __file__=original.__file__,
            source_gate=lambda source: self.checked_source(source), check_worker=lambda worker, source: self.checked_worker(worker, source))
        # Real Alex has no global SOURCE_FIELDS. Preserve this fixture shape.
        if model == 'vgg16':self.pipeline.SOURCE_FIELDS = entry.SOURCE_FIELDS
        command = contract.native_command(self.tools/'heir-opt', self.case/'worker/model.mlir',
            self.case/'native', self.source)
        host_runtime = descriptor.numeric.host if model=='vgg16' else descriptor.host.host
        runtime = host_runtime.runtime_record(SimpleNamespace(python=host_runtime.host_python()))
        worker_command = [str(host_runtime.host_python()), '-B', '-m', descriptor.numeric.MODULE,
            '--worker', '--out', str(self.case/'worker'), '--memory-gib', '24']
        for name in entry.SOURCE_FIELDS:worker_command += ['--'+name.replace('_','-'), self.source['paths'][name]]
        self.record = dict(round=descriptor.host.ROUND, mode=descriptor.host.MODE, model=entry.common.MODELS[model],
            variant=original.VARIANT, status='ok', model_compiled=True, model_compiled_accepted=True,
            strict_math_passed=True, quality_selection=False, original_model_changed=False,
            guard_trigger=None, incoming_signal=None, timeout_signal=None, wall_timeout_observed=False,
            runtime_variant=descriptor.runtime, timing_variant=descriptor.timing, constant_parser_variant=descriptor.parser,
            host_runtime=runtime, attempted=dict(export_math=1,native_compile=1), completed=dict(export_math=1,native_compile=1),
            source=self.source, worker=self.worker, static_native_bootstraps=12, timing=dict(original_elapsed=3.25),
            native_request=dict(parameters=self.parameters, source_mlir=str(self.case/'worker/model.mlir'),
                pipeline=str(self.case/'native/explicit.pipeline'), original_model_contract=self.source['contract'],
                lowered_adapter_contract=self.worker['lowered_adapter_contract']),
            phases=dict(export_math=dict(status='ok',argv=worker_command,observation=observation(),audit=self.math),
                native_compile=dict(status='ok',argv=command,observation=observation(),artifact=self.artifact)),
            **dict.fromkeys(contract.ZERO_FIELDS,0), **dict.fromkeys(contract.FORMAL,None))
        put(self.case/'worker/model.mlir', 'opaque source payload, admission must never read')
        put(self.case/'native/model-lattigo.mlir', 'opaque native payload, admission must never read')
        self.save()

    def checked_source(self, source):
        self.assertEqual(source, {key:Path(value) for key,value in self.source['paths'].items()})
        return copy.deepcopy(self.source)

    def checked_worker(self, worker, source):
        self.assertEqual(worker,self.worker);self.assertEqual(source,self.source)
        return copy.deepcopy(self.math)

    def save(self):
        put(self.case/'result.json',self.record);put(self.case/'source.json',self.source)
        put(self.case/'worker/result.json',self.worker);put(self.case/'native/model-native-artifact.json',self.artifact)
        put(self.case/'completion.json',dict(status='ok',required_artifacts_persisted=True,registry_persisted=True,
            complete_case_elapsed_seconds=4.5))
        metrics={key:self.record[key]for key in('status','attempted','completed','strict_math_passed',
            'model_compiled','model_compiled_accepted','static_native_bootstraps')}
        row=dict(run_id=self.case.name,round=self.record['round'],model=self.record['model'],mode=self.record['mode'],
            git_rev=None,metrics=metrics,artifacts_dir=str(self.case),ts='fixture')
        self.registry.write_text(json.dumps(row)+'\n')

    def fake_tool(self, argv, **options):
        self.calls.append((argv,options))
        if '--emit-lattigo-preprocessing' in argv:put(Path(argv[-1]),'package heircnn_utils\nfunc init() {}\n')
        elif '--emit-lattigo-preprocessed' in argv:put(Path(argv[-1]),'package heircnn\nfunc init() {}\n')
        return observation()

    def execute_fake(self, runner=None):
        with patch.object(entry.paths,'RESULTS_DIR',self.root), \
             patch.object(entry.common,'_available_ram_bytes',return_value=32*entry.common.GIB), \
             patch.object(entry.common.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.common.GIB)), \
             patch.object(entry.common,'run_process',side_effect=runner or self.fake_tool):
            return entry.execute(self.args)

    def test_both_explicit_sources_preserve_original_clocks(self):
        for variant in entry.VARIANTS:
            self.configure(variant);checked=entry.model_source_gate(self.args)
            self.assertEqual(checked['strict_math'],self.math)
            self.assertEqual(checked['host_source']['original_complete_case_elapsed_seconds'],4.5)
            self.assertFalse(checked['host_source']['prior_lowering_clock_added_to_Go_clock'])

    def test_real_alex_facade_has_original_functions_without_SOURCE_FIELDS(self):
        self.original_patch.stop();self.args.model='alexnet';self.args.source_variant='alex-serial-json'
        original=entry.common._pipeline('alexnet');facade=entry._facade(self.args)
        self.assertFalse(hasattr(original,'SOURCE_FIELDS'));self.assertFalse(hasattr(facade,'SOURCE_FIELDS'))
        self.assertIs(facade.source_gate,original.source_gate);self.assertIs(facade.check_worker,original.check_worker)
        self.assertEqual(facade.VARIANT,original.VARIANT)

    def test_old_common_gate_still_rejects_new_host_identity(self):
        with patch.object(entry.common,'_pipeline',return_value=self.pipeline):
            with self.assertRaises(ValueError):entry.common.model_source_gate(self.args)
        self.assertEqual(entry.common.ROUND,'HEIR-COMMON-GO-COMPILE')

    def test_private_execute_code_and_imported_contextmanager_identity(self):
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['execute'].__code__,entry.common.execute.__code__)
        self.assertIs(namespace['serial_lock'],entry.common.serial_lock)
        self.assertIs(namespace['commands'],entry.common.commands)
        self.assertIs(namespace['backend_gate'],entry.common.backend_gate)

    def test_wrong_model_variant_round_mode_and_parser_rejected(self):
        self.args.model='alexnet'
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.args.model='vgg16'
        for change in ({'round':'foreign'},{'mode':'foreign'},{'variant':'foreign'},
                       {'constant_parser_variant':'AST'},{'runtime_variant':'foreign'}):
            original=copy.deepcopy(self.record);self.record.update(change);self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record=original

    def test_completed_fact_without_terminal_acceptance_rejected(self):
        for change in ({'status':'prepared'},{'model_compiled_accepted':False},{'strict_math_passed':False},
                       {'guard_trigger':{'stop':True}},{'incoming_signal':15}):
            original=copy.deepcopy(self.record);self.record.update(change);self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record=original

    def test_full_original_parameters_and_native_argv_reused(self):
        for key in ('Q','P','log_default_scale'):
            original=copy.deepcopy(self.artifact);self.artifact['parameters'][0][key]=[999]if key in('Q','P')else 999
            self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.artifact=original
        self.save();self.record['phases']['native_compile']['argv'][3]='foreign.mlir';self.save()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_actual_host_worker_all_source_arguments_are_bound(self):
        for index,value in ((0,'foreign-python'),(3,'foreign.module'),(-1,'foreign-source'),(9,'12')):
            original=copy.deepcopy(self.record);self.record['phases']['export_math']['argv'][index]=value;self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record=original

    def test_host_runtime_versions_and_nix_provenance_are_bound(self):
        for key,value in (('interpreter','foreign'),('numerical_threads',2),('nix_launcher_used',True),
                          ('last_observed_import_versions',{'torch':'foreign'})):
            original=copy.deepcopy(self.record);self.record['host_runtime'][key]=value;self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record=original

    def test_durable_completion_and_unique_outer_registry_required(self):
        (self.case/'completion.json').unlink()
        with self.assertRaises(FileNotFoundError):entry.model_source_gate(self.args)
        self.save();self.registry.write_text(self.registry.read_text()*2)
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.save();row=json.loads(self.registry.read_text());row['metrics']['model_compiled_accepted']=False
        self.registry.write_text(json.dumps(row)+'\n')
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_saved_signals_or_remaining_children_rejected(self):
        for key,value in (('exit_code',1),('timed_out',True),('signal',15),('remaining_pids',[123]),('signals_sent',[15])):
            original=copy.deepcopy(self.record);self.record['phases']['native_compile']['observation'][key]=value;self.save()
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record=original

    def test_source_gate_stats_payload_without_opening_it(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path.suffix in('.mlir','.npz','.ptdsl'):self.fail('Source payload read by metadata-only gate')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):entry.model_source_gate(self.args)

    def test_full_fake_build_closes_one_new_registry_and_completion(self):
        result=self.execute_fake()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['go_library_compile_accepted'])
        self.assertEqual(result['attempted'],dict.fromkeys(entry.common.PHASE_SECONDS,1))
        self.assertEqual(result['completed'],result['attempted']);self.assertEqual(len(self.calls),3)
        for argv,options in self.calls:
            self.assertEqual(options['mem_gib'],4);self.assertEqual(options['fsize_gib'],8)
            self.assertEqual(options['env']['GOENV'],'off');self.assertEqual(options['env']['GOWORK'],'off')
        self.assertEqual(self.calls[-1][0],[str(self.go),'build','-p=1','-mod=readonly','-buildvcs=false','./...'])
        rows=[json.loads(x)for x in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertEqual(set(rows[-1]),entry.REGISTRY_FIELDS)
        self.assertTrue(rows[-1]['metrics']['go_library_compile_accepted'])
        self.assertTrue(entry.common.read(self.args.out/'completion.json')['registry_persisted'])
        self.assertEqual(result['source']['model']['host_source']['original_timing'],self.record['timing'])
        self.assertTrue(all(result[k]==0 for k in entry.common.ZERO_FIELDS))

    def test_failed_translation_stops_before_build_and_keeps_actual_counts(self):
        def fail(argv,**options):
            value=self.fake_tool(argv,**options)
            if '--emit-lattigo-preprocessed'in argv:value['exit_code']=1
            return value
        result=self.execute_fake(fail)
        self.assertEqual(result['status'],'error');self.assertEqual(len(self.calls),2)
        self.assertEqual(result['completed']['utils_emission'],1);self.assertEqual(result['attempted']['library_build'],0)
        self.assertFalse(result['go_library_compile_accepted']);self.assertTrue((self.args.out/'completion.json').exists())

    def test_full_fake_Alex_uses_serial_source_identity_and_target_model(self):
        self.configure('alex-serial-json');result=self.execute_fake()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['go_library_compile_accepted'])
        self.assertEqual(result['model'],'alexnet_silu_avgpool');self.assertEqual(len(self.calls),3)
        self.assertEqual(result['source']['model']['host_source']['timing_variant'],'fresh-serial-host-runtime')
        self.assertFalse(result['live_source_execution_guard_reentered'])

    def test_source_rejection_is_persisted_before_any_tool(self):
        self.record['model_compiled_accepted']=False;self.save();result=self.execute_fake()
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])
        self.assertTrue((self.args.out/'completion.json').is_file())

    def test_late_resource_rejection_preserves_real_build_fact(self):
        def physical(*args):return dict(stop=len(self.calls)==3,reasons=['late original resource failure']if len(self.calls)==3 else[])
        with patch.object(entry.common,'physical_observation',side_effect=physical):result=self.execute_fake()
        self.assertEqual(result['status'],'error');self.assertTrue(result['library_build_passed'])
        self.assertEqual(result['completed']['library_build'],1);self.assertFalse(result['go_library_compile_accepted'])

    def test_absolute_deadline_and_case_reserve_are_terminal_acceptance_conditions(self):
        self.args.out.mkdir();result=dict(status='ok',library_build_passed=True)
        with patch.object(entry.common,'physical_observation',return_value=dict(stop=False,reasons=[])):
            entry._terminal_policy(self.args.out,result,time.monotonic()-1)
        self.assertEqual(result['status'],'error');self.assertTrue(result['wall_timeout_observed'])
        result=dict(status='ok',library_build_passed=True)
        with patch.object(entry.common,'physical_observation',return_value=dict(stop=False,reasons=[])), \
             patch.object(entry.common,'artifact_bytes',return_value=entry.common.CASE_BYTES):
            entry._terminal_policy(self.args.out,result,time.monotonic()+100)
        self.assertEqual(result['status'],'error');self.assertTrue(result['library_build_passed'])

    def test_prepare_only_keeps_dependencies_and_source_snapshots_no_tools(self):
        self.args.run=False;result=self.execute_fake()
        self.assertEqual(result['status'],'prepared');self.assertEqual(self.calls,[])
        files=entry.common.read(self.args.out/'source/snapshots.json')['files']
        self.assertIn(str(Path(entry.__file__).resolve()),[row['source']for row in files])
        self.assertFalse((self.args.out/'module-cache').exists())
        self.assertEqual((self.args.out/'module/go.mod').read_text(),
            'module heircnn\n'+(self.backend/'module/go.mod').read_text().split('\n',1)[1])

    def test_cli_requires_explicit_host_variant(self):
        argv=['--model','vgg16','--source-case',str(self.case),'--out',str(self.args.out),
            '--backend-case',str(self.backend_case),'--cache-source',str(self.backend),
            '--go',str(self.go),'--translate',str(self.translate)]
        with self.assertRaises(SystemExit):entry.parser().parse_args(argv)
        parsed=entry.parser().parse_args(argv+['--source-variant','vgg-rss12-json-overlap'])
        self.assertEqual(parsed.source_variant,'vgg-rss12-json-overlap')
        self.assertEqual(entry.common.TOTAL_SECONDS,1800)


if __name__=='__main__':unittest.main()
