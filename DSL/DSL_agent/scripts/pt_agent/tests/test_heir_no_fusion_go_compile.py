"""Synthetic provenance and original Go-driver dispatch; no real tools or data."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_no_fusion_go_compile as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)


def clean(limits=None):
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
        started_ns=100, ended_ns=200, limits=limits or dict(memory_gib=24, file_gib=16, wall_seconds=7200))


class NoFusionGoCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.registry = self.root/'registry.jsonl';self.calls = [];self.gates = []
        self.cache = self.root/'cache';self.backend = self.root/'backend'
        self.tool = self.root/'tools/heir-opt';self.translate = self.root/'tools/heir-translate'
        self.go = self.cache/'toolchain/go/bin/go'
        for p in (self.tool, self.translate, self.go):put(p, 'opaque existing tool');p.chmod(0o700)
        put(self.cache/'toolchain/go/VERSION', 'go1.26.0\n')
        put(self.cache/'module/go.mod', 'module original\n\ngo 1.26.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.1.0\n')
        put(self.cache/'module/go.sum', 'opaque original module metadata\n')
        (self.cache/'module-cache/github.com/tuneinsight/lattigo/v6@v6.1.0').mkdir(parents=True)
        put(self.cache/'build-cache/metadata', 'original cache metadata')
        put(self.backend/'result.json', dict(round='R20', mode='heir-go-backend-build-preflight',
            status='ok', backend_library_build_passed=True, approved_go_version='1.26.0', lattigo_version='6.1.0',
            go_sdk_version_observed='go version go1.26.0 linux/arm64', work_directory=str(self.cache)))
        self.configure('alexnet', 'alex-serial-json')
        self.inherited_patch = patch.object(entry.native, 'source_gate', side_effect=self.inherited_gate)
        self.inherited_patch.start();self.addCleanup(self.inherited_patch.stop)
        self.scan_patch = patch.object(entry.native, '_scan', return_value={})
        self.scan_patch.start();self.addCleanup(self.scan_patch.stop)

    def configure(self, model, variant):
        self.index = getattr(self, 'index', 0)+1
        self.case = self.root/('heir-'+model+'-no-fusion-native-'+str(self.index))
        self.old_case = self.root/('original-'+model+'-failed-host-'+str(self.index))
        self.out = self.root/('fresh-'+model+'-Go-'+str(self.index))
        self.args = SimpleNamespace(model=model, source_variant=variant, source_case=self.case, out=self.out,
            backend_case=self.backend, cache_source=self.cache, translate=self.translate, go=self.go,
            registry=self.registry, run=True)
        parameters = dict(logN=16, Q=list(range(101, 112)), P=[113, 127, 131], logDefaultScale=45,
            encryptionType='pk', encryptionTechnique='standard')
        source_paths = {key:str(self.old_case/key) for key in entry.native.original.SOURCE_FIELDS}
        original_pipeline = ('builtin.module(\n  linalg-fuse-linear-ops,\n'
            '  layout-propagation{min-slot-count=16384},\n'
            '  generate-param-ckks{parameter-file='+source_paths['native_parameters']+'},\n'
            '  scheme-to-lattigo\n)')
        original_source = dict(paths=source_paths, native_parameters=parameters, pipeline=original_pipeline,
            contract=dict(model=entry.common.MODELS[model], role='original source contract'))
        source_mlir = self.old_case/'worker/model.mlir';put(source_mlir, 'opaque original source, not opened')
        put(self.old_case/'result.json', dict(status='error', strict_math_passed=True, marker='old failure retained'))
        self.original_failed_record = (self.old_case/'result.json').read_text()
        self.inherited = dict(model=entry.common.MODELS[model], source_variant=variant,
            source_case=str(self.old_case), original_source=original_source,
            original_source_mlir=dict(path=str(source_mlir), resource_size_bytes=17),
            original_native_argv=[str(self.tool), 'opaque original failed command'], original_parameters=parameters,
            inherited_strict_math=dict(passed=True, training_samples=3, activation_pairs=21 if model=='alexnet' else 45,
                logit_elements=30, emitted_constants=31 if model=='alexnet' else 61), new_math_runs=0,
            original_registry=dict(path=str(self.registry), row=dict(status='error')))
        native_dir = self.case/'native'
        pipeline = entry.native.remove_linear_fusion(entry.native._pipeline_at(original_source, native_dir))
        argv = entry.native._compiler_argv(str(self.tool), source_mlir, native_dir, pipeline)+list(entry.native.FAILURE_FLAGS)
        self.artifact = dict(parameters=[dict(logN=16,Q=parameters['Q'],P=parameters['P'],log_default_scale=45)],
            static_bootstrap_operations=9, runtime_bootstraps=None, backend_executed=False, security_validated=False)
        self.record = dict(round=entry.native.ROUND, mode=entry.native.MODE, variant=entry.native.VARIANT,
            model=entry.common.MODELS[model], status='ok', model_compiled=True, model_compiled_accepted=True,
            monitor_stopped=True, strict_math_passed=True, math_inherited=True, new_math_measured=False,
            quality_selection=False, original_model_changed=False, guard_trigger=None, incoming_signal=None,
            timeout_signal=None, child_timeout_signal=None, guard_stop_signal=None, wall_timeout_observed=False,
            attempted=dict(native_compile=1), completed=dict(native_compile=1), source=copy.deepcopy(self.inherited),
            limits=dict(total_seconds=7440,native_seconds=7200,AS_gib=24,subtree_RSS_gib=16,
                minimum_available_gib=1.25,minimum_free_disk_gib=10,file_gib=16,case_gib=32,
                concurrency=1,completion_reserve_bytes=entry.native.RESERVE_BYTES),
            native_argv=argv, static_native_bootstraps=9,
            phases=dict(native_compile=dict(status='ok',observation=clean(),argv=argv,artifact=copy.deepcopy(self.artifact))),
            native_request=dict(parameters=parameters,source_mlir=str(source_mlir),pipeline=str(native_dir/'explicit.pipeline'),
                pass_removed='linalg-fuse-linear-ops',removed_count=1,other_pass_order_unchanged=True),
            **{k:0 for k in entry.native.ZERO_FIELDS}, **{k:None for k in entry.native.original.FORMAL})
        self.completion = dict(round=entry.native.ROUND,status='ok',required_artifacts_persisted=True,
            registry_persisted=True,complete_case_elapsed_seconds=1.5)
        put(self.case/'source.json',self.record['source']);put(native_dir/'explicit.pipeline',pipeline+'\n')
        put(native_dir/'native-parameters.json',parameters);put(native_dir/'model-native-artifact.json',self.artifact)
        put(native_dir/'model-lattigo.mlir','opaque complete native MLIR; payload never read by gate')
        put(self.case/'phases/native_compile/resources.jsonl',json.dumps(dict(outer_process=dict(pid=500,starttime=20),
            processes=[dict(pid=501,starttime=21)]))+'\n')
        self.persist()

    def persist(self):
        put(self.case/'result.json',self.record);put(self.case/'completion.json',self.completion)
        keys=('status','attempted','completed','strict_math_passed','math_inherited','model_compiled',
            'model_compiled_accepted','static_native_bootstraps',*entry.native.original.FORMAL)
        row=dict(run_id=self.case.name,round=self.record['round'],mode=self.record['mode'],model=self.record['model'],
            git_rev=None,metrics={k:self.record[k] for k in keys},artifacts_dir=str(self.case),ts='fixture')
        self.row=row;put(self.registry,json.dumps(row)+'\n')

    def inherited_gate(self,args):
        self.gates.append(args)
        if (args.source_case!=self.old_case or args.source_variant!=self.args.source_variant
                or args.heir_opt!=self.tool or args.registry!=self.registry):raise ValueError('original source binding')
        return copy.deepcopy(self.inherited)

    def fake_process(self,command,**options):
        self.calls.append((command,options));put(options['log'],'fake tool log');put(options['resources'],'{}\n')
        if '--emit-lattigo-preprocessing' in command:put(Path(command[-1]),'package heircnn_utils\n')
        elif '--emit-lattigo-preprocessed' in command:put(Path(command[-1]),'package heircnn\n')
        else:self.assertEqual(command,[str(self.go),'build','-p=1','-mod=readonly','-buildvcs=false','./...'])
        return clean(dict(memory_gib=4,file_gib=8,wall_seconds=options['timeout']))

    def execute(self,process=None):
        with patch.object(entry.paths,'RESULTS_DIR',self.root), \
                patch.object(entry.common,'run_process',side_effect=process or self.fake_process), \
                patch.object(entry.common,'_available_ram_bytes',return_value=3*entry.native.GIB), \
                patch.object(entry.common.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.native.GIB)):
            return entry.execute(self.args)

    def test_exact_Alex_gate_returns_standard_common_payload_with_two_lineages(self):
        checked=entry.model_source_gate(self.args)
        self.assertEqual(checked['input_mlir']['path'],str(self.case/'native/model-lattigo.mlir'))
        self.assertEqual(checked['strict_math']['training_samples'],3)
        self.assertEqual(checked['no_fusion_source']['original_failed_host_case'],str(self.old_case))
        self.assertFalse(checked['previous_lowering_clock_included'])
        self.assertEqual(self.gates[0].source_case,self.old_case)

    def test_VGG_gate_retains_45_activations_and_original_source(self):
        self.configure('vgg16','vgg-rss12-json-overlap')
        checked=entry.model_source_gate(self.args)
        self.assertEqual(checked['strict_math']['activation_pairs'],45)
        self.assertEqual(checked['original_source'],self.inherited['original_source'])

    def test_old_host_gate_still_rejects_the_new_native_type(self):
        with self.assertRaises(ValueError):entry.host_sources.model_source_gate(self.args)

    def test_prepared_unaccepted_or_live_monitor_source_rejects(self):
        for key,value in [('status','prepared'),('model_compiled_accepted',False),('monitor_stopped',False),
                ('math_inherited',False),('new_math_measured',True),('sample_reads',False),('child_timeout_signal',15)]:
            previous=self.record[key];self.record[key]=value;self.persist()
            with self.subTest(key=key),self.assertRaises(ValueError):entry.model_source_gate(self.args)
            self.record[key]=previous

    def test_no_fictional_export_or_unbounded_source_policy(self):
        self.record['phases']['export_math']={};self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.record['phases'].pop('export_math');self.record['limits']['AS_gib']=48;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_original_source_metadata_drift_and_saved_outer_disagreement_reject(self):
        self.record['source']['original_parameters']['logDefaultScale']=50;self.persist()
        put(self.case/'source.json',self.record['source'])
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_complete_native_argv_and_pipeline_change_reject(self):
        self.record['native_argv']=self.record['native_argv'][:-1];self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.record['native_argv']=self.record['phases']['native_compile']['argv'];self.persist()
        p=self.case/'native/explicit.pipeline';put(p,p.read_text().replace('min-slot-count=16384','min-slot-count=8192'))
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_full_generated_prime_literal_or_static_inventory_rejects(self):
        changed=copy.deepcopy(self.artifact);changed['parameters'][0]['P'][1]=991
        put(self.case/'native/model-native-artifact.json',changed)
        self.record['phases']['native_compile']['artifact']=changed;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_exact_current_request_and_foreign_translator_reject(self):
        self.record['native_request']['removed_count']=True;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.record['native_request']['removed_count']=1;self.persist()
        foreign=self.root/'foreign/heir-translate';put(foreign,'foreign tool');foreign.chmod(0o700)
        self.args.translate=foreign
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_incomplete_closure_registry_duplicate_and_native_residual_reject(self):
        self.completion['registry_persisted']=False;self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.completion['registry_persisted']=True;self.persist()
        with self.registry.open('a')as stream:stream.write(json.dumps(self.row)+'\n')
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        self.persist();self.record['phases']['native_compile']['observation']['remaining_pids']=[501];self.persist()
        with self.assertRaises(ValueError):entry.model_source_gate(self.args)

    def test_source_process_birth_identity_checked_without_signaling(self):
        with patch.object(entry.native,'_scan',return_value={501:dict(starttime=21,state='R')}):
            with self.assertRaises(ValueError):entry.model_source_gate(self.args)
        with patch.object(entry.native,'_scan',return_value={501:dict(starttime=22,state='R')}):
            self.assertTrue(entry.model_source_gate(self.args)['no_fusion_source']['closed_native_identities'])

    def test_native_payload_not_opened_and_writing_native_builder_not_called(self):
        read=Path.read_text
        def metadata_only(path,*args,**kwargs):
            if path.suffix=='.mlir':raise AssertionError('native/source MLIR payload read')
            return read(path,*args,**kwargs)
        with patch.object(Path,'read_text',metadata_only), \
                patch.object(entry.native,'native_command',side_effect=AssertionError('writing native builder')):
            entry.model_source_gate(self.args)

    def test_existing_backend_and_offline_toolchain_gate_retained(self):
        checked=entry.source_gate(self.args)
        self.assertEqual(checked['backend']['go_version'],'1.26.0')
        self.assertEqual(checked['backend']['lattigo_version'],'6.1.0')
        put(self.cache/'toolchain/go/VERSION','go1.26.1\n')
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_private_original_execute_dispatch_exact_three_tools_no_runtime(self):
        namespace=entry._namespace(self.args)
        self.assertIs(namespace['execute'].__code__,entry.common.execute.__code__)
        self.assertIs(namespace['execute'].__globals__,namespace)
        self.assertIs(namespace['source_gate'],entry.source_gate)
        self.assertIs(namespace['serial_lock'],entry.common.serial_lock)
        result=self.execute()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['go_library_compile_accepted'])
        self.assertTrue(result['monitor_stopped']);self.assertEqual(len(self.calls),3)
        self.assertEqual(result['completed'],dict(utils_emission=1,main_emission=1,library_build=1))
        self.assertEqual(result['static_native_bootstraps'],9)
        self.assertTrue(all(result[k]==0 for k in entry.common.ZERO_FIELDS))
        self.assertTrue(all(result[k]is None for k in entry.common.FORMAL_FIELDS))
        self.assertFalse(result['generated_initializer_executed'])
        for command,options in self.calls:
            self.assertEqual(options['mem_gib'],4);self.assertEqual(options['fsize_gib'],8)
            self.assertEqual(options['env']['GOPROXY'],'off');self.assertEqual(options['env']['GOSUMDB'],'off')
            self.assertEqual(options['env']['GOENV'],'off');self.assertEqual(options['env']['GOWORK'],'off')
        self.assertEqual((self.old_case/'result.json').read_text(),self.original_failed_record)
        self.assertEqual(json.loads((self.case/'result.json').read_text()),self.record)
        rows=[json.loads(line)for line in self.registry.read_text().splitlines()]
        self.assertEqual(set(rows[-1]),entry.host_sources.REGISTRY_FIELDS)
        self.assertEqual(rows[-1]['round'],entry.ROUND);self.assertTrue(rows[-1]['metrics']['go_library_compile_accepted'])
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'ok')

    def test_fake_translator_error_stops_before_build_and_preserves_partial(self):
        def fail(command,**options):
            put(options['log'],'retained translator failure');return dict(clean(),exit_code=1)
        result=self.execute(fail)
        self.assertEqual(result['status'],'error');self.assertFalse(result['library_build_passed'])
        self.assertEqual(result['attempted'],dict(utils_emission=1,main_emission=0,library_build=0))
        self.assertTrue((self.out/'utils_emission.log').is_file());self.assertTrue((self.out/'completion.json').is_file())

    def test_stuck_private_monitor_withdraws_build_acceptance_only(self):
        joined=[]
        class Stuck:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def is_alive(self):return True
            def join(self,timeout):joined.append(timeout)
        with patch.object(entry.common.threading,'Thread',Stuck):result=self.execute()
        self.assertEqual(joined,[3]);self.assertFalse(result['monitor_stopped'])
        self.assertTrue(result['library_build_passed']);self.assertEqual(result['completed']['library_build'],1)
        self.assertEqual(result['status'],'error');self.assertFalse(result['go_library_compile_accepted'])

    def test_prepare_only_and_invalid_model_source_do_not_invoke_tools(self):
        self.args.run=False;result=self.execute()
        self.assertEqual(result['status'],'prepared');self.assertEqual(self.calls,[])
        self.assertFalse(result['go_library_compile_accepted'])
        self.args.out=self.root/'foreign-model-Go';self.args.model='vgg16'
        result=self.execute()
        self.assertEqual(result['status'],'error');self.assertEqual(self.calls,[])

    def test_parser_explicit_provenance_and_finite_options(self):
        argv=['--model','alexnet','--source-variant','alex-serial-json','--source-case',str(self.case),
            '--out',str(self.out),'--backend-case',str(self.backend),'--cache-source',str(self.cache),
            '--translate',str(self.translate),'--go',str(self.go),'--run']
        args=entry.parser().parse_args(argv)
        self.assertEqual(args.source_case,self.case);self.assertTrue(args.run)
        self.assertEqual(entry.common.PHASE_SECONDS,dict(utils_emission=300,main_emission=300,library_build=600))
        self.assertEqual(entry.common.TOTAL_SECONDS,1800)

    def test_terminal_case_quota_keeps_failed_protocol_and_completed_build_fact(self):
        actual=entry.common.artifact_bytes;terminal=[False]
        def quota(path):
            return entry.common.CASE_BYTES if terminal[0] else actual(path)
        def policy(out,result,deadline):terminal[0]=True
        with patch.object(entry.common,'artifact_bytes',side_effect=quota), \
                patch.object(entry.host_sources,'_terminal_policy',side_effect=policy):
            result=self.execute()
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['original_termination_status'],'resource_stopped')
        self.assertTrue(result['library_build_passed']);self.assertEqual(result['completed']['library_build'],1)
        self.assertFalse(result['go_library_compile_accepted'])
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'error')
        self.assertEqual(json.loads(self.registry.read_text().splitlines()[-1])['metrics']['status'],'error')


if __name__=='__main__':unittest.main()
