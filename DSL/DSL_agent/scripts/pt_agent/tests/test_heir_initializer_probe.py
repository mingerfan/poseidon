"""Bounded fake-process checks; generated model/native tools never execute."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_initializer_probe as entry


class InitializerProbeChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.generated=self.root/'accepted-go';self.module=self.generated/'module';self.module.mkdir(parents=True)
        self.clear=self.root/'accepted-clear';self.clear.mkdir()
        self.parameters=dict(logN=16,Q=list(range(101,112)),P=[201,202,203],log_default_scale=45)
        configure='func Resnet20_silu__configure() { ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:16,Q:[]uint64{'+','.join(map(str,self.parameters['Q']))+'},P:[]uint64{201,202,203},LogDefaultScale:45}) }'
        (self.module/'heircnn.go').write_text('package heircnn\nfunc init() { loadResource_float64("original", 4) }\n'+configure+'\n')
        utils=self.module/'heircnn_utils';utils.mkdir()
        (utils/'heircnn_utils.go').write_text('package heircnn_utils\nfunc init() { loadResource_float64("original",4) }\nfunc Resnet20_silu__preprocessing(param ckks.Parameters, encoder *ckks.Encoder) { encoder.Encode(v,pt); lintrans.NewTransformation(param,p) }\n')
        (self.module/'go.mod').write_text('module heircnn\n\ngo 1.26.0\nrequire github.com/tuneinsight/lattigo/v6 v6.1.0\n')
        (self.module/'go.sum').write_text('synthetic offline module fixture\n')
        for name in ('module-cache','build-cache'):
            (self.generated/name).mkdir();(self.generated/name/'existing').write_text('private cache fixture')
        self.go_result=self.generated/'result.json';self.clear_result=self.clear/'result.json'
        self.write(self.go_result,dict(status='ok',model=entry.MODEL,library_build_passed=True,
            native_artifact=dict(static_bootstrap_operations=102,parameters=[self.parameters])))
        self.write(self.clear_result,dict(status='ok',model=entry.MODEL))
        self.write(self.clear/'worker-result.json',dict(status='ok',math_probes_completed=3,native_math_completed=3,
            keygen_runs=0,samples=[dict(passed=True) for _ in range(3)]))
        self.paths=patch.object(entry.paths,'RESULTS_DIR',self.root);self.paths.start();self.addCleanup(self.paths.stop)
        self.available=patch.object(entry,'_available_ram_bytes',return_value=16*entry.GIB)
        self.available.start();self.addCleanup(self.available.stop)
        self.disk=patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB))
        self.disk.start();self.addCleanup(self.disk.stop)

    @staticmethod
    def write(path,record):path.write_text(json.dumps(record))

    def args(self, *, run=False):
        return Namespace(out=self.root/'new-probe',go_result=self.go_result,clear_result=self.clear_result,
            go=Path('/trusted/offline/go'),registry=self.root/'registry.jsonl',run=run)

    @staticmethod
    def observation(**changes):
        value=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[])
        value.update(changes);return value

    def fake_process(self,command,**kwargs):
        out=Path(kwargs['log']).parent
        (Path(kwargs['log'])).write_text('')
        Path(kwargs['resources']).write_text('')
        if command[1]=='build':
            self.assertEqual(kwargs['mem_gib'],4);self.assertLessEqual(kwargs['timeout'],300)
            self.assertEqual(command[2:5],['-p=1','-mod=readonly','-buildvcs=false'])
            Path(command[command.index('-o')+1]).write_text('unexecuted native fixture')
        else:
            self.assertEqual(kwargs['mem_gib'],12);self.assertLessEqual(kwargs['timeout'],300)
            launch=json.loads((out/'result.json').read_text())
            self.assertEqual(launch['attempts']['initialize'],1)
            self.assertEqual(launch['completed']['initialize'],0)
            self.assertIsNone(launch['main_entered'])
        self.assertEqual(kwargs['env']['GOPROXY'],'off');self.assertEqual(kwargs['env']['GODEBUG'],'inittrace=1')
        return self.observation()

    def test_source_gate_retains_exact_public_literal_and_rejects_changed_or_keyed_init(self):
        source=entry.source_gate(self.go_result,self.clear_result)
        self.assertEqual(source['parameters'],self.parameters);self.assertFalse(source['original_configure_called'])
        path=self.module/'heircnn.go';original=path.read_text()
        path.write_text(original.replace('LogN:16','LogN:17'))
        with self.assertRaisesRegex(ValueError,'public literals differ'):entry.source_gate(self.go_result,self.clear_result)
        path.write_text(original.replace('loadResource_float64("original", 4)','rlwe.NewKeyGenerator(param)'))
        with self.assertRaisesRegex(ValueError,'invokes cryptography'):entry.source_gate(self.go_result,self.clear_result)

    def test_client_imports_original_packages_and_calls_only_public_constructor_encoder_and_preprocessing(self):
        source=entry.client_source(self.parameters)
        self.assertIn('_ "heircnn"',source);self.assertIn('"heircnn/heircnn_utils"',source)
        self.assertIn('heircnn_utils.Resnet20_silu__preprocessing(p,encoder)',source)
        self.assertIn('Q:[]uint64{'+','.join(map(str,self.parameters['Q']))+'}',source)
        self.assertIsNone(entry.FORBIDDEN.search(source));self.assertNotIn('__configure(',source)
        self.assertNotIn('slices.Repeat',source)

    def test_preparation_writes_only_own_client_and_complete_registry_without_a_process(self):
        originals={name:(self.module/name).read_text() for name in ('heircnn.go','heircnn_utils/heircnn_utils.go')}
        with patch.object(entry,'run_process',side_effect=AssertionError('No native process in preparation')):
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'prepared');self.assertEqual(result['attempts'],dict(build=0,initialize=0))
        out=self.args().out;mod=(out/'module/go.mod').read_text()
        self.assertIn('module heir-initializer-probe',mod);self.assertIn('replace heircnn => '+json.dumps(str(self.module)),mod)
        self.assertFalse((out/'module-cache').exists())
        for name,source in originals.items():self.assertEqual((self.module/name).read_text(),source)
        rows=[json.loads(line) for line in self.root.joinpath('registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
        marker=json.loads((out/'completion.json').read_text());self.assertTrue(marker['registry_persisted'])
        self.assertEqual(marker['status'],'prepared')

    def test_memory_failure_before_main_is_a_retained_failed_measurement_with_zero_crypto(self):
        def process(command,**kwargs):
            observation=self.fake_process(command,**kwargs)
            if command[1]!='build':
                Path(kwargs['log']).write_text('fatal error: runtime: out of memory\nheircnn/heircnn_utils.init.0()\n')
                return self.observation(exit_code=2)
            return observation
        with patch.object(entry,'run_process',side_effect=process) as run:
            result=entry.execute(self.args(run=True))
        self.assertEqual(run.call_count,2);self.assertEqual(result['status'],'memory_failure')
        self.assertEqual(result['attempts'],dict(build=1,initialize=1))
        self.assertEqual(result['completed'],dict(build=1,initialize=0))
        self.assertFalse(result['main_entered']);self.assertIsNone(result['preprocessing_entered'])
        self.assertTrue(result['actual_failure_before_own_main']);self.assertTrue(result['diagnostic_completed'])
        for key in ('keygen_runs','encryptions','decryptions','bootstraps','model_forwards'):self.assertEqual(result[key],0)
        self.assertIsNone(result['security_acceptance']);self.assertIsNone(result['accuracy_acceptance'])
        self.assertEqual(json.loads((self.args().out/'completion.json').read_text())['status'],'memory_failure')

    def test_missing_main_marker_alone_leaves_failure_location_unobserved(self):
        def process(command,**kwargs):
            observed=self.fake_process(command,**kwargs)
            if command[1]!='build':
                Path(kwargs['log']).write_text('fatal error: runtime: out of memory\n')
                return self.observation(exit_code=2)
            return observed
        with patch.object(entry,'run_process',side_effect=process):result=entry.execute(self.args(run=True))
        self.assertEqual(result['status'],'memory_failure');self.assertIsNone(result['main_entered'])
        self.assertIsNone(result['preprocessing_entered']);self.assertIsNone(result['actual_failure_before_own_main'])
        self.assertFalse(result['generated_package_init_failure_stack']);self.assertFalse(result['worker_marker_present'])
        self.assertEqual(json.loads((self.args().out/'result.json').read_text())['main_entered'],None)

    def test_main_marker_and_preprocessing_entry_are_separate_from_failure_stack(self):
        main=entry.startup_observation(dict(main_entered=True,preprocessing_attempts=0),'runtime: out of memory')
        self.assertTrue(main['main_entered']);self.assertFalse(main['preprocessing_entered'])
        self.assertFalse(main['actual_failure_before_own_main']);self.assertIsNone(main['public_parameters_completed'])
        encoding=entry.startup_observation(dict(main_entered=True,public_parameters_completed=True,preprocessing_attempts=1),'')
        self.assertTrue(encoding['preprocessing_entered']);self.assertTrue(encoding['public_parameters_completed'])

    def test_success_requires_completed_real_encoding_counts_and_all_zero_operations(self):
        def process(command,**kwargs):
            observation=self.fake_process(command,**kwargs)
            if command[1]!='build':
                self.write(Path(command[1]),dict(status='ok',main_entered=True,preprocessing_attempts=1,
                    preprocessing_completed=1,plaintext_count=1743,linear_transform_count=23,
                    keygen_runs=0,encryptions=0,decryptions=0,bootstraps=0,model_forwards=0))
            return observation
        with patch.object(entry,'run_process',side_effect=process):result=entry.execute(self.args(run=True))
        self.assertEqual(result['status'],'ok');self.assertTrue(result['preprocessing_entered'])
        self.assertEqual(result['completed'],dict(build=1,initialize=1))
        changed=deepcopy(result['worker']);changed['keygen_runs']=1
        self.assertEqual(entry.classify(self.observation(),'',changed),'protocol_error')

    def test_failed_build_never_enters_the_initializer(self):
        def fail(command,**kwargs):
            self.fake_process(command,**kwargs);return self.observation(exit_code=1)
        with patch.object(entry,'run_process',side_effect=fail) as run:result=entry.execute(self.args(run=True))
        run.assert_called_once();self.assertEqual(result['status'],'build_error')
        self.assertEqual(result['attempts']['initialize'],0);self.assertIsNone(result['main_entered'])

    def assert_terminal(self,out,status):
        saved=json.loads((out/'result.json').read_text())
        self.assertEqual(saved['status'],status)
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],status)
        rows=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual([row['metrics']['status'] for row in rows if row['run_id']==out.name],[status])

    def test_outer_signal_handler_retains_failed_process_and_exact_final_classification(self):
        for signum,status in ((signal.SIGTERM,'interrupted'),(signal.SIGINT,'interrupted'),(signal.SIGALRM,'timeout')):
            with self.subTest(signum=signum):
                args=self.args(run=True);args.out=self.root/('signal-'+str(signum))
                def stop(command,**kwargs):
                    self.fake_process(command,**kwargs)
                    signal.raise_signal(signum)
                    self.fail('Outer handler must stop the active fake command')
                with patch.object(entry,'run_process',side_effect=stop) as run:result=entry.execute(args)
                run.assert_called_once();self.assert_terminal(args.out,status)
                self.assertEqual(result['attempts'],dict(build=1,initialize=0))
                self.assertEqual(len(result['processes']),1);row=result['processes'][0]
                self.assertIsNone(row['observation']);self.assertIn('HEIR supervisor received signal',row['supervision_exception'])
                self.assertIsNone(result['guard_trigger'])
                self.assertEqual(result['timeout_signal'],signum if signum==signal.SIGALRM else None)
                self.assertEqual(result['incoming_signal'],None if signum==signal.SIGALRM else signum)

    def test_guard_log_failure_still_terminates_active_command_and_persists_resource_stop(self):
        active=threading.Event();logged=threading.Event();original=entry.durable_json
        def available():return entry.GIB if active.is_set() else 16*entry.GIB
        def write(path,value):
            if Path(path).name=='resource-guard.json':logged.set();raise OSError('synthetic guard log failure')
            return original(path,value)
        def process(command,**kwargs):
            self.fake_process(command,**kwargs);active.set();time.sleep(1)
            self.fail('Resource monitor must interrupt the fake command despite log failure')
        with patch.object(entry,'run_process',side_effect=process) as run, \
                patch.object(entry,'_available_ram_bytes',side_effect=available),patch.object(entry,'durable_json',side_effect=write):
            result=entry.execute(self.args(run=True))
        run.assert_called_once();self.assertTrue(logged.is_set());self.assert_terminal(self.args().out,'resource_stopped')
        self.assertIsNone(result['incoming_signal']);self.assertEqual(result['guard_stop_signal'],signal.SIGTERM)
        self.assertTrue(result['guard_trigger']['stop']);self.assertIn('synthetic guard log failure',result['monitor_stop_reason'])
        self.assertEqual(result['attempts']['initialize'],0);self.assertIsNone(result['processes'][0]['observation'])
        self.assertIn('RuntimeError',result['processes'][0]['supervision_exception'])

    def test_inner_process_timeout_and_incoming_signal_are_saved_separately_from_cleanup_signals(self):
        for changed,status in ((dict(exit_code=-15,timed_out=True,signals_sent=[dict(pid=123,signal=15)]),'timeout'),
                               (dict(exit_code=-2,signal=2,signals_sent=[dict(pid=123,signal=15)]),'interrupted')):
            with self.subTest(status=status):
                args=self.args(run=True);args.out=self.root/('observation-'+status)
                def process(command,**kwargs):
                    self.fake_process(command,**kwargs);return self.observation(**changed)
                with patch.object(entry,'run_process',side_effect=process):result=entry.execute(args)
                self.assert_terminal(args.out,status);self.assertEqual(result['processes'][0]['observation']['signals_sent'],changed['signals_sent'])
                self.assertEqual(result['incoming_signal'],changed.get('signal'))
                self.assertEqual(result['wall_timeout_observed'],changed.get('timed_out',False))

    def test_missing_resource_and_external_interruption_do_not_acquire_a_memory_failure_label(self):
        self.assertEqual(entry.classify(self.observation(exit_code=2),'panic: open original: no such file',{}),'initialization_error')
        self.assertEqual(entry.classify(self.observation(exit_code=-15,signal=15),'',{}),'interrupted')
        self.assertEqual(entry.classify(self.observation(exit_code=-15,timed_out=True),'',{}),'timeout')
        self.assertEqual(entry.classify(self.observation(remaining_pids=[12]),'',{}),'supervision_error')
        path=self.root/'long.log';path.write_text('out of memory\n'+'x'*1024+'\nfinal init stack')
        excerpt=entry.log_excerpt(path,32)
        self.assertLessEqual(len(excerpt),65);self.assertIn('out of memory',excerpt);self.assertIn('final init stack',excerpt)

    def test_physical_guards_use_simultaneous_subtree_rss_available_and_free_space(self):
        sample=dict(subtree_rss_bytes=14*entry.GIB,sample_ts_ns=9)
        self.assertFalse(entry.physical_observation(sample,1.25*entry.GIB,10*entry.GIB)['stop'])
        for rss,available,free in ((14*entry.GIB+1,2*entry.GIB,11*entry.GIB),
                (1,1.25*entry.GIB-1,11*entry.GIB),(1,2*entry.GIB,10*entry.GIB-1)):
            row=entry.physical_observation(dict(sample,subtree_rss_bytes=rss),available,free)
            self.assertTrue(row['stop']);self.assertEqual(len(row['reasons']),1)
            self.assertEqual(entry.classify(self.observation(signal=15),'',{},row),'resource_stopped')

    def test_probe_serial_lock_rejects_an_overlapping_probe(self):
        with entry.serial_lock(self.root):
            with self.assertRaisesRegex(RuntimeError,'serial slot'):
                with entry.serial_lock(self.root):self.fail('Overlapping probe entered')


if __name__=='__main__':unittest.main()
