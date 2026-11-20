"""Thin HEIR orchestration with fake finite commands and synthetic metadata."""
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

from pt_agent.experiments import heir_common_model_pipeline as entry


class CommonPipelineChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.model='vgg16_silu_avgpool';self.original=self.root/'original';self.original.mkdir()
        self.source={name:self.original/name for name in ('intake_result','frontend_result','graph_dir','program',
            'probes','probe_source','pipeline','native_parameters')}
        self.source['graph_dir'].mkdir()
        self.source['probes'].write_text('unread training payload fixture')
        self.source['program'].write_text('def synthetic(x):\n    return x\n')
        self.parameters=dict(logN=16,Q=list(range(101,112)),P=[201,202,203],logDefaultScale=45)
        self.write(self.source['native_parameters'],self.parameters)
        self.source['pipeline'].write_text('--pass-pipeline=builtin.module(secret-insert-mgmt-ckks{after-mul=true parameter-file='+str(self.source['native_parameters'])+'})\n')
        self.write(self.source['probe_source'],dict(split='train',indices=[0,1,2],sample_ids=['train:0','train:1','train:2']))
        self.write(self.source['intake_result'],dict(status='ok',model='pt_agent.experiments.r13_common_models:'+self.model))
        self.contract=dict(model=self.model,graph_nodes=0,activation_count=0,input_shape=[1,3,2,2],output_shape=[1,10],
            nodes=[],original_tensor_constant_names=[],original_tensor_constant_count=0,operator_counts={})
        self.write(self.source['frontend_result'],dict(status='ok',round='R17a',model='pt_agent.experiments.r13_common_models:'+self.model,
            worker=dict(status='ok',model='pt_agent.experiments.r13_common_models:'+self.model,heldout_acceptance=True,
                completed_samples=1000,math_probes_passed=True,artifacts=dict(program=str(self.source['program'])),
                rule_options=dict(deg=63,margin=1.5,slots=16384,placement='depth-dp',refresh='compiler',act_method='cheb'))))
        self.stack=[]
        for item in (patch.object(entry.paths,'RESULTS_DIR',self.root),patch.object(entry,'sources',return_value=self.source),
                     patch.object(entry,'read_graph_contract',side_effect=lambda directory,model:dict(self.contract,model=model)),
                     patch.object(entry,'_available_ram_bytes',return_value=16*entry.GIB),
                     patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100*entry.GIB))):
            item.start();self.addCleanup(item.stop)
        self.order=[]

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def args(self,**changes):
        options=dict(model=self.model,out=self.root/'new-pipeline',run=True,case_gib=32,registry=self.root/'registry.jsonl',
            export_python=Path('/offline/python'),heir_opt=Path('/offline/heir-opt'),translate=Path('/offline/heir-translate'),
            go=Path('/offline/go'),cache_source=self.root/'cache')
        options.update(changes);return Namespace(**options)

    def metadata(self):
        return dict(common_model=self.model,common_model_contract=self.contract,function=self.model,graph_nodes=0,
            input_shape=self.contract['input_shape'],output_shape=self.contract['output_shape'],approximations=[],
            exact_spatial_mean_matvec=True,fixed_local_avgpool_lowerings=[],exact_spatial_mean_lowerings=[])

    def adapter(self):
        return dict(status='ok',model=self.model,worker=dict(status='ok',common_model=self.model,common_model_contract=self.contract,
            activation_pairs=0,reference_forward_completed=3,adapter_forward_completed=3,quality_selection=False,constants=[],
            probes=[dict(sample_id='train:'+str(i),reference=[[float(x) for x in range(10)]],logit_check=dict(passed=True),activation_checks=[]) for i in range(3)]))

    def process(self,command,**kwargs):
        module=command[command.index('-m')+1]
        stage={'heir_cnn_export':'export','heir_cnn_math_check':'adapter_math','heir_cnn_preflight':'native_compile',
            'heir_cnn_codegen':'go_codegen','heir_native_clear_math':'clear_math'}[module.rsplit('.',1)[-1]]
        self.order.append(stage);self.assertEqual(kwargs['mem_gib'],12)
        self.assertLessEqual(kwargs['timeout'],entry.PHASE_SECONDS[stage]);self.assertGreater(kwargs['timeout'],0)
        target=Path(command[command.index('--output' if stage=='export' else '--out')+1]);target.mkdir(parents=True)
        Path(kwargs['resources']).write_text('');Path(kwargs['log']).write_text('')
        artifact=dict(parameters=[dict(logN=16,Q=self.parameters['Q'],P=self.parameters['P'],log_default_scale=45)],static_bootstrap_operations=17)
        dirs=entry.phase_directories(self.args().out)
        if stage=='export':
            self.write(target/'export.json',self.metadata());(target/'model.mlir').write_text('synthetic trusted MLIR fixture')
        elif stage=='adapter_math':self.write(target/'result.json',self.adapter())
        elif stage=='native_compile':
            self.write(target/'result.json',dict(status='ok',model=self.model,model_compiled=True,common_model_contract=self.contract))
            self.write(target/'model-native-artifact.json',artifact);(target/'model-lattigo.mlir').write_text('synthetic native source fixture')
        elif stage=='go_codegen':
            self.write(target/'result.json',dict(status='ok',model=self.model,library_build_passed=True,native_artifact=artifact,
                source_compiler_result=str(dirs['native_compile']/'result.json'),input_mlir=str(dirs['native_compile']/'model-lattigo.mlir')))
        else:
            self.write(target/'result.json',dict(status='ok',model=self.model,worker=dict(status='ok',native_math_attempts=3,native_math_completed=3,
                math_probes_completed=3,keygen_runs=0,encryptions=0,bootstraps=0,decryptions=0,
                samples=[dict(sample_id='train:'+str(i),reference=[float(x) for x in range(10)],observed=[float(x) for x in range(10)],passed=True) for i in range(3)])))
        return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])

    def test_model_commands_preserve_existing_contracts_and_expand_total_around7200_compile(self):
        for model in entry.MODELS:
            args=self.args(model=model);dirs=entry.phase_directories(args.out)
            commands=entry.commands(args,self.source,dirs,args.out/'native.pipeline')
            self.assertEqual(list(commands),list(entry.PHASE_SECONDS))
            for stage,command in commands.items():
                if stage!='go_codegen':self.assertEqual(command[command.index('--common-model')+1],model)
                self.assertNotIn('--exact-optiona-selector',command)
            for stage in ('export','adapter_math'):
                self.assertEqual('--exact-spatial-mean-matvec' in commands[stage],model=='vgg16_silu_avgpool')
            native=commands['native_compile'];self.assertEqual(native[native.index('--compile-seconds')+1],'7200')
            self.assertGreater(entry.PHASE_SECONDS['native_compile'],7200)
            self.assertEqual(entry.TOTAL_SECONDS,sum(entry.PHASE_SECONDS.values())+600)
            self.assertTrue(all(args.out in directory.parents for directory in dirs.values()))
        self.assertEqual(len(entry.MODELS),2)

    def test_source_gate_reuses_original_train_inputs_and_rejects_changed_policy(self):
        checked=entry.source_gate(self.model,self.source)
        self.assertEqual(checked['source_samples'],['train:0','train:1','train:2'])
        self.assertFalse(checked['weights_or_probe_payloads_read_by_source_gate'])
        self.assertFalse(checked['pipeline'].startswith('--pass-pipeline='))
        self.write(self.source['probe_source'],dict(split='report',indices=[0,1,2],sample_ids=['train:0','train:1','train:2']))
        with self.assertRaisesRegex(ValueError,'saved R13 training'):entry.source_gate(self.model,self.source)

    def test_source_only_preparation_launches_no_phase_and_closes_real_clock(self):
        with patch.object(entry,'run_process',side_effect=AssertionError('Preparation must execute no model/tool')):
            result=entry.execute(self.args(run=False))
        self.assertEqual(result['status'],'prepared');self.assertTrue(all(value==0 for value in result['attempted'].values()))
        self.assertEqual(len(result['commands']),5)
        marker=json.loads((self.args().out/'completion.json').read_text())
        self.assertTrue(marker['registry_persisted']);self.assertTrue(marker['required_artifacts_persisted'])
        self.assertEqual(marker['status'],'prepared');self.assertLess(marker['complete_case_elapsed_seconds'],5)
        self.assertNotIn('--pass-pipeline=',(self.args().out/'native.pipeline').read_text())

    def test_all_five_existing_phases_get_one_registry_and_after_persistence_completion(self):
        with patch.object(entry,'run_process',side_effect=self.process):result=entry.execute(self.args())
        self.assertEqual(self.order,list(entry.PHASE_SECONDS));self.assertEqual(result['status'],'ok')
        self.assertTrue(all(value==1 for value in result['attempted'].values()))
        self.assertTrue(all(value==1 for value in result['completed'].values()))
        self.assertEqual(result['static_native_bootstraps'],17)
        self.assertIsNone(result['compiled_bootstraps']);self.assertIsNone(result['accuracy_acceptance'])
        for name in entry.ZERO_FIELDS:self.assertEqual(result[name],0)
        rows=[json.loads(line) for line in self.root.joinpath('registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),6);self.assertEqual(len({row['run_id'] for row in rows}),6)
        expected={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
        self.assertTrue(all(set(row)==expected for row in rows))
        for stage,row in result['phases'].items():
            marker=json.loads((self.args().out/'phase-records'/stage/'completion.json').read_text())
            self.assertTrue(marker['registry_persisted']);self.assertEqual(marker['status'],'ok')
            self.assertGreaterEqual(marker['complete_case_elapsed_seconds'],row['elapsed_before_persistence_seconds'])
        final=json.loads((self.args().out/'completion.json').read_text())
        self.assertGreaterEqual(final['complete_case_elapsed_seconds'],sum(row['complete_phase_seconds'] for row in result['phases'].values()))

    def test_one_failed_compile_stops_this_model_without_retry_or_go_or_clear(self):
        def fail(command,**kwargs):
            observation=self.process(command,**kwargs)
            if self.order[-1]=='native_compile':observation['exit_code']=1
            return observation
        with patch.object(entry,'run_process',side_effect=fail):result=entry.execute(self.args())
        self.assertEqual(self.order,['export','adapter_math','native_compile']);self.assertEqual(result['failed_stage'],'native_compile')
        self.assertEqual(result['status'],'error');self.assertEqual(result['completed']['native_compile'],0)
        self.assertEqual(result['attempted']['go_codegen'],0);self.assertEqual(result['attempted']['clear_math'],0)
        self.assertEqual(json.loads((self.args().out/'phase-records/native_compile/completion.json').read_text())['status'],'error')
        self.assertEqual(json.loads((self.args().out/'completion.json').read_text())['status'],'error')

    def test_clear_audit_recomputes_original_tolerance_even_when_saved_pass_flag_is_true(self):
        def changed(command,**kwargs):
            observation=self.process(command,**kwargs)
            if self.order[-1]=='clear_math':
                path=entry.phase_directories(self.args().out)['clear_math']/'result.json';r=json.loads(path.read_text())
                r['worker']['samples'][0]['observed'][0]=1.1e-12;self.write(path,r)
            return observation
        with patch.object(entry,'run_process',side_effect=changed):result=entry.execute(self.args())
        self.assertEqual(result['status'],'error');self.assertEqual(result['failed_stage'],'clear_math')
        self.assertIn('Strict original mathematics',result['error']);self.assertEqual(result['completed']['clear_math'],0)

    def test_wrong_go_source_or_parameter_literal_stops_before_saved_inputs_read(self):
        def changed(command,**kwargs):
            observation=self.process(command,**kwargs)
            if self.order[-1]=='go_codegen':
                path=entry.phase_directories(self.args().out)['go_codegen']/'result.json';r=json.loads(path.read_text())
                r['native_artifact']['parameters'][0]['P']=[999,998,997];self.write(path,r)
            return observation
        with patch.object(entry,'run_process',side_effect=changed):result=entry.execute(self.args())
        self.assertEqual(result['failed_stage'],'go_codegen');self.assertIn('parameter literal differs',result['error'])
        self.assertNotIn('clear_math',self.order)

    def test_missing_clean_observation_fields_or_cleanup_signal_halts_next_stage(self):
        def interrupted(command,**kwargs):
            observation=self.process(command,**kwargs);observation['signal']=15;return observation
        with patch.object(entry,'run_process',side_effect=interrupted):result=entry.execute(self.args())
        self.assertEqual(self.order,['export']);self.assertEqual(result['failed_stage'],'export')
        self.assertEqual(result['completed']['export'],0);self.assertEqual(result['status'],'interrupted')
        self.assertEqual(result['incoming_signal'],15)

    def test_disk_reserve_failure_launches_no_child_and_preserves_terminal_record(self):
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=9*entry.GIB)), \
                patch.object(entry,'run_process',side_effect=AssertionError('Disk reserve must prevent launch')):
            result=entry.execute(self.args())
        self.assertEqual(result['status'],'resource_stopped');self.assertTrue(all(value==0 for value in result['attempted'].values()))
        self.assertTrue(result['resource_guard']['stop'])
        self.assertEqual(json.loads((self.args().out/'completion.json').read_text())['status'],'resource_stopped')

    def assert_terminal(self,out,status):
        self.assertEqual(json.loads((out/'result.json').read_text())['status'],status)
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],status)
        self.assertEqual(json.loads((out/'phase-records/export/result.json').read_text())['status'],status)
        self.assertEqual(json.loads((out/'phase-records/export/completion.json').read_text())['status'],status)
        rows=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        selected=[row for row in rows if row['run_id'] in (out.name,out.name+'--phase-export')]
        self.assertEqual(len(selected),2)
        self.assertTrue(all(row['metrics']['status']==status for row in selected))

    def test_outer_signal_handler_classifies_final_phase_and_case_without_losing_process_attempt(self):
        for signum,status in ((signal.SIGTERM,'interrupted'),(signal.SIGINT,'interrupted'),(signal.SIGALRM,'timeout')):
            with self.subTest(signum=signum):
                args=self.args(out=self.root/('signal-'+str(signum)))
                def stop(command,**kwargs):
                    Path(kwargs['resources']).write_text('');Path(kwargs['log']).write_text('')
                    signal.raise_signal(signum)
                    self.fail('Outer handler must stop the active fake command')
                with patch.object(entry,'run_process',side_effect=stop) as run:result=entry.execute(args)
                run.assert_called_once();self.assert_terminal(args.out,status)
                self.assertEqual(result['failed_stage'],'export');self.assertEqual(result['attempted']['export'],1)
                self.assertEqual(result['completed']['export'],0);self.assertEqual(sum(result['attempted'].values()),1)
                row=result['phases']['export'];self.assertIsNone(row['observation'])
                self.assertIn('HEIR supervisor received signal',row['supervision_exception'])
                self.assertIsNone(result['guard_trigger'])
                self.assertEqual(result['incoming_signal'],None if signum==signal.SIGALRM else signum)
                self.assertEqual(result['timeout_signal'],signum if signum==signal.SIGALRM else None)

    def test_physical_monitor_log_failure_still_signals_and_records_resource_stop(self):
        active=threading.Event();logged=threading.Event();original=entry.durable_json
        def available():return entry.GIB if active.is_set() else 16*entry.GIB
        def write(path,value):
            if Path(path).name=='resource-guard.json':logged.set();raise OSError('synthetic pipeline guard log failure')
            return original(path,value)
        def process(command,**kwargs):
            Path(kwargs['resources']).write_text('');Path(kwargs['log']).write_text('')
            active.set();time.sleep(1)
            self.fail('Monitor must terminate the fake command despite a guard log write failure')
        with patch.object(entry,'run_process',side_effect=process) as run, \
                patch.object(entry,'_available_ram_bytes',side_effect=available),patch.object(entry,'durable_json',side_effect=write):
            result=entry.execute(self.args())
        run.assert_called_once();self.assertTrue(logged.is_set());self.assert_terminal(self.args().out,'resource_stopped')
        self.assertIsNone(result['incoming_signal']);self.assertEqual(result['guard_stop_signal'],signal.SIGTERM)
        self.assertTrue(result['guard_trigger']['stop']);self.assertIn('synthetic pipeline guard log failure',result['monitor_stop_reason'])
        self.assertEqual(result['failed_stage'],'export');self.assertIsNone(result['phases']['export']['observation'])
        self.assertEqual(result['attempted']['adapter_math'],0)

    def test_inner_timeout_observation_is_preserved_and_stops_before_next_phase(self):
        def timeout(command,**kwargs):
            Path(kwargs['resources']).write_text('');Path(kwargs['log']).write_text('')
            return dict(exit_code=-15,timed_out=True,signal=None,signals_sent=[dict(pid=123,signal=15)],remaining_pids=[])
        with patch.object(entry,'run_process',side_effect=timeout) as run:result=entry.execute(self.args())
        run.assert_called_once();self.assert_terminal(self.args().out,'timeout')
        self.assertTrue(result['wall_timeout_observed']);self.assertIsNone(result['incoming_signal'])
        self.assertEqual(result['phases']['export']['observation']['signals_sent'],[dict(pid=123,signal=15)])
        self.assertEqual(result['attempted']['adapter_math'],0)


if __name__=='__main__':unittest.main()
