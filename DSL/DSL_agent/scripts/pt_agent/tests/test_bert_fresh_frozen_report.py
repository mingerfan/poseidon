"""Synthetic metadata fixtures; no model, dataset, report array or native tools."""
import copy
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_fresh_frozen_report as entry


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def clean():
    return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],
                started_ns=100,ended_ns=200)


def arguments(root):
    return SimpleNamespace(source_case=root/'fresh',supervisor_case=root/'owner',out=root/'report',
        compile_case=root/'fresh',metadata_template=root/'template.json',data_python=Path('/data-python'),
        intake_python=Path('/intake-python'),ledger=root/'ledger',allowance=root/'allowance',
        host_pid_namespace=os.readlink('/proc/self/ns/pid'))


class IsolatedReportTests(unittest.TestCase):
    def test_original_shell_functions_and_report_command_delegated(self):
        namespace=entry.namespace()
        for name in('shell_bindings','command','check_report','resources','claim','registry_row'):
            self.assertIs(namespace[name].__code__,getattr(entry.base,name).__code__)
        self.assertEqual(namespace['ROUND'],entry.ROUND)
        self.assertEqual(entry.base.ROUND,'BERT-FROZEN-FORK-REPORT')
        a=arguments(Path('/fixture'));command=namespace['command'](a,{'model':'/model','inputs':'/inputs'},200)
        self.assertEqual(entry._option(command,'--phase'),'report')
        self.assertEqual(entry._option(command,'--model-dir'),str(a.out/'model'))
        self.assertEqual(entry._option(command,'--session-dir'),str(a.out/'session'))
        self.assertNotIn(str(a.source_case/'session'),command)

    def test_real_shell_only_rebases_frozen_program_and_keeps_source_unmodified(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a=arguments(root);source=a.source_case
            for name in('model_info.json','calibration.json','samples/source.json'):
                write(source/'model'/name,{'original':name})
            (source/'model/graph').mkdir()
            for name in('calibration','heldout','confirm'):
                (source/'model/samples'/(name+'.npz')).write_text('unopened fixture')
            program='trusted tiny fixture'
            selected=dict(source=program,iteration=1,origin='llm')
            evaluation=dict(iteration=1,origin='llm',total=1000,passed=True,diverged=False,n_diverged=0,
                acc_drop_pp=0,max_u_calibrated=.5,max_u_analytic=.8,
                tail=dict(passed=True,total=3000,diverged=False,n_diverged=0,metrics={'acc_drop_pp':0}))
            confirm=dict(selected=selected,evals=[evaluation],confirm_rule='c2',applicable=True)
            state=dict(entry.serial.STATE_POLICY,phase='done',calls=1,iteration=2,failed=None)
            write(source/'session/SESSION.json',state)
            write(source/'session/AGENT.json',dict(status='feasible',confirm=confirm))
            (source/'session/program.ptdsl').write_text(program)
            marker=dict(program=str(source/'session/program.ptdsl'),confirm=confirm,origin='llm',iteration=1,
                **{key:state[key]for key in('prompt_version','tune_rule','confirm_rule','split_version')})
            write(source/'session/FROZEN.json',marker)
            a.out.mkdir()
            proof=entry.namespace()['prepare_shell'](a,{})
            self.assertEqual(proof['original_case_writes'],0)
            self.assertEqual(entry.base.read(source/'session/FROZEN.json'),marker)
            self.assertEqual(entry.base.read(source/'session/SESSION.json'),state)
            self.assertEqual(entry.base.read(a.out/'session/FROZEN.json')['program'],str(a.out/'session/program.ptdsl'))
            self.assertFalse((source/'session/REPORT_ACCESS.json').exists())
            self.assertFalse((source/'model/samples/report.npz').exists())
            self.assertTrue((a.out/'model/graph').is_symlink())
            write(a.out/'session/REPORT_ACCESS.json',{'report':True})
            self.assertFalse((source/'session/REPORT_ACCESS.json').exists())

    def _admission_fixture(self,root):
        a=arguments(root);case=a.source_case
        counters=dict(prepare=1,checkpoint=1,step=2,confirm=1,math=1,emit=1,trace=1,compile=1)
        checkpoint=root/'checkpoint/result.json';math=dict(source={'checkpoint_audit':str(checkpoint)})
        expected=dict(original_model='/original-model',model_info=str(case/'model/model_info.json'),
            program=str(case/'session/program.ptdsl'),frozen={'frozen':True},task_policy={'acc_drop_pp':.5},split_version='sst2-native-v1')
        phases=dict(math=math,**{key:dict(status='ok',source=expected,runtime_plan='/plan',compiled_bootstraps=3)
                               for key in('emit','trace','compile')})
        original=['python','-m','worker','--python',str(a.intake_python),'--ledger',str(a.ledger),'--allowance',str(a.allowance)]
        result=dict(round=entry.fresh.ROUND,mode=entry.fresh.MODE,model=entry.MODEL,status='ok',
            compile_accepted=True,frozen=True,monitor_stopped=True,termination=None,
            attempted=counters,completed=counters,phases=phases,runtime_plan='/plan',native_artifacts={'accepted':True},
            source={'metadata_template':str(a.metadata_template)},
            processes=[dict(phase='prepare',argv=original,observation=clean())],
            **dict.fromkeys(('report_samples','keygen_runs','ckks_runs','downloads','installations','integrity_checks'),0))
        write(case/'math/math-result.json',math)
        write(case/'completion.json',{'started_ns':0,'ended_ns':100})
        write(case/'two-hour-record.json',dict(round=entry.fresh.ROUND,mode=entry.fresh.MODE,status='ok',
            compile_accepted=True,persisted_pipeline_within_two_hours=True,started_ns=0,
            required_artifacts_and_registry_completed_ns=100,observed_ns=200))
        write(case/'model/samples/source.json',{'reader_python':str(a.data_python),'source':'/original-inputs'})
        return a,result,expected

    def test_new_source_gate_accepts_only_fresh_same_program_math_and_two_hour_record(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a,result,expected=self._admission_fixture(root)
            with patch.object(entry.paths,'RESULTS_DIR',root),\
                 patch.object(entry.base,'closed_case',return_value=(result,{'unique':True})),\
                 patch.object(entry.deferred,'compile_source',return_value=expected)as math_gate,\
                 patch.object(entry.fresh,'native_artifacts',return_value={'accepted':True}),\
                 patch.object(entry,'producer_closure',return_value={'saved_births_checked':3}):
                accepted=entry.admission(a)
                self.assertTrue(accepted['fresh_two_hour_accepted']);self.assertEqual(accepted['report_samples'],872)
                math_gate.assert_called_once()
                result['phases']['trace']['source']={'different_program':True}
                with self.assertRaisesRegex(ValueError,'same frozen source'):entry.admission(a)

    def test_unfinished_two_hour_or_reported_source_early_rejected(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a,result,expected=self._admission_fixture(root)
            with patch.object(entry.paths,'RESULTS_DIR',root),\
                 patch.object(entry.base,'closed_case',return_value=(result,{})),\
                 patch.object(entry.deferred,'compile_source')as gate:
                record=entry.base.read(a.source_case/'two-hour-record.json')
                write(a.source_case/'two-hour-record.json',dict(record,persisted_pipeline_within_two_hours=False))
                with self.assertRaisesRegex(ValueError,'two-hour'):entry.admission(a)
                gate.assert_not_called()
                result['round']='BERT-BUDGET-FORK-COMPILE'
                with self.assertRaisesRegex(ValueError,'new cold'):entry.admission(a)

    def test_real_owner_metadata_requires_all_recorded_births_closed(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a=arguments(root);owner=a.supervisor_case;case=a.source_case
            argv=['python','-m',entry.fresh.MODULE,'--out',str(case)]
            job=dict(id='fresh',out=str(case),status='ok',experiment_status='ok',cleanup_unknown=False,
                argv=argv,process=clean())
            run=dict(round='AUTONOMOUS-SERIAL',mode='finite-trusted-serial-experiments',status='completed_with_observations',
                     jobs=[job,{'id':'independent-search','status':'failed','process':{'remaining_pids':[]}}])
            write(owner/'result.json',run)
            write(owner/'completion.json',dict(round=run['round'],status=run['status'],required_artifacts_persisted=True,registry_persisted=True))
            write(owner/'manifest.json',{'jobs':[dict(id='fresh',out=str(case),argv=argv)]})
            write(owner/'fresh-command.json',argv)
            write(owner/'supervisor.json',dict(pid=11,process={'pid':11,'starttime':110}))
            sample=dict(ts_ns=150,outer_process={'pid':11,'starttime':110},processes=[{'pid':12,'starttime':120}])
            (owner/'fresh-resources.jsonl').write_text(json.dumps(sample)+'\n')
            case.mkdir()
            with patch.object(entry.base,'registry_row',return_value={}):
                accepted=entry.producer_closure(case,owner,a.host_pid_namespace,process_reader=lambda pid:None)
                self.assertEqual(accepted['saved_births_checked'],2)
                with self.assertRaisesRegex(ValueError,'birth remains'):
                    entry.producer_closure(case,owner,a.host_pid_namespace,
                        process_reader=lambda pid:{'starttime':120}if pid==12 else None)

    def test_private_finalization_masks_real_signals_during_join_and_persists(self):
        with TemporaryDirectory()as folder:
            root=Path(folder);a=arguments(root);a.source_case.mkdir();(root/'registry.jsonl').touch()
            scope=entry.namespace();phases=[]
            class Thread:
                def __init__(self,*args,**kwargs):self.live=False
                def start(self):self.live=True
                def is_alive(self):return self.live
                def join(self,timeout):
                    for sig in(signal.SIGTERM,signal.SIGALRM):
                        self.assert_masked(sig);signal.raise_signal(sig)
                    self.live=False
                def assert_masked(self,sig):
                    if signal.getsignal(sig)!=signal.SIG_IGN:raise AssertionError('Unmasked finalization signal')
            def runner(*args,**kwargs):
                write(a.out/'worker/worker_result.json',{'status':'ok','task_policy_passed':True})
                return clean()
            scope.update(admission=lambda args:{'model':'/original','inputs':'/inputs'},resources=lambda args:{'stop':False},
                claim=lambda args:'synthetic-claim',prepare_shell=lambda args,source:{},shell_bindings=lambda args:{},
                check_report=lambda args,worker:None,run_process=runner)
            with patch.object(entry.paths,'RESULTS_DIR',root),\
                 patch.object(entry.base.serial,'foreign_heavy_processes',return_value=[]),\
                 patch.object(entry.base.threading,'Thread',Thread):
                result=scope['execute'](a)
            self.assertEqual(result['status'],'ok');self.assertTrue(result['monitor_stopped'])
            self.assertTrue(entry.base.read(a.out/'completion.json')['registry_persisted'])
            self.assertEqual(len((root/'registry.jsonl').read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
