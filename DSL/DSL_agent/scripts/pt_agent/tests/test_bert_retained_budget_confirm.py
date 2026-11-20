"""Synthetic closed-boundary recovery only; no model, sample, API or tool run."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from pt_agent.experiments import bert_retained_budget_confirm as entry


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def clean(code=0):
    return dict(exit_code=code, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case, self.run, self.out = [self.root/name for name in ('source', 'continuation', 'branch')]
        self.case.mkdir(); self.run.mkdir()
        self.args = SimpleNamespace(source_case=self.case, continuation_case=self.run,
            boundary_snapshot=self.run/'phases/002-step/before/session-SESSION.json', out=self.out,
            metadata_template=self.root/'compile_argv.json', intake_python=self.root/'nix-python',
            data_python=self.root/'data-python', ledger=self.root/'ledger.jsonl', allowance=self.root/'allowance.json')
        self.patch = mock.patch.object(entry.paths, 'RESULTS_DIR', self.root); self.patch.start(); self.addCleanup(self.patch.stop)
        self.compiler = mock.patch('pt_agent.experiments.r29_local_compile_matrix.compiler_argv', return_value=['compiler'])
        self.compiler.start(); self.addCleanup(self.compiler.stop)
        self.args.intake_python.touch(); self.args.data_python.touch()
        put(self.args.metadata_template, ['original-template']); (self.root/'registry.jsonl').touch()
        self.pool = [dict(source='candidate-'+str(i), origin=origin, iteration=3, latency_pred_us=100+i,
                          boots_pred=9180+i) for i, origin in enumerate(('tuned-r1','tuned-r2','tuned-r3','llm','tuned'))]
        self.state = dict(entry.serial.STATE_POLICY, phase='evaluate', iteration=4, calls=3, pending=None, failed=None,
            best_iteration=3, first_feasible=3, best_program=self.pool[-1]['source'], best=dict(boots_pred=9180),
            history=[dict(iteration=i) for i in (1,2,3)], pool=self.pool,
            records={'$dict': [[{'$tuple':[3,'tuned',self.pool[-1]['source']]}, [{'latency_pred_us':104}, 1]]]},
            messages=[dict(role='system',content='synthetic-only'),dict(role='user',content='synthetic-only')],
            tune_cache={}, prompt_tokens=99, completion_tokens=66, elapsed_s=123., stale=0)
        put(self.args.boundary_snapshot, self.state)
        after = entry.state_summary(self.state)
        current = copy.deepcopy(self.state); current.update(phase='tune', calls=4, pending=dict(iteration=4))
        put(self.case/'session/SESSION.json', current)
        rows = [dict(phase='step',source_state_before=dict(after, phase='tune',iteration=3),
                     source_state_after=after,observation=clean()),
                dict(phase='step',source_state_before=after,source_state_after=entry.state_summary(current),observation=clean()),
                dict(phase='step',source_state_before=entry.state_summary(current),source_state_after=None,
                     observation=dict(clean(143), signal=15, signals_sent=[dict(pid=123,signal=15)]))]
        put(self.run/'result.json', dict(round=entry.serial.ROUND, model=entry.MODEL,status='error',
            source=dict(source_case=str(self.case)), processes=rows,attempted=dict(report=0,compile=0)))
        put(self.run/'completion.json', dict(status='error',required_artifacts_persisted=True,registry_persisted=True,
                                             complete_case_elapsed_seconds=900.))
        tune_worker = self.worker('evaluate'); tune_worker['step'].update(step='tune',next_step='evaluate')
        put(self.args.boundary_snapshot.with_name('result.json'),dict(status='ok',model=entry.MODEL,
                                                                     worker=tune_worker,process=clean()))
        model = self.root/'HF'; model.mkdir(); (model/'pytorch_model.bin').write_text('synthetic metadata-only fixture')
        cfg = dict(model_type='bert',architectures=['BertForSequenceClassification'],hidden_size=2,num_hidden_layers=2,num_labels=2)
        put(model/'config.json',cfg)
        partition=dict(kind='sst2_native',version='sst2-native-v1',seed=20261024,
            **{key:list(range(start,start+count)) for key,start,count in
               [('calibration',0,1000),('heldout',1000,1000),('confirm',2000,1000),('report',3000,872)]})
        info=dict(task='classification',metric='classification_top1',threshold={'acc_drop_pp':.5},graph='graph',
            split=partition,source=dict(local_path=str(model),inputs_path=str(self.root/'sst2'),config=cfg))
        put(self.case/'model/model_info.json',info); put(self.case/'model/calibration.json',{})
        put(self.case/'model/samples/source.json',dict(partition=partition,native_test_read=False,source=str(self.root/'sst2')))
        for name in ('samples/calibration.npz','samples/heldout.npz','graph/graph.json','graph/weights.npz'):
            p=self.case/'model'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic only')
        prepared=dict(status='ok',phase='prepare',model=entry.MODEL,import_completed=True,
            model_dir=str(self.case/'model'),model_metadata=dict(unchanged_original_model=str(model)),
            reference_checks=[dict(passed=True)]*3,**dict.fromkeys(('native_test_reads','report_samples','confirm_samples',
            'llm_calls','compiler_runs','keygen_runs','real_ckks_attempts','real_ckks_completed'),0))
        put(self.case/'phases/001-prepare/worker_result.json',prepared)

    def worker(self, next_step='done'):
        return dict(status='ok',phase='step',model=entry.MODEL,next_step=next_step,llm_calls=0,frozen=next_step=='done',
            report_samples=0,step=dict(step='confirm',next_step=next_step,session_calls=3),
            **dict.fromkeys(entry.original.FORMAL,None),**dict.fromkeys(('native_test_reads','compiler_runs','keygen_runs',
                'real_ckks_attempts','real_ckks_completed'),0))

    def finish(self, *, selected=True, mutate_pool=False):
        state = entry.read(self.out/'session/SESSION.json')
        self.assertEqual(state['phase'],'confirm'); self.assertEqual(state['pool'],self.pool)
        state.update(phase='done',elapsed_s=124.)
        if mutate_pool: state['pool']=state['pool'][:-1]
        put(self.out/'session/SESSION.json',state)
        item=copy.deepcopy(self.pool[-1]) if selected else None
        row=dict(iteration=3,origin='tuned',total=1000,passed=True,diverged=False,n_diverged=0,
            acc_drop_pp=.1,max_u_calibrated=.8,max_u_analytic=.9,
            tail=dict(passed=True,total=3000,diverged=False,n_diverged=0,metrics=dict(acc_drop_pp=.1)))
        confirmation=dict(selected=item,evals=[row] if selected else [],confirm_rule='c2',applicable=True)
        put(self.out/'session/AGENT.json',dict(status='feasible' if selected else 'confirm_failed',confirm=confirmation))
        if selected:
            (self.out/'session/program.ptdsl').write_text(item['source'])
            put(self.out/'session/FROZEN.json',dict(program=str(self.out/'session/program.ptdsl'),confirm=confirmation,
                origin=item['origin'],iteration=3,**{key:self.state[key] for key in ('prompt_version','tune_rule','confirm_rule','split_version')}))
        worker=self.worker(); worker['frozen']=selected
        put(self.out/'phases/001-confirm/worker_result.json',worker)
        return clean()

    def execute(self, runner=None, resources=None):
        with mock.patch.object(entry,'run_process',side_effect=runner or (lambda *a,**k:self.finish())), \
             mock.patch.object(entry,'check_resources',side_effect=resources or (lambda *a,**k:{})), \
             mock.patch('pt_agent.registry.git_rev',return_value='synthetic'):
            return entry.execute(self.args)

    def test_actual_closed_metadata_shape_and_whole_pool_are_admitted(self):
        result=entry.source_gate(self.args)
        self.assertEqual(result['pool_size'],5); self.assertEqual(result['inherited_calls'],3)
        self.assertEqual(result['original_calls'],4); self.assertEqual(result['excluded_original_calls'],[4])
        self.assertEqual(result['boundary']['max_iters'],10)

    def test_running_missing_completion_or_unknown_cleanup_refuse_before_numeric(self):
        run=entry.read(self.run/'result.json')
        for key,value in (('status','running'),('processes',[])):
            with self.subTest(key=key):
                changed=copy.deepcopy(run);changed[key]=value;put(self.run/'result.json',changed)
                with self.assertRaises(ValueError):entry.source_gate(self.args)
        put(self.run/'result.json',run)
        run['processes'][-1]['observation']['cleanup_unknown']=True;put(self.run/'result.json',run)
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_incomplete_boundary_policy_pool_and_pending_refuse(self):
        for change in (dict(max_iters=3),dict(pending={'iteration':3}),dict(calls=4),dict(pool=self.pool[:1]),dict(confirmation={})):
            with self.subTest(change=list(change)):
                put(self.args.boundary_snapshot,dict(self.state,**change))
                with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_existing_confirmation_report_or_cache_never_reused(self):
        for name in ('session/AGENT.json','session/FROZEN.json','session/REPORT_ACCESS.json','model/samples/confirm.npz'):
            with self.subTest(name=name):
                p=self.case/name;p.write_text('{}')
                with self.assertRaises(ValueError):entry.source_gate(self.args)
                p.unlink()
        current=entry.read(self.case/'session/SESSION.json');current['phase']='confirm';put(self.case/'session/SESSION.json',current)
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_shell_has_only_phase_change_and_new_cache_cannot_write_source(self):
        source=entry.source_gate(self.args);self.out.mkdir()
        before=entry.prepare_branch(self.args,source)
        expected=copy.deepcopy(self.state);expected['phase']='confirm'
        self.assertEqual(before,expected);self.assertEqual(entry.read(self.args.boundary_snapshot),self.state)
        self.assertFalse((self.out/'model').is_symlink());self.assertFalse((self.out/'model/samples').is_symlink())
        self.assertTrue((self.out/'model/graph').is_symlink())
        self.assertTrue((self.out/'model/samples/calibration.npz').is_symlink())
        self.assertEqual(entry.read(self.out/'model/model_info.json'),entry.read(self.case/'model/model_info.json'))
        (self.out/'model/samples/confirm.npz').write_text('fake confirm cache')
        self.assertFalse((self.case/'model/samples/confirm.npz').exists())

    def test_original_recorded_worker_command_keeps_nix_and_no_report_or_api_phase(self):
        source=entry.source_gate(self.args);argv=entry.command(self.args,source,self.out/'work')
        self.assertEqual(argv[argv.index('--seconds')+1],'21660')
        self.assertEqual(argv[argv.index('--step-seconds')+1],'21600')
        self.assertEqual(argv[argv.index('--phase')+1],'step')
        self.assertEqual(argv[argv.index('--model-dir')+1],str(self.out/'model'))
        self.assertIn(entry.serial.RECORDED,argv);self.assertIn('--delegate-worker',argv)
        self.assertNotIn('report',argv);self.assertNotIn('evaluate',argv)

    def test_real_original_worker_dispatches_confirm_without_constructing_provider(self):
        from pt_agent.experiments import bert_sst2_worker as worker
        from pt_agent.planners import transformer_session as sessions
        source=entry.source_gate(self.args);self.out.mkdir();entry.prepare_branch(self.args,source)
        restored=sessions.load_state(self.out/'session/SESSION.json')
        self.assertIn((3,'tuned',self.pool[-1]['source']),restored['records'])
        args=SimpleNamespace(session_dir=self.out/'session')
        recorder=entry.StageRecorder(self.out/'dispatch-events.jsonl');result={};seen=[]
        def session(ctx,directory,**kwargs):
            seen.append((directory,kwargs))
            return dict(next_step='done',session_calls=3,step='confirm')
        with mock.patch.object(worker,'load_context',return_value=SimpleNamespace()), \
             mock.patch.object(sessions,'session_step',side_effect=session), \
             mock.patch('pt_agent.autonomous_budget.RenewingLLMClient',side_effect=AssertionError('Provider construction forbidden')):
            worker.step(args,recorder,result)
        self.assertEqual(len(seen),1);self.assertIsNone(seen[0][1]['client'])
        self.assertEqual(seen[0][1]['max_iters'],10);self.assertEqual(seen[0][1]['confirm_rule'],'c2')
        self.assertEqual(result['llm_calls'],0);self.assertEqual(result['report_samples'],0)

    def test_source_continuation_prior_report_or_compile_attempt_refuses(self):
        run=entry.read(self.run/'result.json')
        for phase in ('report','compile'):
            changed=copy.deepcopy(run);changed['attempted'][phase]=1;put(self.run/'result.json',changed)
            with self.subTest(phase=phase):
                with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_once_success_original_freeze_readback_and_registry_completion(self):
        original=entry.read(self.case/'session/SESSION.json')
        result=self.execute()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['frozen_accepted'])
        self.assertEqual((result['attempted_confirm'],result['completed_confirm']),(1,1))
        self.assertEqual(entry.read(self.case/'session/SESSION.json'),original)
        self.assertEqual(entry.read(self.out/'session/SESSION.json')['pool'],self.pool)
        row=json.loads((self.root/'registry.jsonl').read_text())
        self.assertEqual(set(row),entry.serial.REGISTRY_FIELDS)
        completion=entry.read(self.out/'completion.json')
        self.assertTrue(completion['required_artifacts_persisted']);self.assertTrue(completion['registry_persisted'])
        self.assertEqual(completion['status'],'ok');self.assertFalse((self.out/'session/REPORT_ACCESS.json').exists())

    def test_rejected_c2_normal_quality_endpoint_no_freeze(self):
        result=self.execute(runner=lambda *a,**k:self.finish(selected=False))
        self.assertEqual(result['status'],'quality_rejected');self.assertFalse(result['frozen'])
        self.assertEqual(result['completed_confirm'],1)
        self.assertFalse((self.out/'session/FROZEN.json').exists())

    def test_admission_failure_persists_zero_attempt_and_no_claim(self):
        (self.run/'completion.json').unlink()
        with mock.patch.object(entry,'run_process') as runner, mock.patch('pt_agent.registry.git_rev',return_value='synthetic'):
            result=entry.execute(self.args)
        runner.assert_not_called();self.assertFalse(result['admission_passed'])
        self.assertEqual(result['attempted_confirm'],0);self.assertEqual(result['status'],'error')
        self.assertTrue((self.out/'completion.json').is_file())
        self.assertFalse((self.root/'bert-budget-stop-c2-claims').exists())

    def test_original_process_inflight_refuses_without_signal_or_model_shell(self):
        result=self.execute(resources=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('original numeric worker remains')))
        self.assertEqual(result['status'],'error');self.assertEqual(result['attempted_confirm'],0)
        self.assertFalse((self.out/'model').exists())

    def test_public_scan_refuses_native_compiler_and_bert_worker_with_no_signal(self):
        table={123:dict(ppid=1,state='R',starttime=3),777:dict(ppid=1,state='R',starttime=42)}
        for command in (b'/tools/heir-opt\0source.mlir\0',b'python3\0-m\0pt_agent.experiments.bert_sst2_worker\0'):
            with self.subTest(command=command.split(b'\0')[0]),mock.patch.object(entry,'_scan',return_value=table), \
                 mock.patch.object(entry.os,'getpid',return_value=123),mock.patch.object(entry.Path,'read_bytes',return_value=command), \
                 mock.patch.object(entry.os,'kill') as kill:
                self.assertEqual(entry.original_inflight(),[dict(pid=777,starttime_ticks=42)])
                kill.assert_not_called()

    def test_existing_source_claim_refuses_second_branch(self):
        _,claim=entry._claim(self.args);claim.close()
        result=self.execute()
        self.assertEqual(result['status'],'error');self.assertEqual(result['attempted_confirm'],0)
        self.assertIn('FileExistsError',result['error'])

    def test_numeric_failure_claim_retained_and_no_retry(self):
        result=self.execute(runner=lambda *a,**k:clean(1))
        self.assertEqual(result['status'],'error');self.assertEqual(result['completed_confirm'],0)
        self.assertEqual(result['attempted_confirm'],1)
        self.assertTrue((self.root/'bert-budget-stop-c2-claims/source.json').is_file())

    def test_pool_mutation_is_error_even_when_fake_worker_freezes(self):
        result=self.execute(runner=lambda *a,**k:self.finish(mutate_pool=True))
        self.assertEqual(result['status'],'error');self.assertFalse(result['frozen_accepted'])
        self.assertIn('pool',result['error'])

    def test_readback_rejects_fork_model_metadata_or_foreign_reference(self):
        self.out.mkdir();entry.prepare_branch(self.args,entry.source_gate(self.args))
        self.assertTrue(entry.branch_bindings(self.args)['original_import_metadata_equal'])
        info=entry.read(self.out/'model/model_info.json');changed=copy.deepcopy(info);changed['threshold']={'acc_drop_pp':1}
        put(self.out/'model/model_info.json',changed)
        with self.assertRaisesRegex(ValueError,'metadata'):entry.branch_bindings(self.args)
        put(self.out/'model/model_info.json',info)
        graph=self.out/'model/graph';graph.unlink();graph.symlink_to(self.root/'foreign')
        with self.assertRaisesRegex(ValueError,'reference destination'):entry.branch_bindings(self.args)

    def test_terminal_resource_failure_revokes_acceptance_preserves_actual_completion(self):
        observations=[]
        def resources(*a,**k):
            observations.append(1)
            if len(observations)>=3:raise RuntimeError('late host reserve changed')
            return {}
        result=self.execute(resources=resources)
        self.assertEqual(result['status'],'error');self.assertTrue(result['frozen'])
        self.assertEqual(result['completed_confirm'],1);self.assertFalse(result['frozen_accepted'])
        self.assertEqual(entry.read(self.out/'completion.json')['status'],'error')

    def test_rss_and_available_guards_are_independent(self):
        self.out.mkdir()
        with mock.patch.object(entry,'artifact_bytes',return_value=0), \
             mock.patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.GIB)), \
             mock.patch.object(entry,'original_inflight',return_value=[]), \
             mock.patch.object(entry.serial,'foreign_heavy_processes',return_value=[]), \
             mock.patch.object(entry.Path,'read_text',return_value='MemAvailable: 1310719 kB\n'), \
             mock.patch.object(entry,'_scan',return_value={123:dict(ppid=0,vmrss_kib=1)}):
            with self.assertRaisesRegex(RuntimeError,'Host available'):entry.check_resources(self.args,own_pid=123)
        with mock.patch.object(entry,'artifact_bytes',return_value=0), \
             mock.patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.GIB)), \
             mock.patch.object(entry.Path,'read_text',return_value='MemAvailable: 16000000 kB\n'), \
             mock.patch.object(entry,'_scan',return_value={123:dict(ppid=0,vmrss_kib=15*1024**2)}):
            with self.assertRaisesRegex(RuntimeError,'RSS'):entry.check_resources(self.args,own_pid=123)

    def test_unclosed_original_last_observation_and_wrong_source_refuse(self):
        run=entry.read(self.run/'result.json');run['processes'][-1]['observation']=None;put(self.run/'result.json',run)
        with self.assertRaises(ValueError):entry.source_gate(self.args)
        run['processes'][-1]['observation']=clean();run['source']['source_case']=str(self.root/'foreign');put(self.run/'result.json',run)
        with self.assertRaises(ValueError):entry.source_gate(self.args)

    def test_main_keeps_host_visible_interpreter_symlink(self):
        link=self.root/'venv/bin/python';link.parent.mkdir(parents=True);link.symlink_to('/nonexistent-nix-store/python')
        with mock.patch.object(entry,'execute',return_value=dict(status='error')) as execute:
            entry.main(['--source-case',str(self.case),'--continuation-case',str(self.run),
                        '--boundary-snapshot',str(self.args.boundary_snapshot),'--out',str(self.out),
                        '--intake-python',str(link)])
        self.assertEqual(execute.call_args.args[0].intake_python,link)


if __name__ == '__main__': unittest.main()
