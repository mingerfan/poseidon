"""New-clock delegation and actual-pair admission fixtures; no model payloads."""
import copy
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_ordered_fresh_cold_2h as entry
from pt_agent.experiments.a_group_runs import append_record, empty_record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    case = root/'pair';case.mkdir()
    write(case/'supervisor.json', dict(process=dict(pid=1234567, starttime=222)))
    source = dict(model=str(root/'model'), inputs=str(root/'inputs'),
                  metadata_template=str(root/'template.json'))
    args = SimpleNamespace(pair_case=case, out=root/'new-cold', checkpoint_out=root/'new-checkpoint',
        model=root/'model', inputs=root/'inputs', metadata_template=root/'template.json')
    result = dict(round=entry.pair.ROUND, mode=entry.pair.MODE, model=entry.original.MODEL,
        pair_accepted=True, attempted_pair=1, completed_pair=1, guard_trigger=None,
        incoming_signal=None, monitor_stopped=True, heavy_lease=dict(released=True),
        source=source, worker={'actual-worker':True}, resource_sampling=dict(
            complete_samples=True, sampled_rss_policy_passed=True,
            max_sampled_subtree_rss_gib=5.5, saved_numeric_births_closed=9))
    return args, result


class OrderedFreshColdTests(unittest.TestCase):
    def qualification(self, root, *, update=None, owner=None):
        args, result = fixture(root)
        if update:update(args, result)
        with patch.object(entry.paths, 'RESULTS_DIR', root), \
                patch.object(entry.pair, 'closed_case', return_value=(result, {})) as closed, \
                patch.object(entry.pair.reference, 'registry_row') as registry, \
                patch.object(entry.pair, 'check_worker') as worker, \
                patch.object(entry, '_process', return_value=owner), \
                patch.object(entry, 'producer_births', return_value=9):
            proof = entry.pair_qualification(args)
            closed.assert_called_once_with(args.pair_case, ('ok',))
            registry.assert_called_once_with(args.pair_case, result)
            worker.assert_called_once()
            self.assertEqual(worker.call_args.args[0].out, args.pair_case)
            self.assertIs(worker.call_args.args[1], result['worker'])
            self.assertIs(worker.call_args.args[2], result['source'])
        return args, proof

    def test_real_pair_sources_and_resource_proofs_are_delegated(self):
        with TemporaryDirectory() as folder:
            args, proof = self.qualification(Path(folder))
            self.assertEqual(proof['full_heldout_rows'], 1000)
            self.assertEqual(proof['qualification_rows'], 40)
            self.assertFalse(proof['paired_program_supplied_to_new_Agent'])
            self.assertFalse(proof['warm_session_or_candidate_reuse'])
            self.assertTrue(proof['actual_full_1000_memory_and_two_hour_outcome_unmeasured'])

    def test_live_actual_pair_owner_prevents_new_cold_case(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'owner must exit'):
                self.qualification(Path(folder), owner=dict(pid=1234567, starttime=222))

    def test_incomplete_resource_or_changed_source_prevents_admission(self):
        changes = [lambda a,r:r.update(pair_accepted=False),
                   lambda a,r:r['heavy_lease'].update(released=False),
                   lambda a,r:r['resource_sampling'].update(sampled_rss_policy_passed=False),
                   lambda a,r:r['resource_sampling'].update(max_sampled_subtree_rss_gib=14.1),
                   lambda a,r:r['source'].update(model='/different-model')]
        for update in changes:
            with TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):self.qualification(Path(folder), update=update)

    def test_private_driver_keeps_original_code_and_separate_round(self):
        before = dict(vars(entry.original));values = entry.namespace()
        self.assertEqual(vars(entry.original), before)
        self.assertEqual(values['ROUND'], entry.ROUND)
        self.assertEqual(values['MODE'], entry.MODE)
        self.assertEqual(values['MODULE'], entry.MODULE)
        self.assertNotEqual(values['ROUND'], entry.original.ROUND)
        self.assertIs(values['main'].__code__, entry.original.main.__code__)
        self.assertIs(values['two_hour_observation'].__code__, entry.original.two_hour_observation.__code__)
        self.assertIs(values['two_hour_observation'].__globals__, values)
        self.assertEqual(values['TOTAL_SECONDS'], 7200)
        self.assertEqual(values['SEARCH_CEILING'], 5400)
        self.assertEqual(values['TAIL_RESERVE'], 1800)

    def test_only_step_command_changes_and_new_sources_are_saved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder);args, _ = fixture(root);args.out.mkdir()
            args._ordered_execution_admission={'actual_admitted_fixture':True}
            args._heavy_owner=dict(pid=1234567, starttime=222)
            def base(args, phase, work, seconds):
                return ['python','-m','pt_agent.experiments.bert_sst2_recorded',
                        '--phase', phase, '--seconds', str(seconds)]
            for phase in ('prepare','confirm','step'):
                actual = entry.command(args, phase, root/'work', 500, base=base)
                if phase == 'step':
                    self.assertIn(entry.STEP_MODULE, actual)
                    self.assertIn('--ordered-qualification', actual)
                else:self.assertEqual(actual, base(args, phase, root/'work', 500))
            self.assertTrue((args.out/'original-two-hour-driver.py').is_file())
            self.assertEqual(json.loads((args.out/'ordered-execution-admission.json').read_text()),
                             args._ordered_execution_admission)

    def test_successful_new_step_requires_completed_attribution(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);work=root/'work';work.mkdir()
            good=dict(schema='bert-ordered-tuner-step-result-v1', status='ok',
                timing=dict(attribution_ready=True),report_samples=0, confirmation_execution_changed=False,
                cold_chain_measurement=False,qualifies_two_hour_claim=False)
            base=lambda *args:dict(status='ok', original_worker=True)
            with self.assertRaises(FileNotFoundError):entry.read_worker(None,'step',work,base=base)
            write(work/'ordered-step-result.json',good)
            actual=entry.read_worker(None,'step',work,base=base)
            self.assertTrue(actual['original_worker'])
            self.assertEqual(actual['ordered_tool_observation'],good)
            write(work/'ordered-step-result.json',dict(good,timing=dict(attribution_ready=False)))
            with self.assertRaises(ValueError):entry.read_worker(None,'step',work,base=base)

    def test_original_confirmation_and_failed_step_do_not_require_new_observer(self):
        with TemporaryDirectory() as folder:
            for phase, value in [('confirm',dict(status='ok')),('step',dict(status='error')),('step',None)]:
                self.assertEqual(entry.read_worker(None,phase,Path(folder),base=lambda *args:value),value)

    def test_new_two_hour_record_is_measured_without_warm_time(self):
        values=entry.namespace();start=100
        result=dict(compile_accepted=True,native_completed_ns=6000*10**9+start)
        completion=dict(required_artifacts_persisted=True,registry_persisted=True,
            ended_ns=start+7199*10**9,complete_case_elapsed_seconds=7199)
        actual=values['two_hour_observation'](result,completion,start,start+7199*10**9)
        self.assertTrue(actual['persisted_pipeline_within_two_hours'])
        self.assertEqual(actual['round'],entry.ROUND)
        self.assertEqual(actual['mode'],entry.MODE)
        late=values['two_hour_observation'](result,completion,start,start+7201*10**9)
        self.assertFalse(late['persisted_pipeline_within_two_hours'])

    def test_inner_timeout_is_scoped_and_original_classifier_row_is_preserved(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);out=root/'new-cold';out.mkdir()
            args=SimpleNamespace(out=out,paper_runs=root/'canonical.jsonl')
            def base(args,result,record):
                append_record(args.paper_runs,empty_record(run_id='new-cold',task='fixture',
                    model='bert_tiny',system='arranger',status='driver_error',source=str(out/'result.json')))
            observation=dict(exit_code=124,timed_out=False,signal=None,signals_sent=[],remaining_pids=[],
                             started_ns=100,ended_ns=100+5000*10**9)
            result=dict(processes=[dict(phase='step',argv=['python','-m','pt_agent.tools.nixrun',
                                  '--seconds','5000'],observation=observation)])
            entry.paper_record(args,result,dict(elapsed_seconds=5399),base=base)
            canonical=json.loads(args.paper_runs.read_text())
            original=json.loads((out/'original-classifier-paper-row.jsonl').read_text())
            self.assertEqual(original['status'],'driver_error')
            self.assertEqual(canonical['status'],'timeout')
            self.assertEqual(canonical['timeout_scope'],'inner_search_phase_allocation')
            self.assertFalse(canonical['complete_7200_seconds_consumed'])

    def test_complete_original_orchestration_uses_new_clock_and_persists_new_identity(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args,_=fixture(root)
            args.ledger=root/'fake-ledger.jsonl';args.allowance=root/'fake-allowance.json'
            args.profile=root/'profile.json';args.frontend=root/'frontend.py'
            args.intake_python=Path('/fixture/intake-python');args.data_python=Path('/fixture/data-python')
            args.paper_runs=None;write(args.metadata_template,{})
            proof=dict(actual_pair_fixture=True)
            calls=[];step_count=0

            def run(argv,**kwargs):
                nonlocal step_count
                kwargs['resources'].write_text('')
                if 'pt_agent.experiments.bert_checkpoint_audit' in argv:
                    phase='checkpoint';work=args.checkpoint_out;target=work/'result.json'
                    value=dict(status='ok',checkpoint_semantic_coverage_passed=True)
                else:
                    work=Path(argv[argv.index('--work-dir')+1])
                    if '--worker-phase' in argv:
                        phase=argv[argv.index('--worker-phase')+1]
                        target=work/(phase+'-result.json');value=dict(status='ok')
                        if phase=='math':value['math_passed']=True
                        if phase=='compile':
                            write(work/'runtime-plan.json',{'fixture_only':True})
                            value.update(runtime_plan=str(work/'runtime-plan.json'),compiled_bootstraps=12,prediction=None)
                    else:
                        requested=argv[argv.index('--phase')+1]
                        target=work/'worker_result.json';value=dict(status='ok')
                        if requested=='prepare':phase='prepare'
                        else:
                            current=json.loads((args.out/'session/SESSION.json').read_text())if(args.out/'session/SESSION.json').exists()else None
                            if current and current['phase']=='confirm':
                                phase='confirm';current['phase']='done'
                                write(args.out/'session/SESSION.json',current)
                                selected=dict(iteration=1,origin='tuner')
                                write(args.out/'session/AGENT.json',dict(confirm=dict(selected=selected,
                                    evals=[dict(selected,passed=True,top1=93.,exact_top1=93.,total=1000,agree=1000,acc_drop_pp=0.)])))
                            else:
                                phase='step';step_count+=1
                                state=dict(entry.original.serial.STATE_POLICY,phase='tune'if step_count==1 else'evaluate',
                                    iteration=1,calls=1,failed=None,pending={}if step_count==1 else None,
                                    pool=[]if step_count==1 else[{'fixture_candidate':True}],
                                    history=[{'fixture_only':True}],records=[{'fixture_only':True}])
                                write(args.out/'session/SESSION.json',state)
                                self.assertIn(entry.STEP_MODULE,argv)
                                write(work/'ordered-step-result.json',dict(schema='bert-ordered-tuner-step-result-v1',
                                    status='ok',timing=dict(attribution_ready=True),report_samples=0,
                                    confirmation_execution_changed=False,cold_chain_measurement=False,qualifies_two_hour_claim=False))
                write(target,value);calls.append(phase)
                return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])

            with patch.object(entry.paths,'RESULTS_DIR',root), \
                    patch.object(entry.original,'source_gate',return_value={'new_intake_fixture':True}), \
                    patch.object(entry,'pair_qualification',return_value=proof), \
                    patch.object(entry.original.serial,'foreign_heavy_processes',return_value=[]), \
                    patch.object(entry.original.serial,'frozen_state',return_value={'frozen':True}), \
                    patch.object(entry.original.original,'check_worker'), \
                    patch('pt_agent.autonomous_budget.call_records',return_value=[]), \
                    patch('pt_agent.registry.git_rev',return_value='fixture-no-git-read'):
                values=entry.namespace();values['run_process']=run
                values['native_artifacts']=lambda compiled:{'fixture_only':True}
                before=time.perf_counter_ns();actual=values['execute'](args);after=time.perf_counter_ns()
            self.assertEqual(actual['status'],'ok')
            self.assertEqual(calls,['prepare','checkpoint','step','step','confirm','math','emit','trace','compile'])
            self.assertEqual(actual['wall_policy']['tuner_budget_s'],2400)
            self.assertEqual(actual['wall_policy']['max_iters'],10)
            self.assertEqual(actual['budget_stop']['only_session_change'],'phase:evaluate->confirm')
            record=json.loads((args.out/'two-hour-record.json').read_text())
            completion=json.loads((args.out/'completion.json').read_text())
            registry=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()]
            self.assertTrue(record['persisted_pipeline_within_two_hours'])
            self.assertTrue(before<=record['started_ns']<=completion['ended_ns']<=after)
            self.assertEqual(record['started_ns'],completion['started_ns'])
            self.assertEqual(record['round'],entry.ROUND)
            self.assertEqual(len(registry),1)
            self.assertEqual(registry[0]['round'],entry.ROUND)
            self.assertEqual(registry[0]['mode'],entry.MODE)
            self.assertEqual(registry[0]['metrics'],actual['metrics'])


if __name__ == '__main__':unittest.main()
