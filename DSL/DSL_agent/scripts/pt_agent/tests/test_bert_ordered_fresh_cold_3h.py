"""New three-hour adapter checks with synthetic observations and no payloads."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_ordered_fresh_cold_3h as entry
from pt_agent.experiments import bert_ordered_tuner_step_3h as step
from pt_agent.experiments.a_group_runs import append_record, empty_record


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def arguments(root):
    return SimpleNamespace(model=root/'model', inputs=root/'inputs',
        out=root/'new-cold', checkpoint_out=root/'checkpoint', pair_case=root/'pair',
        metadata_template=root/'template.json', ledger=root/'ledger.jsonl',
        allowance=root/'allowance.json', profile=root/'profile.json',
        frontend=root/'frontend.py', intake_python=Path('/fixture/intake-python'),
        data_python=Path('/fixture/data-python'), paper_runs=None)


class ThreeHourExecutionAdapterTests(unittest.TestCase):
    def test_original_three_hour_driver_and_numerical_workers_are_private_delegates(self):
        before = dict(vars(entry.original))
        values = entry.namespace()
        self.assertEqual(vars(entry.original), before)
        for name in ('worker', 'stop_search', 'state_summary', 'three_hour_observation',
                     'native_artifacts', '_numeric_args', '_budget', '_registry', 'main'):
            self.assertIs(values[name].__code__, getattr(entry.original, name).__code__)
            self.assertIs(values[name].__globals__, values)
        self.assertEqual((values['TOTAL_SECONDS'], values['SEARCH_CEILING'], values['TAIL_RESERVE']),
                         (10800, 9000, 1800))
        self.assertEqual(values['ROUND'], entry.ROUND)
        self.assertNotEqual(entry.ROUND, entry.original.ROUND)
        self.assertEqual(values['serial'].STATE_POLICY, entry.original.serial.STATE_POLICY)

    def test_closed_real_pair_gate_is_reused_without_claiming_new_cold_measurement(self):
        proof = dict(actual_full_1000_memory_and_two_hour_outcome_unmeasured=True,
                     pair_accepted=True, full_heldout_rows=1000, workers=3)
        with patch.object(entry.qualified, 'pair_qualification', return_value=proof) as gate:
            args = object()
            actual = entry.pair_qualification(args)
        gate.assert_called_once_with(args)
        self.assertTrue(actual['actual_full_1000_memory_and_three_hour_outcome_unmeasured'])
        self.assertNotIn('actual_full_1000_memory_and_two_hour_outcome_unmeasured', actual)
        self.assertEqual(actual['cold_clock_budget_seconds'], 10800)
        self.assertEqual(actual['full_heldout_rows'], 1000)

    def test_only_new_step_command_changes_and_retains_original10800_option(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root); args.out.mkdir()
            args._ordered_execution_admission = {'closed_pair_fixture': True}
            args._heavy_owner = dict(pid=1234567, starttime=222)
            def base(args, phase, work, seconds):
                return ['python', '-m', 'pt_agent.experiments.bert_sst2_recorded',
                        '--phase', phase, '--step-seconds', '10800']
            for phase in ('prepare', 'confirm', 'step', 'math', 'emit', 'trace', 'compile'):
                actual = entry.command(args, phase, root/'work', 500, base=base)
                if phase == 'step':
                    expected = base(args, phase, root/'work', 500)
                    expected[2] = entry.STEP_MODULE
                    expected += ['--ordered-qualification', str(args.out/'ordered-execution-admission.json')]
                    self.assertEqual(actual, expected)
                else:
                    self.assertEqual(actual, base(args, phase, root/'work', 500))
            self.assertTrue((args.out/'original-three-hour-driver.py').is_file())
            self.assertTrue((args.out/'ordered-three-hour-step-entry.py').is_file())
            with self.assertRaisesRegex(ValueError, 'three-hour step allowance'):
                entry.command(args, 'step', root/'work', 500,
                    base=lambda *args: ['python', '-m', 'pt_agent.experiments.bert_sst2_recorded',
                                        '--step-seconds', '7200'])

    def test_successful_step_still_requires_complete_original_tool_observation(self):
        values = entry.namespace()
        with patch.object(entry.qualified, 'read_worker', return_value={'fixture': True}) as read:
            actual = values['_read_worker']('args', 'step', 'work')
        self.assertEqual(actual, {'fixture': True})
        self.assertEqual(read.call_args.args, ('args', 'step', 'work'))
        self.assertIs(read.call_args.kwargs['base'].__code__, entry.original._read_worker.__code__)

    def test_real_persistence_endpoint_decides_separate_two_and_three_hour_claims(self):
        values = entry.namespace(); start = 100
        result = dict(compile_accepted=True, native_completed_ns=start+7990*10**9)
        complete = dict(required_artifacts_persisted=True, registry_persisted=True,
                        ended_ns=start+8000*10**9, complete_case_elapsed_seconds=8000)
        actual = values['three_hour_observation'](result, complete, start, start+8000*10**9)
        self.assertTrue(actual['persisted_pipeline_within_three_hours'])
        self.assertFalse(actual['persisted_pipeline_within_two_hours'])
        self.assertEqual(actual['round'], entry.ROUND)
        late = values['three_hour_observation'](result, complete, start, start+10801*10**9)
        self.assertFalse(late['persisted_pipeline_within_three_hours'])
        missing = dict(complete, registry_persisted=False)
        self.assertFalse(values['three_hour_observation'](result, missing, start,
            start+8000*10**9)['persisted_pipeline_within_three_hours'])

    def test_original_three_hour_failure_classifier_is_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root); args.out.mkdir()
            args.paper_runs = root/'canonical.jsonl'
            def base(args, result, record):
                append_record(args.paper_runs, empty_record(run_id='fixture', task='fixture', model='bert_tiny',
                    system='arranger', status='timeout', timeout_scope='inner_phase_allocation',
                    budget_s=10800, source=str(args.out/'result.json')))
            entry.paper_record(args, {}, {}, base=base)
            row = json.loads(args.paper_runs.read_text())
            original = json.loads((args.out/'original-classifier-paper-row.jsonl').read_text())
            self.assertEqual({k:row[k] for k in original}, original)
            self.assertEqual(row['timeout_scope'], 'inner_phase_allocation')
            self.assertEqual(row['budget_s'], 10800)

    def test_global_heavy_owner_prevents_nesting_or_numerical_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root)
            with patch.object(entry.paths, 'RESULTS_DIR', root), \
                    patch.object(entry, 'acquire_lease') as lease, \
                    patch.object(entry.original.serial, 'foreign_heavy_processes', return_value=['active']):
                with self.assertRaisesRegex(RuntimeError, 'remains active'):
                    entry.namespace()['execute'](args)
                lease.assert_called_once()
            self.assertFalse(args.out.exists())

    def test_complete_fake_cold_pipeline_keeps_c2_math_compile_and_single_new_clock(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); args=arguments(root); write(args.metadata_template, {})
            steps=0; calls=[]
            def run(argv, **kwargs):
                nonlocal steps
                kwargs['resources'].write_text('')
                if 'pt_agent.experiments.bert_checkpoint_audit' in argv:
                    phase='checkpoint'; target=args.checkpoint_out/'result.json'
                    value=dict(status='ok', checkpoint_semantic_coverage_passed=True)
                else:
                    work=Path(argv[argv.index('--work-dir')+1])
                    if '--worker-phase' in argv:
                        phase=argv[argv.index('--worker-phase')+1]
                        target=work/(phase+'-result.json'); value=dict(status='ok')
                        if phase=='math': value['math_passed']=True
                        if phase=='compile':
                            write(work/'runtime-plan.json', {'fixture_only':True})
                            value.update(runtime_plan=str(work/'runtime-plan.json'), compiled_bootstraps=12,
                                         prediction=None)
                    else:
                        requested=argv[argv.index('--phase')+1]
                        target=work/'worker_result.json'; value=dict(status='ok')
                        if requested=='prepare': phase='prepare'
                        else:
                            path=args.out/'session/SESSION.json'
                            state=json.loads(path.read_text()) if path.exists() else None
                            if state and state['phase']=='confirm':
                                phase='confirm'; state['phase']='done'; write(path, state)
                                selected=dict(iteration=1, origin='tuner')
                                write(args.out/'session/AGENT.json',dict(confirm=dict(selected=selected,
                                    evals=[dict(selected, passed=True, top1=93., exact_top1=93.,
                                                total=1000, agree=1000, acc_drop_pp=0.)])))
                            else:
                                phase='step'; steps+=1
                                state=dict(entry.original.serial.STATE_POLICY,
                                    phase='tune' if steps==1 else 'evaluate', iteration=1, calls=1,
                                    failed=None, pending={} if steps==1 else None,
                                    pool=[] if steps==1 else [{'fixture_candidate':True}],
                                    history=[{'fixture_only':True}], records=[{'fixture_only':True}])
                                write(path, state); self.assertIn(entry.STEP_MODULE, argv)
                                self.assertEqual(argv[argv.index('--step-seconds')+1], '10800')
                                write(work/'ordered-step-result.json',dict(schema='bert-ordered-tuner-step-result-v1',
                                    status='ok', timing=dict(attribution_ready=True), report_samples=0,
                                    confirmation_execution_changed=False, cold_chain_measurement=False,
                                    qualifies_two_hour_claim=False))
                write(target, value); calls.append(phase)
                return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])

            with patch.object(entry.paths, 'RESULTS_DIR', root), \
                    patch.object(entry.original, 'source_gate', return_value={'fresh_fixture':True}), \
                    patch.object(entry, 'pair_qualification', return_value={'qualified_fixture':True}), \
                    patch.object(entry.original.serial, 'foreign_heavy_processes', return_value=[]), \
                    patch.object(entry.original.serial, 'frozen_state', return_value={'frozen':True}), \
                    patch.object(entry.original.original, 'check_worker'), \
                    patch('pt_agent.autonomous_budget.call_records', return_value=[]), \
                    patch('pt_agent.registry.git_rev', return_value='fixture-no-git-read'):
                values=entry.namespace(); values['run_process']=run
                values['native_artifacts']=lambda compiled: {'fixture_only':True}
                before=time.perf_counter_ns(); actual=values['execute'](args); after=time.perf_counter_ns()
            self.assertEqual(actual['status'], 'ok')
            self.assertEqual(calls, ['prepare','checkpoint','step','step','confirm','math','emit','trace','compile'])
            self.assertEqual(actual['wall_policy']['tuner_budget_s'], 2400)
            self.assertEqual(actual['wall_policy']['max_iters'], 10)
            self.assertEqual(actual['budget_stop']['only_session_change'], 'phase:evaluate->confirm')
            record=json.loads((args.out/'three-hour-record.json').read_text())
            completion=json.loads((args.out/'completion.json').read_text())
            self.assertTrue(record['persisted_pipeline_within_three_hours'])
            self.assertTrue(before<=record['started_ns']<=completion['ended_ns']<=after)
            self.assertEqual(record['round'], entry.ROUND)
            registry=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry[0]['round'], entry.ROUND)
            self.assertEqual(registry[0]['metrics'], actual['metrics'])


class ThreeHourStepOptionTests(unittest.TestCase):
    def test_three_hour_option_is_consumed_before_original_worker_parser(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            args=['--delegate-worker','--step-seconds','10800','--metadata-template',str(root/'template.json'),
                  '--ordered-qualification',str(root/'proof.json'),'--phase','step','--work-dir',str(root/'work')]
            with patch.object(step, 'delegate', return_value=0) as delegate:
                self.assertEqual(step.main(args), 0)
            delegate.assert_called_once_with(['--phase','step','--work-dir',str(root/'work')],
                                             root/'template.json', root/'proof.json')
            for seconds in (None, '7200', '21600'):
                changed=list(args)
                if seconds is None: del changed[1:3]
                else: changed[2]=seconds
                with patch.object(step, 'delegate') as delegate, \
                        patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                    step.main(changed)
                delegate.assert_not_called()

    def test_original_numeric_delegate_receives10800_and_adapter_restores_after_success(self):
        original=step.recorded.worker_delegate
        with patch.object(step.recorded, 'worker_delegate', return_value=0) as numeric:
            def delegate(argv, template, proof):
                return step.recorded.worker_delegate(argv, template)
            with patch.object(step.original, 'delegate', side_effect=delegate):
                self.assertEqual(step.delegate(['--phase','step'], Path('template'), Path('proof')), 0)
            self.assertIs(step.recorded.worker_delegate, numeric)
            numeric.assert_called_once_with(['--phase','step'], Path('template'), step_seconds=10800)
        self.assertIs(step.recorded.worker_delegate, original)

    def test_original_numeric_delegate_restores_after_failure(self):
        original=step.recorded.worker_delegate
        with patch.object(step.original, 'delegate', side_effect=ValueError('fixture failure')):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                step.delegate([], Path('template'), Path('proof'))
        self.assertIs(step.recorded.worker_delegate, original)


if __name__=='__main__': unittest.main()
