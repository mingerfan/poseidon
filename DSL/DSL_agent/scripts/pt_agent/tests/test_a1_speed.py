"""A1 §6.10 observer and finite-cohort contracts; no model execution."""

import gzip
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments.a1_speed import (
    CandidateLimitReached, CandidateMemoryExceeded, CandidateObserver,
    EvaluationClock, compare_dag, compare_tune, dag_records, first_difference,
    inventory, observed_validation, run_cases, tune_phase, write,
)
from pt_agent.paths import RESULTS_DIR
from pt_agent.tests._artifacts import require


class SpeedObserverTests(unittest.TestCase):
    def test_direct_estimate_retains_interface_diagnostic_without_gating_cost(self):
        from pt_agent.experiments import a1_speed
        for direct in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);source=root/'program.ptdsl';source.write_text('fixture')
                row=dict(group='v7', source=str(source), normalized=True)
                graph=SimpleNamespace(weights={})
                with patch.object(a1_speed, 'graph_for', return_value=(graph, object())), \
                     patch('pt_agent.dsl.parser.parse', return_value=object()), \
                     patch('pt_agent.dsl.interface.check_interface', return_value=dict(ok=False)), \
                     patch('pt_agent.dsl.typecheck.check', return_value=SimpleNamespace(ok=True)), \
                     patch('pt_agent.cost.estimate', return_value=dict(boots_pred=2)) as estimator, \
                     patch.object(a1_speed, 'cost_observer', return_value=nullcontext(dict(observer_seconds=0.))):
                    result=a1_speed.measure(row,root,direct_estimate=direct)
                self.assertEqual(result['stage'],'complete' if direct else 'interface')
                self.assertEqual(estimator.call_count, int(direct))
                if direct:
                    self.assertEqual(result['interface_diagnostic'],dict(ok=False))
                    self.assertEqual(result['cost'],dict(boots_pred=2))

    def test_budget_counts_cost_rejections_as_evaluations(self):
        clock = EvaluationClock(2400, evaluations=3)
        self.assertEqual(clock.monotonic(), 0)
        costs = clock.cost(lambda candidate: {'latency_pred_us':candidate})
        self.assertEqual(costs(100), {'latency_pred_us':100})
        self.assertEqual(costs(200), {'latency_pred_us':200})
        self.assertEqual(clock.monotonic(), 0)
        self.assertEqual(costs(300), {'latency_pred_us':300})
        self.assertEqual(clock.monotonic(), 2401)
        self.assertEqual(clock.calls, 3)

    def test_dag_records_preserve_exact_node_numbers_and_constant_names(self):
        backend = SimpleNamespace(_inputs=[SimpleNamespace(nid=0)],
            _outputs=[SimpleNamespace(nid=1)], _constants={b'a':SimpleNamespace(name='k0')},
            kind=['input','mul_cp'], args=[(),(0,)], prog_op=[None,7],
            _depth=[0,1], _total_depth=[0,1], operands=[(),(('cipher',0),('plain','k0'))],
            steps=[None,None])
        rows = list(dag_records(backend))
        self.assertEqual(rows[0], dict(record='boundary', inputs=[0], outputs=[1], constants=['k0']))
        self.assertEqual(rows[2]['id'], 1)
        self.assertEqual(rows[2]['args'], (0,))
        self.assertEqual(rows[2]['operands'], (('cipher',0),('plain','k0')))

    def test_streaming_dag_comparison_reports_first_changed_operand(self):
        with tempfile.TemporaryDirectory() as temporary:
            a,b = (Path(temporary)/name for name in ('a.gz','b.gz'))
            for path, operand in ((a,3),(b,4)):
                with gzip.open(path,'wt') as stream:
                    for row in ({'id':0,'args':[]},{'id':1,'args':[operand]}):
                        stream.write(json.dumps(row)+'\n')
            result = compare_dag(a,b)
            self.assertFalse(result['equal'])
            self.assertEqual(result['difference']['row'], 2)
            self.assertEqual(result['difference']['path'], '$.args[0]')

    def test_complete_nested_cost_difference_including_order(self):
        self.assertIsNone(first_difference({'cuts':[1,2],'counts':{'rot':3}},
                                           {'counts':{'rot':3},'cuts':[1,2]}))
        self.assertEqual(first_difference({'cuts':[1,2]}, {'cuts':[2,1]})['path'], '$.cuts[0]')

    def test_time_budget_comparison_records_changes_and_count_budget_rejects_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for side,history in (('before',[1]),('after',[1,2])):
                (directory/side).mkdir()
                write(directory/side/'REPORT.json',dict(complete=True,tune=dict(history=history)))
            self.assertTrue(compare_tune(directory,None)['complete'])
            compared=compare_tune(directory,4)
            self.assertFalse(compared['complete'])
            self.assertEqual(compared['first_difference']['path'], '$.history')

    def test_ten_candidates_exclude_start_and_alignment_and_keep_validation_time(self):
        observed = CandidateObserver(EvaluationClock(2400), limit=10)
        small_memory = dict(VmRSS_bytes=100, VmHWM_bytes=200, peak_rss_bytes=200)
        with patch('pt_agent.experiments.a1_speed.memory',return_value=small_memory), \
             patch('pt_agent.experiments.a1_speed.tuner_state',return_value={'history':['retained']}):
            for phase in ('initial','alignment'):
                observed.begin_cost(phase,object(),None)
                observed.component('cost',phase,2)
            self.assertEqual(len(observed.rows),0)
            for _ in range(10):
                observed.begin_cost('candidate',object(),None)
                # This represents a cached call too: counting starts before
                # the original cache handler, without requiring an estimator miss.
                observed.component('cost','candidate',1)
                observed.component('validation','candidate',.25,engine='tensor',coarse=True)
            with self.assertRaises(CandidateLimitReached) as stopped:
                observed.begin_cost('candidate',object(),None)
            self.assertEqual(stopped.exception.state,{'history':['retained']})
            self.assertEqual(len(observed.rows),10)
            self.assertTrue(all(row['cost_calls']==1 and row['validation_calls']==1
                                and row['wall_seconds']==1.25 for row in observed.rows))

    def test_qwen_peak_stop_uses_historical_single_process_peak(self):
        observed=CandidateObserver(EvaluationClock(2400),memory_limit=3800)
        with patch('pt_agent.experiments.a1_speed.memory',return_value=dict(
                VmRSS_bytes=500,VmHWM_bytes=3801,peak_rss_bytes=3801)):
            with self.assertRaises(CandidateMemoryExceeded):
                observed.begin_cost('candidate',object(),None)

    def test_tuner_phase_uses_calls_not_misses_or_program_names(self):
        namespace=dict(__name__='pt_agent.planners.transformer_tuner',phase=tune_phase)
        exec('def tune():\n'
             ' current=object()\n'
             ' first=phase(current)[0]\n'
             ' aligned=object()\n'
             ' second=phase(aligned)[0]\n'
             ' candidate=object()\n'
             ' third=phase(candidate)[0]\n'
             ' def finish(): return phase(candidate)[0]\n'
             ' return first,second,third,finish()\n',namespace)
        self.assertEqual(namespace['tune'](),('initial','alignment','candidate','finish'))

    def test_fragment_comparison_checks_exactly_ten_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for side in ('before','after'):
                (directory/side).mkdir()
                write(directory/side/'REPORT.json',dict(complete=True,tune=dict(history=[1]),
                      candidate_evaluations=[{}]*9))
            compared=compare_tune(directory,None,10)
            self.assertFalse(compared['complete'])
            self.assertEqual(compared['candidate_counts'],(9,9))

    def test_alignment_resume_preserves_worker_and_merges_existing_case(self):
        for has_comparison in (False, True):
            with self.subTest(saved_comparison=has_comparison), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                row=dict(id='case-003',label='qwen-R1',group='baselines',model='qwen')
                manifest=root/'MANIFEST.json';write(manifest,dict(cases=[row]))
                out=root/'run';worker=out/'case-003/after';worker.mkdir(parents=True)
                retained=dict(complete=True,case=row,stage='alignment',diagnostics=[dict(samples=[])])
                worker_report=worker/'REPORT.json';write(worker_report,retained)
                text=worker_report.read_text()
                if has_comparison:
                    write(out/'case-003/COMPARISON.json',retained)
                args=SimpleNamespace(manifest=manifest,case='case-003',group=None,
                    candidate_evaluations=None,evaluations=None,part='alignment',out=out,
                    resume=True,after=root/'source',before=None,direct_estimate=False)
                with patch('pt_agent.experiments.a1_speed.run_worker') as launch:
                    result=run_cases(args)
                launch.assert_not_called()
                self.assertTrue(result['complete'])
                self.assertEqual(result['finished'],1)
                self.assertEqual(result['cases'][0]['case'],row)
                self.assertEqual(result['cases'][0]['diagnostics'],retained['diagnostics'])
                self.assertEqual(worker_report.read_text(),text)
                self.assertEqual(json.loads((out/'REPORT.json').read_text()),result)

    def test_validation_observer_preserves_result_and_times_only_validation(self):
        from pt_agent.domain_criteria import record_violations
        expected=dict(feasible=False,analytic_domain_violation=True)
        chains=dict(samples=[dict(sample_index=0,chain=[dict(role='inv')])])
        records=[];components=[]
        observer=SimpleNamespace(check_memory=lambda:None,
            record_validation=lambda *args,**kwargs:records.append((args,kwargs)),
            component=lambda *args,**kwargs:components.append((args,kwargs)))
        def validate(program,context,**kwargs):
            record_violations(chains)
            return expected
        clock=iter((10.,12.))
        with patch('pt_agent.experiments.a1_speed.time',
                   SimpleNamespace(monotonic=lambda:next(clock))):
            actual=observed_validation(validate,observer,object(),'finish',object(),engine='mock')
        self.assertIs(actual,expected)
        self.assertIs(records[0][0][2],expected)
        self.assertEqual(records[0][0][4],[chains])
        self.assertEqual(records[0][1]['seconds'],2.)
        self.assertEqual(components[0][0],('validation','finish',2.))
        self.assertEqual(expected,dict(feasible=False,analytic_domain_violation=True))

    def test_validation_observer_preserves_exception_and_retains_partial_chain(self):
        from pt_agent.domain_criteria import record_violations
        problem=RuntimeError('retained numerical diagnostic')
        chains=dict(samples=[dict(sample_index=2,chain=[dict(role='exp')])])
        records=[]
        observer=SimpleNamespace(check_memory=lambda:None,
            record_validation=lambda *args,**kwargs:records.append((args,kwargs)),
            component=lambda *args,**kwargs:None)
        def validate(program,context,n,**kwargs):
            record_violations(chains)
            raise problem
        with self.assertRaises(RuntimeError) as raised:
            observed_validation(validate,observer,object(),'candidate',object(),20,engine='tensor')
        self.assertIs(raised.exception,problem)
        self.assertEqual(records[0][0][3],dict(type='RuntimeError',message=str(problem)))
        self.assertEqual(records[0][0][4],[chains])
        self.assertTrue(records[0][1]['coarse'])

    def test_validation_record_writes_program_and_chain_before_later_stop(self):
        fake_modules={
            'pt_agent.dsl.printer':SimpleNamespace(print_program=lambda program:'retained program\n'),
            'pt_agent.planners.transformer_session':SimpleNamespace(json_safe=lambda value:value),
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict('sys.modules',fake_modules):
            directory=Path(temporary)
            observer=CandidateObserver(EvaluationClock(2400),directory=directory)
            observer.active=dict(candidate=3)
            chains=[dict(samples=[dict(sample_index=0,first_violation=dict(role='inv'),
                                        chain=[dict(role='inv',max_u=1.2)])])]
            observer.record_validation(object(),'candidate',dict(feasible=False,analytic_domain_violation=True),
                None,chains,seconds=.25,engine='tensor',coarse=True)
            record=json.loads((directory/'validations/0001-candidate-003/REPORT.json').read_text())
            self.assertEqual(record['diagnostics'],chains)
            self.assertEqual(record['candidate'],3)
            self.assertEqual(Path(record['program']).read_text(),'retained program\n')
            progress=json.loads((directory/'EVALUATIONS.json').read_text())
            self.assertEqual(progress['last_validation']['diagnostics'],chains)
            self.assertEqual(progress['validations'][0]['seconds'],.25)
            self.assertGreaterEqual(observer.serialization_seconds,0)


@require(RESULTS_DIR/'A1-bert-replay-20261027T160313-a92e/REPLAY.json',
         RESULTS_DIR/'A1-transformers-s68-stop-20261028T060310-e9da/REPORT.json',
         RESULTS_DIR/'u4-cost78-inventory-20261026T155556805705/MANIFEST.json')
class SpeedInventoryTests(unittest.TestCase):
    def test_retained_cohorts_and_no_report_access(self):
        manifest=inventory(RESULTS_DIR)
        self.assertEqual(manifest['counts'],dict(baselines=4,v7=26,v8=14,**{'v8-tuned':4},M=20))
        self.assertEqual(len(manifest['cases']),68)
        self.assertEqual(len({row['id'] for row in manifest['cases']}),68)
        self.assertEqual(manifest['llm_calls'],0)
        self.assertEqual(manifest['report_images'],0)
        self.assertEqual(manifest['compile_calls'],0)
        labels={row['label'] for row in manifest['cases']}
        self.assertTrue({'M-resnet20','M-cnn_plain','v8-call-014','v8-tuned-013'}<=labels)


if __name__=='__main__':
    unittest.main()
