"""A1 §6.14 local dominance, bounded convergence and persistent result reuse."""

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.planners import agent, transformer_tuner as tuner
from pt_agent.planners.transformer_session import load_state, save_state, migrate_tune_rule
from pt_agent.tests.test_a1_session_steps import Client, feedback, reply
from pt_agent.tests.test_a1_agent import SOURCE
from pt_agent.tests.test_a1_tuner import context
from pt_agent.tests.test_a1_tune_rule import measured
from pt_agent.tests.test_u4_squared_softmax import fixture


NEW = 'v8-6.14'


class DominanceTests(unittest.TestCase):
    def test_replay_comparison_uses_saved_dsl_reference_representation(self):
        import json
        from pt_agent.experiments.a1_transformer_agent import pruning_comparison
        from pt_agent.planners.tuner import TuneResult
        p=parse(fixture(2));attrs=next(op.attrs for op in p.ops if op.kind=='softmax')
        result=TuneResult(p,print_program(p),[dict(line=5,old=attrs,new=attrs)],
                          {},{},0,1,1.,[],True)
        record=asdict(result);record.pop('program')
        original=json.loads(json.dumps(record))
        self.assertTrue(all(pruning_comparison(record,original).values()))

    def test_local_counts_are_single_point_counts(self):
        from pt_agent.backends import CountBackend
        from pt_agent.kernels import transformer as tr
        attrs=next(op.attrs for op in parse(fixture(2)).ops if op.kind=='softmax')
        for role in ('exp','inv','step_inv'):
            b=CountBackend(8)
            value=tr.apply(b,b.input(0),tr.approximation(attrs,role,'exp' if role=='exp' else 'inv'))
            self.assertEqual(tuner.local_approximation_cost(attrs,role),
                             dict(depth=value.depth,counts=b.stats()['counts']))

    def test_all_categories_and_equality_are_required(self):
        current=dict(depth=3,counts=dict(add=2,sub=1,neg=0,mul_cc=4,mul_cp=3,rot=0,boot=0))
        self.assertTrue(tuner.dominated_approximation(current,copy.deepcopy(current)))
        for key in current['counts']:
            candidate=copy.deepcopy(current);candidate['counts'][key]-=1
            self.assertFalse(tuner.dominated_approximation(current,candidate))
        candidate=copy.deepcopy(current);candidate['depth']-=1
        self.assertFalse(tuner.dominated_approximation(current,candidate))

    def simulate(self,kind='approximation',*,dominated=True,change=False,sweeps=8,clock=None):
        p=parse(fixture(2));ctx=context();ctx.prompt_version='r8-v8'
        calls=[]
        def cost(program,*args,**kwargs):
            op=next(op for op in program.ops if op.kind=='softmax')
            calls.append(op.attrs['exp_deg'])
            return dict(latency_pred_us=op.attrs['exp_deg'],boots_pred=1)
        def moves(program,*args):
            op=next(op for op in program.ops if op.kind=='softmax')
            if op.attrs['exp_deg']<=7:return
            attrs=dict(op.attrs,exp_deg=op.attrs['exp_deg']-1 if change else 7)
            yield dict(kind=kind,role='exp'),attrs
        with tuner.tune_rule_scope(NEW), \
             patch('pt_agent.dsl.interface.require_interface'), \
             patch('pt_agent.planners.approximation_failures.program_failures',return_value=[]), \
             patch.object(tuner,'cached_estimate',side_effect=cost), \
             patch.object(tuner,'validate',return_value=measured(True)), \
             patch.object(tuner,'exact_check',return_value=dict(ok=True)), \
             patch.object(tuner,'softmax_moves',side_effect=moves), \
             patch.object(tuner,'dominated_approximation',return_value=dominated) as dominance, \
             patch.object(tuner.time,'monotonic',side_effect=clock or (lambda:0.)):
            result=tuner.tune(p,ctx,max_sweeps=sweeps,engine_profile='tensor')
        return result,calls,dominance.call_count

    def test_dominated_skips_full_estimation_and_records_both_costs(self):
        result,calls,checks=self.simulate()
        self.assertEqual(len(calls),2)  # Original and domain alignment only.
        row=next(row for row in result.history if row['phase']=='softmax')
        self.assertEqual(row['rejected'],'dominated')
        self.assertEqual(set(row['local_current']),{'depth','counts'})
        self.assertEqual(set(row['local_candidate']),{'depth','counts'})
        self.assertEqual(checks,1)
        self.assertEqual((result.sweeps,result.converged),(1,True))

    def test_structure_bypasses_dominance_and_accepts_both_validation_levels(self):
        for kind in ('steps','inv_budget'):
            with self.subTest(kind=kind):
                result,calls,checks=self.simulate(kind)
                self.assertEqual(checks,0)
                self.assertEqual(len(calls),3)
                self.assertEqual(len(result.changes),1)
                self.assertEqual((result.coarse_evals,result.full_evals),(1,3))
                self.assertEqual((result.sweeps,result.converged),(2,True))

    def test_nondominated_continues_then_stops_after_empty_sweep(self):
        result,calls,_=self.simulate(dominated=False)
        self.assertEqual(calls,[15,15,7])
        self.assertEqual((result.sweeps,result.converged),(2,True))
        self.assertEqual(result.history[-1]['sweeps'],2)
        self.assertTrue(result.history[-1]['converged'])

    def test_sweep_cap_and_time_budget_are_not_convergence(self):
        result,_,_=self.simulate(dominated=False,change=True,sweeps=8)
        self.assertEqual((result.sweeps,result.converged),(8,False))
        times=iter([0.,0.,0.,2401.,2401.,2401.,2401.,2401.,2401.])
        result,_,_=self.simulate(clock=lambda:next(times))
        self.assertFalse(result.converged)
        self.assertTrue(any(row.get('budget_exhausted') for row in result.history))


class ConvergentCacheTests(unittest.TestCase):
    def test_qwen_first_tune_peak_stop_preserves_old_and_later_limits(self):
        from pt_agent.experiments.a1_transformer_agent import tuning_peak_error,first_convergent_scan
        self.assertTrue(first_convergent_scan(dict(history=[])))
        self.assertTrue(first_convergent_scan(dict(history=[dict(tuned=dict(sweeps=0))])))
        self.assertFalse(first_convergent_scan(dict(history=[dict(tuned=dict(sweeps=1))])))
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'first.json'
            self.assertTrue(first_convergent_scan(dict(history=[]),marker))
            marker.write_text('{}')
            self.assertFalse(first_convergent_scan(dict(history=[]),marker))
        limit=int(3.8*(1<<30))
        report=dict(first_convergent_tune=True,tune_rule=NEW,peak_rss_bytes=limit)
        self.assertIsNone(tuning_peak_error(report))
        report['peak_rss_bytes']=limit+1
        self.assertIsNotNone(tuning_peak_error(report))
        self.assertIsNone(tuning_peak_error(dict(report,first_convergent_tune=False)))
        self.assertIsNone(tuning_peak_error(dict(report,tune_rule='v8-6.11')))

    def run_case(self,*,converged=True,normalized=False,once=False):
        output=SOURCE.replace('deg=15','deg=7')
        second=output.replace('a =','a=')+'\n' if normalized else output
        ctx=SimpleNamespace(prompt_version='r8-v8')
        client=Client([reply(SOURCE),reply(second)])
        tune_calls=[]
        def tuning(program,*args,**kwargs):
            tune_calls.append(kwargs['max_sweeps'])
            cost=dict(latency_pred_us=5,boots_pred=1)
            return tuner.ConvergentTuneResult(parse(output),print_program(parse(output)),[],cost,cost,
                1,2,3.,[feedback()['validation']],True,sweeps=2,converged=converged)
        def confirm(pool,*args,**kwargs):
            return dict(selected=pool[0],evals=[],deduplicated=dict(before=len(pool),after=len(pool)),duplicates=[])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(agent,'build_system_prompt',return_value='system'), \
             patch.object(agent,'initial_message',return_value='initial'), \
             patch.object(agent,'evaluate',side_effect=[feedback(20),feedback(10)]), \
             patch.object(agent,'tune',side_effect=tuning), \
             patch.object(agent,'confirm_select',side_effect=confirm):
            trace=Path(tmp)/'trace.jsonl'
            if once:
                result=agent._run_agent_once(ctx,client,max_iters=2,trace_path=trace)
                state=None
            else:
                state=agent.new_agent_state(ctx,max_iters=2)
                while state['phase']!='done':
                    agent.agent_step(ctx,state,trace_path=trace,client=client)
                    save_state(Path(tmp)/'SESSION.json',state)
                    state=load_state(Path(tmp)/'SESSION.json')
                result=agent.agent_result(state)
        return result,state,tune_calls

    def test_converged_output_is_reused_in_both_orchestrations(self):
        for once in (False,True):
            with self.subTest(once=once):
                result,_,calls=self.run_case(once=once)
                self.assertEqual(calls,[8])
                self.assertEqual(result['history'][1]['tuned_reused_from'],1)
                self.assertEqual(result['tune_rule'],NEW)
                self.assertTrue(result['history'][1]['tuned']['converged'])

    def test_unconverged_output_is_not_registered(self):
        for once in (False,True):
            with self.subTest(once=once):
                result,_,calls=self.run_case(converged=False,once=once)
                self.assertEqual(calls,[8,8])
                self.assertIsNone(result['history'][1]['tuned_reused_from'])

    def test_normalized_equivalent_output_is_reused(self):
        for once in (False,True):
            with self.subTest(once=once):
                result,_,calls=self.run_case(normalized=True,once=once)
                self.assertEqual(calls,[8])
                self.assertEqual(result['history'][1]['tuned_reused_from'],1)

    def test_new_default_is_scoped_to_v8(self):
        self.assertEqual(tuner.validate_tune_rule('r8-v8',None),NEW)
        for version in ('r8-v4','r8-v5','r8-v6','r8-v7'):
            self.assertEqual(tuner.validate_tune_rule(version,None),'v8-6.11')
            with self.assertRaises(ValueError):tuner.validate_tune_rule(version,NEW)

    def test_boundary_migration_preserves_history_counts_pool_and_archives_cache(self):
        state=dict(prompt_version='r8-v8',tune_rule='v8-6.11',phase='evaluate',pending=None,
            calls=4,iteration=5,llm_min=100,has_tuned=True,stale=1,
            messages=['history'],history=['old'],pool=['candidates'],records={(1,'llm','source'):3},
            tune_cache={'original':('saved',4,'source')},best={'latency_pred_us':90},first_feasible=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'SESSION.json';save_state(path,state)
            record=migrate_tune_rule(root,next_model_call=33,target_rule=NEW)
            result=load_state(path)
            for key in state.keys()-{'tune_rule','tune_cache'}:self.assertEqual(result[key],state[key])
            self.assertEqual(result['tune_cache'],{})
            self.assertEqual(result['tune_cache_archive']['v8-6.11'],state['tune_cache'])
            self.assertEqual(result['tune_rule'],NEW)
            self.assertEqual(record['next_model_call'],33)
            self.assertEqual(migrate_tune_rule(root,next_model_call=33,target_rule=NEW),record)
            with self.assertRaises(ValueError):migrate_tune_rule(root,next_model_call=34,target_rule=NEW)


if __name__=='__main__':unittest.main()
