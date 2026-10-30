"""A1 6.17 cache lifetime, tuning-only precision and saved-response recovery."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import cost, tensor_sim
from pt_agent.backends.dag import DagBackend
from pt_agent.dsl.parser import parse
from pt_agent.planners import transformer_tuner as tuner
from pt_agent.planners.transformer_session import save_state,load_state,recover_pending_tune
from pt_agent.tests.test_a1_tune_rule import SOURCE,measured


class EstimateReleaseTests(unittest.TestCase):
    def test_tensor_and_mock_batch_policies_are_separate(self):
        from pt_agent.planners import agent_tools
        from pt_agent import transformer_validation as validation
        from pt_agent.tests.test_a1_agent import fixture, SOURCE as source
        info,calibration,samples=fixture()
        ctx=agent_tools.build_context(info,calibration,samples=samples)
        ctx.prompt_version='r8-v7'
        ctx.val_images=np.zeros((20,1,2))
        ctx.val_inputs=dict(x=ctx.val_images)
        ctx.val_exact_logits=np.tile([1.,0.],(20,1))
        for engine,expected in (('mock',[10,10]),('tensor',[20])):
            sizes=[]
            def execute(p,weights,inputs,**kwargs):
                count=len(inputs['x']);sizes.append(count)
                return np.tile([1.,0.],(count,1))
            with patch.object(validation,'execute',side_effect=execute):
                result=validation.validate(parse(source),ctx,engine=engine)
            self.assertEqual(sizes,expected)
            self.assertEqual(result['agree'],20)

    def test_releases_payloads_and_keeps_cost_result(self):
        backend=DagBackend(8)
        vector=np.arange(8,dtype=np.float64);vector.flags.writeable=False
        backend.const(vector)
        cost._estimate_backends.add(backend)
        before=copy.deepcopy(tensor_sim._estimates)
        sentinel=dict(boots_pred=17,boot_nodes=[1,3])
        tensor_sim._estimates['6.17-test']=sentinel
        try:
            cost.release_estimate_caches()
            self.assertEqual(backend._constant_objects,{})
            self.assertEqual(backend._constants,{})
            self.assertIs(tensor_sim._estimates['6.17-test'],sentinel)
        finally:
            tensor_sim._estimates.clear();tensor_sim._estimates.update(before)

    def test_every_token_mock_execution_releases_before_backend(self):
        from pt_agent import transformer_validation as validation
        p=parse(SOURCE);events=[]
        with patch.object(cost,'release_estimate_caches',side_effect=lambda:events.append('release')), \
             patch.object(validation,'pack_inputs',side_effect=RuntimeError('after-release')):
            with self.assertRaisesRegex(RuntimeError,'after-release'):
                validation.execute(p,{},dict(x=np.ones((1,1,2))),engine='mock')
        self.assertEqual(events,['release'])


class MoveAccuracyTests(unittest.TestCase):
    def ctx(self,task='classification',threshold=None):
        return SimpleNamespace(model_info=SimpleNamespace(task=task,threshold=threshold or {}))

    def test_classification_margin_and_stricter_task(self):
        ctx=self.ctx()
        self.assertTrue(tuner.move_accuracy_ok(dict(logit_rel_err=.0099),ctx,'v8-6.17'))
        self.assertFalse(tuner.move_accuracy_ok(dict(logit_rel_err=.0101),ctx,'v8-6.17'))
        self.assertFalse(tuner.move_accuracy_ok(dict(logit_rel_err=.0099),
            self.ctx(threshold=dict(logit_rel_err=.005)),'v8-6.17'))
        for value in (None,float('nan'),float('inf')):
            self.assertFalse(tuner.move_accuracy_ok(dict(logit_rel_err=value),ctx,'v8-6.17'))

    def test_causal_lm_acceptance_matches_old_rule(self):
        from pt_agent.metrics import passes
        for limit in (.01,.005):
            ctx=self.ctx('causal_lm',dict(agreement_pct=99.,logit_rel_err=limit))
            for error in (0.,.0049,.0051,.0099,.01,.0101,None,float('inf')):
                for agreement in (98.,100.):
                    result=dict(logit_rel_err=error,agreement_pct=agreement)
                    original=passes(result,ctx.model_info.threshold)
                    self.assertEqual(original and tuner.move_accuracy_ok(result,ctx,'v8-6.14'),
                                     original and tuner.move_accuracy_ok(result,ctx,'v8-6.17'))

    def test_old_rules_acceptance_unchanged(self):
        for rule in ('v8-pre-6.11','v8-6.11','v8-6.14'):
            self.assertTrue(tuner.move_accuracy_ok({},self.ctx(),rule))

    def test_new_rule_reuses_convergent_search_without_changing_default(self):
        self.assertTrue(tuner.convergent_rule('v8-6.17'))
        self.assertEqual(tuner.validate_tune_rule('r8-v9','v8-6.17'),'v8-6.17')
        self.assertEqual(tuner.validate_tune_rule('r8-v8',None),'v8-6.14')
        for version in ('r8-v4','r8-v5','r8-v6','r8-v7'):
            with self.assertRaises(ValueError):tuner.validate_tune_rule(version,'v8-6.17')

    def test_alignment_and_scan_margin_do_not_change_start_or_finish(self):
        ctx=SimpleNamespace(prompt_version='r8-v8',weights={},profile='unused',
            model_info=SimpleNamespace(task='classification',threshold=dict(agreement_pct=99.)),
            val_exact_logits=np.array([[1.,2.]]),
            calibration={key:dict(min=-.4,max=.4) for key in ('first','second')})
        calls=[]
        def validate(p,*args,**kwargs):
            calls.append(kwargs['engine'])
            # Start remains feasible even with a large error; alignment fails
            # only the additional move condition, then scan candidates pass it.
            error=.05 if len(calls)==1 else .0101 if len(calls)==2 else .0099
            kwargs['output_callback'](ctx.val_exact_logits+2*error)
            return measured(True)
        def estimate(p,*args,**kwargs):
            return dict(latency_pred_us=sum(op.attrs.get('deg',0) for op in p.ops),boots_pred=1)
        with tuner.tune_rule_scope('v8-6.17'), \
             patch('pt_agent.dsl.interface.require_interface'), \
             patch.object(tuner,'point_names',return_value=['first','second']), \
             patch.object(tuner,'cached_estimate',side_effect=estimate), \
             patch.object(tuner,'validate',side_effect=validate), \
             patch.object(tuner,'exact_check',return_value=dict(ok=True)):
            result=tuner.tune(parse(SOURCE),ctx,max_sweeps=1)
        self.assertFalse(result.history[0]['accepted'])
        self.assertEqual(len(result.changes),2)
        self.assertTrue(result.feasible)
        self.assertEqual(calls[-1],'mock')


class PendingRecoveryTests(unittest.TestCase):
    def test_saved_pending_response_preserves_consumption_and_history(self):
        response=dict(content=json.dumps(dict(program=SOURCE)),usage=dict(prompt_tokens=7))
        state=dict(phase='tune',prompt_version='r8-v9',tune_rule='v8-6.14',calls=2,iteration=2,
                   messages=['m'],history=['h'],pool=['p'],records={(1,'llm'):2},
                   tune_cache={'old':1},pending=dict(reply=response,source=SOURCE))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir();(old/'sources').mkdir()
            (old/'sources/iteration-2.ptdsl').write_text(SOURCE)
            (old/'agent_trace.jsonl').write_text('original\n')
            save_state(old/'SESSION.json',state)
            saved=root/'response.json';saved.write_text(json.dumps(response))
            result=recover_pending_tune(old,root/'new',saved)
            restored=load_state(root/'new/SESSION.json')
            for key in ('calls','iteration','messages','history','pool','records','pending'):
                self.assertEqual(restored[key],state[key])
            self.assertEqual(restored['tune_rule'],'v8-6.17')
            self.assertEqual(restored['tune_cache'],{})
            self.assertEqual(result['llm_calls'],0)
            self.assertEqual(load_state(old/'SESSION.json'),state)

    def test_qwen_tune_peak_guard_includes_early_return(self):
        from pt_agent.experiments.a1_transformer_agent import tuning_peak_error
        record=dict(qwen_s617_tune=True,tune_rule='v8-6.17',peak_rss_bytes=int(3.8*2**30))
        self.assertIsNone(tuning_peak_error(record))
        record['peak_rss_bytes']+=1
        self.assertIsNotNone(tuning_peak_error(record))
