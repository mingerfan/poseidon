"""E7 confirmation-only tail coverage, intermediate states and offline replies."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent.planners import confirm, agent, tuner
from pt_agent.planners.agent_tools import AgentContext
from pt_agent.planners.confirm import Candidate
from pt_agent.planners.tune_replay import SavedTuneClient, ReplayMismatch
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program


SOURCE = '@fhe.program(slots=1024)\ndef net(x: Cipher[1,2]):\n a=pack(x,layout="flat")\n y=act(a,fn="relu",method="cheb",deg=63,domain=(-1.,1.))\n return y\n'


def context():
    class Samples:
        def __init__(self): self.reads = []
        def get(self, split):
            self.reads.append(split)
            start, count = {'calibration': (0, 83), 'heldout': (83, 20), 'confirm': (103, 20)}[split]
            x = np.column_stack((np.arange(start+1, start+count+1, dtype=float), np.ones(count)))
            return {'x': x}, None
    inputs, _ = Samples().get('heldout')
    info = SimpleNamespace(task='classification', threshold=dict(agreement_pct=99., logit_rel_err=.01),
                           split=dict(version='split-v2'), input_handling={})
    return AgentContext(graph=SimpleNamespace(inputs=['x'], outputs=['y']), weights={}, calibration={},
        weight_shapes={}, val_images=inputs['x'], val_inputs=inputs, val_exact_logits=inputs['x'],
        val_labels=None, val_exact_top1=None, probe_images=None, probe_act_inputs=[], needed_margin={},
        act_tensors=[], model_name='fixture', val_split='heldout', model_info=info, samples=Samples())


def numeric(error=0., u=.5):
    return dict(agree=20, total=20, agreement_pct=100., logit_rel_err=error, feasible=True,
                max_u=u, diverged=False, n_diverged=0, boot_input_bound=None, boot_input_max_abs=None)


class TailTests(unittest.TestCase):
    def tail_context(self):
        ctx = context()
        with patch.object(confirm, 'reference_forward', side_effect=lambda g, x: {'y': x['x']}):
            result = confirm.build_non_report_context(ctx)
        return ctx, result

    def test_partition_order_and_123_samples(self):
        ctx, result = self.tail_context()
        self.assertEqual(ctx.samples.reads, ['calibration', 'heldout', 'confirm'])
        np.testing.assert_array_equal(result.val_images[:, 0], np.arange(1, 124))
        np.testing.assert_array_equal(result.val_exact_logits, result.val_images)
        self.assertEqual(result.tail_sample_positions[82], dict(partition='calibration', index=82))
        self.assertEqual(result.tail_sample_positions[83], dict(partition='heldout', index=0))
        self.assertEqual(result.tail_sample_positions[-1], dict(partition='confirm', index=19))

    def test_report_is_rejected_before_sample_access(self):
        ctx = context()
        with self.assertRaisesRegex(ValueError, 'report is forbidden'):
            confirm.build_non_report_context(ctx, splits=('calibration', 'heldout', 'report'))
        self.assertEqual(ctx.samples.reads, [])

    def test_confirm_pass_tail_outlier_rejects_then_selects_other(self):
        ctx, tail = self.tail_context()
        ctx.val_split = 'confirm'
        pool = [Candidate(SOURCE, 'llm', 1, 1., 1), Candidate(SOURCE.replace('net(', 'other('), 'llm', 2, 2., 2)]
        def validate(program, current, **kwargs):
            result = numeric()
            if current.val_split == 'non_report':
                actual = current.val_exact_logits.copy()
                if program.name == 'net': actual[82, 0] += 10
                kwargs['output_callback'](actual)
                result.update(total=123, agree=123, logit_rel_err=10/123 if program.name == 'net' else 0.,
                              max_u=4., feasible=False)
            return result
        with patch.object(confirm, 'validate', side_effect=validate), \
             patch.object(confirm, 'build_non_report_context', return_value=tail):
            result = confirm.confirm_select(pool, ctx, confirm_rule='c2', engine_profile='tensor')
        self.assertIs(result['selected'], pool[1])
        self.assertFalse(result['evals'][0]['passed'])
        checked = result['evals'][0]['tail']
        self.assertEqual(checked['total'], 123)
        self.assertEqual(checked['worst_samples'][0]['partition'], 'calibration')
        self.assertEqual(checked['worst_samples'][0]['index'], 82)
        self.assertEqual(checked['worst_samples'][0]['relative_error'], 10/123)
        self.assertTrue({'agree', 'agreement_pct', 'logit_rel_err', 'n_diverged', 'passed', 'seconds'} <= set(checked))
        self.assertTrue(result['evals'][1]['tail']['passed'])
        self.assertEqual(result['tail_evals'], 2)

    def test_c1_does_not_load_tail_and_keeps_result_shape(self):
        ctx = context(); ctx.val_split = 'confirm'
        with patch.object(confirm, 'validate', return_value=numeric()), \
             patch.object(confirm, 'build_non_report_context') as tail:
            pool = [Candidate(SOURCE, 'llm', 1, 1., 1)]
            result = confirm.confirm_select(pool, ctx)
            tail.assert_not_called()
        self.assertEqual(set(result), {'selected', 'evals', 'deduplicated', 'duplicates'})
        self.assertIs(result['selected'], pool[0])

    def test_max_evals_is_still_six(self):
        ctx = context(); ctx.val_split = 'confirm'
        pool = [Candidate(SOURCE.replace('net(', f'net{i}('), 'llm', i, i, i) for i in range(8)]
        with patch.object(confirm, 'validate', return_value=numeric(.02)), \
             patch.object(confirm, 'build_non_report_context') as tail:
            result = confirm.confirm_select(pool, ctx, confirm_rule='c2')
            tail.assert_not_called()
        self.assertEqual(len(result['evals']), 6)
        self.assertTrue(result['max_evals_reached'])

    def test_cifar_stays_c1_and_is_marked_not_applicable(self):
        ctx = context(); ctx.val_split = 'confirm'; ctx.model_info.split['kind'] = 'legacy_cifar'
        with patch.object(confirm, 'validate', return_value=numeric()), \
             patch.object(confirm, 'build_non_report_context') as tail:
            result = confirm.confirm_select([Candidate(SOURCE, 'llm', 1, 1., 1)], ctx, confirm_rule='c2')
            tail.assert_not_called()
        self.assertEqual(result['confirm_rule'], 'c1')
        self.assertFalse(result['applicable'])

    def test_causal_worst_error_uses_last_nonpadding_token(self):
        ctx = context(); ctx.model_info.task = 'causal_lm'
        ctx.val_inputs = {'attention_mask': np.array([[1, 1, 0], [1, 0, 0]])}
        values = np.arange(12).reshape(2, 3, 2)
        np.testing.assert_array_equal(confirm._metric_outputs(ctx, values), values[[0, 1], [1, 0]])


class IntermediateTests(unittest.TestCase):
    def test_llm_round_states_match_accepted_history_and_final(self):
        from pt_agent.tests.test_e6_tuning import simulation, Client
        result, _, _ = simulation(Client([dict(candidates=[dict(moves=['m1'])])]*4), degree=127)
        rounds = [h['round'] for h in result.history if h['phase'] == 'llm_round' and h['accepted_candidates']]
        self.assertEqual([s['round'] for s in result.accepted_states], rounds)
        self.assertEqual(result.accepted_states[-1]['source'], result.source)
        self.assertEqual(result.accepted_states[-1]['cost'], result.cost_after)

    def test_token_sweep_records_without_extra_evaluations(self):
        from pt_agent.tests.test_a1_tuner import TunerTests, measured
        result, validation = TunerTests().simulate(measured(), measured())
        self.assertEqual(len(result.accepted_states), 1)
        self.assertEqual(result.accepted_states[0]['round'], 1)
        self.assertEqual(result.accepted_states[0]['source'], result.source)
        self.assertEqual(validation.call_count, 5)

    def test_flat_sweep_states(self):
        program = parse(SOURCE)
        ctx = SimpleNamespace(prompt_version='r8-v4', weights={}, profile='qwen48')
        def validate(p, *args, **kwargs):
            return dict(numeric(), acc_drop_pp=0., rows=[dict(obs_min=-.5, obs_max=.5)], worst_acts=[])
        def cost(p, *args, **kwargs):
            return dict(latency_pred_us=p.ops[-1].attrs['deg'], boots_pred=1, boot_nodes=[])
        with patch.object(tuner, 'cached_estimate', side_effect=cost), \
             patch.object(tuner, 'validate', side_effect=validate), \
             patch.object(tuner, '_exact_check', return_value=dict(ok=True)):
            result = tuner.tune(program, ctx, max_sweeps=2)
        self.assertEqual([s['round'] for s in result.accepted_states], [1, 2])
        self.assertEqual(result.accepted_states[-1]['source'], result.source)

    def test_c2_pool_only_does_not_change_loop_best_stale_or_feedback(self):
        from pt_agent.tests.test_a1_session_steps import feedback, reply, Client
        ctx = SimpleNamespace(prompt_version='r8-v8', model_info=SimpleNamespace(split={}))
        states = []
        with TemporaryDirectory() as tmp, patch.object(agent, 'build_system_prompt', return_value='system'), \
             patch.object(agent, 'initial_message', return_value='initial'):
            for rule in ('c1', 'c2'):
                state = agent.new_agent_state(ctx, max_iters=3, engine_profile='tensor', confirm_rule=rule,
                                              tune_rule='v8-6.11')
                client = Client([reply(SOURCE)]*3)
                p = parse(SOURCE)
                final = print_program(p)
                intermediate = final.replace('deg=63', 'deg=31')
                t = tuner.TuneResult(p, final, [], {}, dict(latency_pred_us=10., boots_pred=2), 1, 1, 1.,
                    [feedback()['validation']], True, accepted_states=[dict(source=intermediate,
                    cost=dict(latency_pred_us=1., boots_pred=1), round=1)])
                with patch.object(agent, 'build_system_prompt', return_value='system'), \
                     patch.object(agent, 'initial_message', return_value='initial'), \
                     patch.object(agent, 'evaluate', return_value=feedback()), \
                     patch.object(agent, 'tune', return_value=t):
                    while state['phase'] != 'confirm':
                        agent.agent_step(ctx, state, trace_path=Path(tmp)/rule/'trace.jsonl', client=client)
                states.append(state)
        for key in ('best', 'stale', 'messages', 'history', 'first_feasible', 'calls', 'llm_min', 'has_tuned'):
            self.assertEqual(states[0][key], states[1][key])
        self.assertFalse(any(c['origin'].startswith('tuned-r') for c in states[0]['pool']))
        self.assertTrue(any(c['origin'] == 'tuned-r1' for c in states[1]['pool']))


class ReplayTests(unittest.TestCase):
    def test_equal_request_replays_saved_response_without_provider(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); calls = root/'calls'; calls.mkdir()
            request = dict(messages=[dict(role='user', content='fixed')], purpose='tune-select-iter2-round1')
            response = dict(content='{"candidates":[]}', seconds=3., usage={})
            (calls/'call-001.request.json').write_text(json.dumps(request))
            (calls/'call-001.response.json').write_text(json.dumps(response))
            client = SavedTuneClient(calls, root/'replay', iteration=2, limit=9)
            self.assertEqual(client.chat(**request), response)
            self.assertEqual(client.calls, 1)
            report = json.loads((root/'replay/001-comparison.json').read_text())
            self.assertTrue(report['equal']); self.assertEqual(report['api_calls'], 0)

    def test_different_request_stops_with_saved_difference(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); calls = root/'calls'; calls.mkdir()
            request = dict(messages=[dict(role='user', content='fixed')], purpose='tune-select-iter2-round1')
            (calls/'call-001.request.json').write_text(json.dumps(request))
            client = SavedTuneClient(calls, root/'replay', iteration=2, limit=9)
            with self.assertRaises(ReplayMismatch):
                client.chat([dict(role='user', content='changed')], purpose=request['purpose'])
            report = json.loads((root/'replay/001-comparison.json').read_text())
            self.assertFalse(report['equal'])
            self.assertEqual(report['differences'][0]['path'], '$.messages[0].content')
            self.assertEqual(client.calls, 0)


from pt_agent.tests._artifacts import RESULTS_DIR, require


@require(RESULTS_DIR/'E6r2-resume-20261107/e6r2-b/bert/p1/session/SESSION.json',
         RESULTS_DIR/'E6r2-resume-20261107/e6r2-b/qwen/p2/session/SESSION.json')
class HistoricalC1Tests(unittest.TestCase):
    def test_saved_sessions_keep_c1_confirmation(self):
        from dataclasses import asdict
        from pt_agent.experiments.e7_confirm_tail_check import context
        from pt_agent.experiments.e6_llm_select_tuning import configurations
        from pt_agent.planners.transformer_session import json_safe
        from pt_agent.tensor_sim import clear_caches
        for model, label in (('bert','p1'),('qwen','p2')):
            clear_caches()
            session=RESULTS_DIR/'E6r2-resume-20261107/e6r2-b'/model/label/'session'
            case=dict(model=model,model_dir=str(configurations()[model]['model']),session=str(session))
            ctx,state=context(case)
            result=confirm.confirm_select([Candidate(**item) for item in state['pool']],ctx,engine_profile='tensor')
            if result['selected'] is not None:result['selected']=asdict(result['selected'])
            expected=json.loads((session/'AGENT.json').read_text())['confirm']
            self.assertEqual(json_safe(result),expected)
            clear_caches()

    def test_e6r2_replayed_response_count_comes_from_agent(self):
        from pt_agent.experiments.e6r2_resume_after_outage import session_summary
        for model, count in (('bert',6),('qwen',8)):
            session=RESULTS_DIR/'E6r2-resume-20261107/e6r2-a'/model/'E6r2-a/session'
            self.assertEqual(session_summary(session)['replayed_responses'],count)


if __name__ == '__main__': unittest.main()
