import contextlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.intake import Graph
from pt_agent.llm import LLMClient, LLMError
from pt_agent.planners.agent import run_agent
from pt_agent.planners.agent_tools import build_context, evaluate
from pt_agent.planners.prompts import build_system_prompt, feedback_message, initial_message
from pt_agent.planners.rule import plan
from pt_agent.dsl.parser import parse
from pt_agent.paths import pt_result


R3 = pt_result('r3-resnet20-20261020T131658')


class FakeClient:
    def __init__(self, responses, budget_path):
        self.responses = iter(responses)
        self.budget_path = budget_path

    def chat(self, messages, *, purpose):
        content = next(self.responses)
        with self.budget_path.open('a') as stream:
            stream.write(json.dumps(dict(ts='offline', purpose=purpose, model='fake', status=200, seconds=0,
                                         prompt_tokens=10, completion_tokens=20, finish_reason='stop', error=None))+'\n')
        return dict(content=content, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=20),
                    id='offline', seconds=0)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(R3 / 'graph')
        cls.calibration = json.loads((R3 / 'calibration.json').read_text())
        cls.ctx = build_context(cls.graph, cls.calibration, n_val=50)
        cls.high = plan(cls.graph, cls.calibration, deg=127, margin=1.5)
        cls.low = plan(cls.graph, cls.calibration, deg=63, margin=1.4)

    def test_context(self):
        self.assertEqual(len(self.ctx.act_tensors), 19)
        self.assertTrue(all(value > 0 for value in self.ctx.needed_margin.values()))
        self.assertEqual(self.ctx.val_images.shape, (50, 3, 32, 32))
        message = json.loads(initial_message(self.ctx))
        self.assertEqual(len(message['nodes']), len(self.graph.nodes))
        self.assertTrue(all('calib' in node and 'needed_margin' in node for node in message['nodes'] if node['op']=='act'))

    def test_evaluate_rule(self):
        feedback = evaluate(self.high, self.ctx)
        self.assertEqual(feedback['stage'], 'complete')
        self.assertTrue(feedback['exact']['ok'])
        self.assertEqual(feedback['cost']['boots_pred'], 383)
        self.assertFalse(feedback['validation']['diverged'])
        self.assertTrue(feedback['feasible'])

    def test_narrow_act_domain(self):
        lines = self.high.splitlines()
        act_lines = [index for index, line in enumerate(lines) if ' = act(' in line]
        lines[act_lines[2]] = re.sub(r'domain=\([^)]*\)', 'domain=(-0.5, 0.5)', lines[act_lines[2]])
        feedback = evaluate('\n'.join(lines)+'\n', self.ctx)
        self.assertFalse(feedback['feasible'])
        worst = feedback['validation']['worst_acts'][0]
        self.assertEqual(worst['graph_tensor'], self.ctx.act_tensors[2])
        self.assertEqual(worst['program_line'], act_lines[2]+1)

    def test_wrong_stride(self):
        changed = self.high.replace('stride=(1, 1)', 'stride=(2, 2)', 1)
        feedback = evaluate(changed, self.ctx)
        self.assertFalse(feedback['feasible'])
        self.assertTrue(feedback['stage'] in ('typecheck', 'lower')
                        or (feedback['exact'] is not None and not feedback['exact']['ok']))
        if feedback['stage'] == 'lower':
            self.assertIn('line', feedback['error'])

    def test_syntax_error(self):
        self.assertEqual(evaluate('def broken(', self.ctx)['stage'], 'parse')

    def test_loop_selects_and_stops(self):
        responses = ['not JSON', json.dumps(dict(program='def broken(')), json.dumps(dict(program=self.high)),
                     json.dumps(dict(program=self.low)), json.dumps(dict(program=self.low)), json.dumps(dict(program=self.low))]
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            client = FakeClient(responses, directory/'budget.jsonl')
            def scripted(latency):
                return dict(stage='complete', feasible=True, exact=dict(ok=True),
                            cost=dict(latency_pred_us=latency, boots_pred=0),
                            validation=dict(agree=50, total=50, diverged=False, max_u=.8,
                                            top1=92.0, exact_top1=92.0, acc_drop_pp=0.0,
                                            boot_input_bound=None, boot_input_max_abs=0.0))

            feedback = [dict(stage='parse', feasible=False, exact=None, cost=None, validation=None),
                        scripted(200), scripted(100), scripted(100), scripted(100)]

            def fake_confirm(candidates, ctx, **kwargs):
                ordered = sorted(candidates, key=lambda c: (c.latency_pred_us, c.iteration, c.origin))
                return dict(selected=ordered[0] if ordered else None, evals=[])

            with patch('pt_agent.planners.agent.evaluate', side_effect=feedback), \
                    patch('pt_agent.planners.agent.confirm_select', side_effect=fake_confirm):
                result = run_agent(self.ctx, client, max_iters=8, trace_path=directory/'trace.jsonl', tuner=False)
            rows = [json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len((directory/'budget.jsonl').read_text().splitlines()), 6)
            for row in rows:
                self.assertIsInstance(json.loads(row['messages'][1]['content']), dict)
        self.assertEqual(result['status'], 'feasible')
        self.assertEqual(result['first_feasible_iteration'], 3)
        self.assertEqual(result['best_iteration'], 4)
        self.assertEqual(result['llm_calls'], 6)
        self.assertEqual(result['best_program'], self.low)
        self.assertEqual((result['prompt_tokens'], result['completion_tokens']), (60, 120))

    def test_budget_exhausted_never_starts_worker(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'budget.jsonl'
            path.write_text('{}\n'*300)
            with patch('pt_agent.llm.subprocess.run', side_effect=AssertionError('worker must not run')):
                with self.assertRaisesRegex(LLMError, 'budget_exhausted'):
                    LLMClient(path).chat([], purpose='offline-budget-test')
            self.assertEqual(len(path.read_text().splitlines()), 300)

    def test_system_example_and_feedback(self):
        prompt = build_system_prompt(self.ctx)
        source = prompt.split('# example: residual_small\n', 1)[1].split('\n# example:', 1)[0]
        program = parse(source)
        self.assertEqual(program.name, 'tinynet')
        for feedback in (dict(error='parse'), dict(feasible=True), dict(feasible=False)):
            self.assertIn('next', json.loads(feedback_message(feedback, 1, None)))


class LLMOfflineTests(unittest.TestCase):
    def test_worker_framing_and_content_free_accounting(self):
        response = dict(id='offline', choices=[dict(message=dict(content='{"ok":true}'), finish_reason='stop')],
                        usage=dict(prompt_tokens=3, completion_tokens=4, total_tokens=7))
        frames = [SimpleNamespace(returncode=0, stdout=b'200\n'+json.dumps(response).encode(), stderr=b''),
                  SimpleNamespace(returncode=1, stdout=b'', stderr=b'transport_tls_failed\nDIAG {"stage":"connect","tls_error":"unexpected_eof"}\n')]
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'calls.jsonl'
            for index, frame in enumerate(frames):
                with self.subTest(index=index), patch('pt_agent.llm.load_api_key', return_value='offline-test-token'), \
                        patch('pt_agent.llm.subprocess.run', return_value=frame) as worker:
                    client = LLMClient(path)
                    if index == 0:
                        reply = client.chat([dict(role='user', content='private-message')], purpose='offline-test')
                        self.assertEqual(reply['content'], '{"ok":true}')
                        wire = json.loads(worker.call_args.kwargs['input'])
                        self.assertEqual(wire['loopback_proxy_port'], 0)
                        self.assertEqual(wire['body']['thinking'], dict(type='disabled'))
                    else:
                        with self.assertRaisesRegex(LLMError, 'transport_tls_failed'):
                            client.chat([], purpose='offline-error')
            text = path.read_text()
            self.assertNotIn('offline-test-token', text)
            self.assertNotIn('private-message', text)
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['status'], 200)
            self.assertEqual(rows[1]['error'], 'transport_tls_failed')
