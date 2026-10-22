import copy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pt_agent.capabilities import render_capabilities
from pt_agent.cost import estimate
from pt_agent.dsl.ir import canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.printer import print_program
from pt_agent.intake import Graph
from pt_agent.models.zoo import get
from pt_agent.planners.agent import run_agent
from pt_agent.planners.agent_tools import FEASIBLE_MAX_U, build_context, evaluate, validate
from pt_agent.planners.prompts import PROMPT_VERSION, TEMPLATE, build_system_prompt, feedback_message
from pt_agent.planners.tuner import TuneResult, tune
from pt_agent.paths import RESULTS_DIR


ROOT = RESULTS_DIR
R3 = ROOT/'r3-resnet20-20261020T131658'
R6C = ROOT/'run-resnet20-rule-20261020T165817/program.ptdsl'
R7B = ROOT/'run-resnet20-agent-20261021T033036/program.ptdsl'
R8D = ROOT/'r8d-methods-20261021T093203/cnn_plain'


class TunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.graph = Graph.load(R3/'graph')
        cls.calibration = json.loads((R3/'calibration.json').read_text())
        cls.ctx = build_context(cls.graph, cls.calibration, n_val=200)
        cls.ctx50 = copy.copy(cls.ctx)
        cls.ctx50.val_images = cls.ctx.val_images[:50]
        cls.ctx50.val_exact_logits = cls.ctx.val_exact_logits[:50]

    def test_01_validate_matches_evaluate(self):
        source = R6C.read_text()
        actual = validate(parse(source), self.ctx50)
        feedback = evaluate(source, self.ctx50)
        self.assertEqual(actual['agree'], feedback['validation']['agree'])
        self.assertEqual(actual['max_u'], feedback['validation']['max_u'])
        self.assertEqual(actual['feasible'], feedback['feasible'])
        self.assertEqual(len(actual['rows']), 19)
        self.assertTrue(all(row['obs_min'] < row['obs_max'] for row in actual['rows']))

    def test_02_real_resnet_monotonic_and_feasible(self):
        original = parse(R7B.read_text())
        snapshot = canonical(original)
        with patch('pt_agent.planners.tuner.N_COARSE', 100):
            result = tune(original, self.ctx, budget_s=900, max_sweeps=1)
        self.assertTrue(result.feasible)
        self.assertTrue(validate(result.program, self.ctx)['feasible'])
        self.assertLessEqual(result.cost_after['latency_pred_us'], result.cost_before['latency_pred_us'])
        self.assertEqual(snapshot, canonical(original))
        wanted = []
        for old, new in zip(original.ops, result.program.ops):
            self.assertEqual((old.kind, old.inputs, old.outputs, old.region, old.call_lines),
                             (new.kind, new.inputs, new.outputs, new.region, new.call_lines))
            for key in old.attrs:
                if old.kind == 'act' and key in ('deg', 'domain'):
                    if old.attrs[key] != new.attrs[key]:
                        wanted.append(dict(line=old.lineno, call_lines=list(old.call_lines), kind=key,
                                           old=old.attrs[key], new=new.attrs[key]))
                else:
                    self.assertEqual(old.attrs[key], new.attrs[key])
        self.assertCountEqual(result.changes, wanted)
        before, after = snapshot, canonical(result.program)
        for old, new in zip(before['ops'], after['ops']):
            if old[0] == 'act':
                for key in ('deg', 'domain'):
                    new[3][key] = old[3][key]
        self.assertEqual(before, after)
        self.assertEqual(canonical(parse(result.source)), canonical(result.program))

    def test_03_prefix_rollback(self):
        source = ('@fhe.program(slots=16384)\ndef net(v: Cipher[1, 2]):\n'
                  '    x = pack(v, layout="flat")\n')
        for index in range(8):
            source += f'    x = act(x, fn="relu", method="cheb", deg={127 if index < 2 else 63}, domain=(-8.,8.))\n'
        source += '    return x\n'
        ctx = SimpleNamespace(weights={}, profile='qwen48')

        def fake_validate(program, context, n=None):
            acts = [op for op in program.ops if op.kind == 'act']
            feasible = n is not None or all(op.attrs['deg'] >= 63 for op in acts)
            return dict(agree=200 if n is not None else 1000 if feasible else 980,
                        total=200 if n is not None else 1000, feasible=feasible, diverged=False, max_u=.8,
                        rows=[dict(obs_min=-2., obs_max=2.) for _ in acts])

        def fake_cost(program, weights, **kwargs):
            acts = [op for op in program.ops if op.kind == 'act']
            return dict(latency_pred_us=sum(op.attrs['deg'] for op in acts), boots_pred=len(acts),
                        hotspots=[dict(line=op.lineno, call_lines=list(op.call_lines), latency_share=.1)
                                  for op in acts])

        with patch('pt_agent.planners.tuner.validate', side_effect=fake_validate), \
                patch('pt_agent.planners.tuner.estimate', side_effect=fake_cost), \
                patch('pt_agent.planners.tuner._exact_check', return_value={'ok': True}):
            result = tune(parse(source), ctx)
        self.assertTrue(result.feasible)
        self.assertTrue(all(op.attrs['deg'] >= 63 for op in result.program.ops if op.kind == 'act'))
        rollback = next(row for row in result.history if row['phase'] == 'rollback')
        self.assertEqual(rollback['kept'], 2)
        self.assertEqual(rollback['removed'], 6)
        self.assertLessEqual(result.full_evals, 7)

    def test_04_minimax_keeps_degrees_and_symmetry(self):
        source = '@fhe.program(slots=16384)\ndef net(v: Cipher[1, 2]):\n    x = pack(v, layout="flat")\n'
        for _ in range(3):
            source += '    x = act(x, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-8.0, 8.0))\n'
        source += '    return x\n'
        original = parse(source)
        ctx = SimpleNamespace(weights={}, profile='qwen48')

        def fake_validate(program, context, n=None):
            acts = [op for op in program.ops if op.kind == 'act']
            total = 200 if n is not None else 1000
            return dict(agree=total, total=total, feasible=True, diverged=False, max_u=.8,
                        rows=[dict(obs_min=-3.0, obs_max=2.0) for _ in acts])

        def fake_cost(program, weights, **kwargs):
            acts = [op for op in program.ops if op.kind == 'act']
            return dict(latency_pred_us=100.0, boots_pred=len(acts),
                        hotspots=[dict(line=op.lineno, call_lines=list(op.call_lines), latency_share=.1) for op in acts])

        with patch('pt_agent.planners.tuner.validate', side_effect=fake_validate), \
                patch('pt_agent.planners.tuner.estimate', side_effect=fake_cost), \
                patch('pt_agent.planners.tuner._exact_check', return_value={'ok': True}):
            result = tune(original, ctx)
        self.assertTrue(result.feasible)
        self.assertEqual(result.coarse_evals, 0)
        acts = [op for op in result.program.ops if op.kind == 'act']
        self.assertEqual(len(acts), 3)
        for op in acts:
            self.assertEqual(op.attrs['method'], 'minimax_comp')
            self.assertEqual(tuple(op.attrs['degrees']), (15, 15, 27))
            self.assertIsNone(op.attrs['deg'])
            self.assertEqual(tuple(op.attrs['domain']), (-3.75, 3.75))
        self.assertEqual(len(result.changes), 3)
        self.assertTrue(all(change['kind'] == 'domain' for change in result.changes))

    def test_05_agent_tuner_on_off(self):
        source = R7B.read_text()
        program = parse(source)
        fb = evaluate(source, self.ctx50)
        # Isolate orchestration; the real coordinate search is covered above.
        feedback = dict(fb, feasible=True)
        cost = estimate(program, self.ctx50.weights)
        result = TuneResult(program, print_program(program), [], cost, cost, 1, 2, .1,
                            [dict(agree=50, total=50, max_u=.8, feasible=True)], True)

        class FakeClient:
            def chat(self, messages, *, purpose):
                return dict(content=json.dumps(dict(program=source)), seconds=0,
                            usage=dict(prompt_tokens=1, completion_tokens=2))

        with TemporaryDirectory() as directory:
            for enabled in (True, False):
                path = Path(directory)/f'trace-{enabled}.jsonl'
                with patch('pt_agent.planners.agent.evaluate', return_value=feedback), \
                        patch('pt_agent.planners.agent.tune', return_value=result) as tuner:
                    answer = run_agent(self.ctx50, FakeClient(), max_iters=2, trace_path=path, tuner=enabled)
                trace = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(trace), 2)
                self.assertEqual(tuner.call_count, 1 if enabled else 0)
                self.assertEqual(trace[0]['tuned'] is not None, enabled)
                self.assertEqual(answer['best']['source'] in ('tuned', 'llm'), True)
                parse(answer['best_program'])
                if not enabled:
                    self.assertEqual(answer['best_program'], source)

    def test_06_prompt_and_tuned_feedback(self):
        self.assertEqual(PROMPT_VERSION, 'r8-v3')
        generic = TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U,
                                  n_val=50, examples='').lower()
        for word in ('resnet', 'layer1.', 'conv1.weight', 'linear.weight', 'fc1.weight', 'fc2.weight', 'cifar'):
            self.assertNotIn(word, generic)
        for word in ('resnet', 'cifar', 'layer1.'):
            self.assertNotIn(word, build_system_prompt(self.ctx50).lower())
        changes = [dict(line=i, line_text=f'line{i}', call_texts=[], kind='deg', old=63, new=31)
                   for i in range(39, 0, -1)]
        result = json.loads(feedback_message(dict(feasible=True), 1, None,
                            tuned=dict(cost_after=dict(latency_pred_us=1, boots_pred=2), changes=changes)))
        self.assertEqual([row['line'] for row in result['tuned']['changes']], list(range(1, 31)))
        self.assertIn('structural changes', result['next'])


if __name__ == '__main__':
    unittest.main()
