"""E5a: depth-dp refresh plans recorded by call number and replayed by the plan policy."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pt_agent.backends import CountBackend, MockBackend, RefreshBackend
from pt_agent.backends.refresh import RValue
from pt_agent.cost import estimate
from pt_agent.dsl.errors import PTDSLError
from pt_agent.dsl.parser import parse
from pt_agent.examples.shapes import mlp_weight_shapes
from pt_agent.lower import lower, pack_inputs
from pt_agent.plan import PlanError, express, levels, plan_program, record_dag, verify, verify_program


MLP = '''@fhe.program(slots=8192{header})
def mlp(x: Cipher[1, 64]):
    x = pack(x, layout="flat")
    h = linear(x, w=W["fc1.weight"], b=W["fc1.bias"], algo="bsgs")
    h = act(h, fn="silu", method="cheb", deg=31, domain=(-4.0, 4.0))
    h = act(h, fn="silu", method="cheb", deg=15, domain=(-4.0, 4.0))
    out = linear(h, w=W["fc2.weight"], b=W["fc2.bias"], algo="diag")
    return out
'''
AGENT = ', refresh="agent", placement="depth-dp-plan", l_max=5, boot_out_level=5'
SOFTMAX = '''@fhe.program(slots=8192{header})
def sm(x: Cipher[1,3,8]):
 a=pack(x,layout="token")
 p=softmax(a,steps=2,exp_method="remez",exp_domain=(-1,1),exp_deg=7,inv_method="newton",inv_domain=(0.01,1),inv_deg0=15,inv_iters=2,step_inv_method="remez",step_inv_domain=(0.05,1.5),step_inv_deg=63)
 return p
'''


def mlp_weights():
    rng = np.random.default_rng(5)
    return {name: rng.normal(scale=.1, size=shape) for name, shape in mlp_weight_shapes().items()}


def toy(b):
    """Duplicated subexpressions, a frontend subtraction, a long chain and a dead branch."""
    x, y = b.input(0), b.input(1)
    first, second = x*x, x*x
    for _ in range(2):
        first, second = first*first, second*second
    difference = first-y
    chain = difference*difference
    for _ in range(4):
        chain = chain*chain
    dead = chain
    for _ in range(9):
        dead = dead*dead
    result = chain*second+x
    b.output([result, chain*0.5])


def mock_outputs(run, slots, plan=None):
    data = [np.linspace(-.6, .6, slots), np.linspace(.2, .4, slots)]
    inner = MockBackend(slots)
    inner.bind(data)
    active = RefreshBackend(inner, policy='plan', plan=plan, **levels()) if plan is not None else inner
    run(active)
    return [value.value for value in inner._outputs]


class HeaderTests(unittest.TestCase):
    def test_depth_dp_plan_requires_agent_refresh(self):
        self.assertEqual(parse(MLP.format(header=AGENT)).header.placement, 'depth-dp-plan')
        with self.assertRaises(PTDSLError):
            parse(MLP.format(header=', refresh="compiler", placement="depth-dp-plan"'))


class ToyPlanTests(unittest.TestCase):
    def check_toy(self, frontend_normalize):
        inner, recorder, _ = record_dag(toy, 64, frontend_normalize=frontend_normalize)
        from pt_agent.cost import depth_dp_estimate
        placement = depth_dp_estimate(inner)
        self.assertGreater(placement.boots, 0)
        plan = express(inner, recorder, placement)
        self.assertEqual(plan['calls'], recorder.call_count)
        replay = verify(toy, 64, plan)
        self.assertEqual(replay['boots'], placement.boots)
        self.assertGreaterEqual(replay['min_level'], levels()['boot_in_level'])
        for expected, actual in zip(mock_outputs(toy, 64), mock_outputs(toy, 64, plan)):
            np.testing.assert_array_equal(actual, expected)
        return plan

    def test_cost_dag_duplicates_subtraction_and_dead_calls(self):
        plan = self.check_toy(True)
        self.assertTrue(plan['dead'])

    def test_shared_frontend_negation(self):
        def shared(b):
            x, y = b.input(0), b.input(1)
            y = y*y
            early = x*x-y
            late = x
            for _ in range(10):
                late = late*late
            late = late-y
            b.output([early*late])
        inner, recorder, _ = record_dag(shared, 64, frontend_normalize=True)
        from pt_agent.cost import depth_dp_estimate
        placement = depth_dp_estimate(inner)
        plan = express(inner, recorder, placement)
        self.assertIn('neg', {inner.kind[node] for _, _, node in plan['substitutions']})
        self.assertEqual(plan['merged'], [])
        self.assertEqual(verify(shared, 64, plan)['boots'], placement.boots)
        for expected, actual in zip(mock_outputs(shared, 64), mock_outputs(shared, 64, plan)):
            np.testing.assert_array_equal(actual, expected)

    def test_plain_dag(self):
        def live(b):
            x = b.input(0)
            value = x*x
            for _ in range(8):
                value = value*value+x
            b.output([value])
        inner, recorder, _ = record_dag(live, 64, frontend_normalize=False)
        from pt_agent.cost import depth_dp_estimate
        placement = depth_dp_estimate(inner)
        plan = express(inner, recorder, placement)
        self.assertEqual(plan['dead'], [])
        self.assertEqual(verify(live, 64, plan)['boots'], placement.boots)

    def test_changed_call_sequence_is_rejected(self):
        plan = self.check_toy(True)
        def shorter(b):
            x, y = b.input(0), b.input(1)
            b.output([x*y])
        with self.assertRaises(Exception):
            verify(shorter, 64, plan)

    def test_lazy_policy_rejects_plans(self):
        with self.assertRaises(Exception):
            RefreshBackend(CountBackend(64, boot_out_level=5), plan=dict(calls=0, substitutions=[], outputs=[],
                                                                          dead=[]), **levels())


class ProgramPlanTests(unittest.TestCase):
    def check_program(self, text, weights, arrays):
        agent, compiler = parse(text.format(header=AGENT)), parse(text.format(header=''))
        predicted = estimate(agent, weights)
        reference = estimate(compiler, weights)
        self.assertEqual(predicted['placement_model'], 'depth-dp-plan')
        self.assertEqual(predicted['boots_pred'], reference['boots_pred'])
        self.assertEqual(predicted['latency_pred_us'], reference['latency_pred_us'])
        self.assertEqual(predicted['refresh_plan']['replay']['boots'], predicted['boots_pred'])
        plan, _, _ = plan_program(agent, weights)
        if arrays is None:
            return agent, plan
        outputs = []
        for program, active_plan in ((compiler, None), (agent, plan)):
            inner = MockBackend(program.header.slots, batch=len(arrays[0]))
            inner.bind(pack_inputs(program, arrays))
            active = (RefreshBackend(inner, policy='plan', plan=active_plan, **levels())
                      if active_plan is not None else inner)
            values, _ = lower(program, weights, active)
            outputs.append([(value.value if isinstance(value, RValue) else value).value for value in values])
        for expected, actual in zip(*outputs):
            np.testing.assert_array_equal(actual, expected)
        return agent, plan

    def test_mlp(self):
        weights = mlp_weights()
        arrays = [np.random.default_rng(1).normal(size=(2, 64))]
        agent, plan = self.check_program(MLP, weights, arrays)
        self.assertGreater(plan['boots'], 0)
        from pt_agent.export import export_package
        with tempfile.TemporaryDirectory() as directory:
            package = export_package(MLP.format(header=AGENT), agent, weights, Path(directory))
            saved = json.loads((package/'config/refresh_plan.json').read_text())
            self.assertEqual(saved['boots'], plan['boots'])
            self.assertEqual(saved['source'], 'package.runtime.invoke')
            self.assertEqual(saved['replay']['boots'], plan['boots'])
            self.assertEqual(json.loads((package/'config/compile_argv.json').read_text()).count('--pars'), 1)

    def test_token_softmax(self):
        # Estimate and replay only: this parser fixture's domains are not fitted to any data,
        # so Mock equality is covered by the CostDag toy test.
        _, plan = self.check_program(SOFTMAX, {}, None)
        self.assertGreater(plan['boots'], 0)

    def test_refresh_statements_are_rejected(self):
        text = MLP.format(header=AGENT).replace('    out = linear', '    h = refresh(h)\n    out = linear')
        with self.assertRaises(PlanError):
            plan_program(parse(text), mlp_weights())


if __name__ == '__main__':
    unittest.main()
