"""E5b: stage labels, stage-cut placement, pruning and the program-plan selection."""

from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pt_agent.backends import MockBackend, RefreshBackend
from pt_agent.backends.cost_dag import CostDag
from pt_agent.backends.refresh import RValue
from pt_agent.cost import depth_dp_estimate, estimate
from pt_agent.dsl.parser import parse
from pt_agent.lower import lower, pack_inputs
from pt_agent.plan import levels, plan_program, record_dag, verify, express
from pt_agent.stage_plan import (MUL, _users, candidates, labels, model_for, prune, select, simulate, stage_cut)
from pt_agent.tests.test_e5a_plan import MLP, SOFTMAX, mlp_weights


PLAN = ', refresh="agent", placement="program-plan", l_max=5, boot_out_level=5'


def staged(b):
    """Three statements; the second has two named stages repeated per item, like a per-token kernel."""
    x, y = b.input(0), b.input(1)
    b.op_context = 10
    a = x*x
    c = y*y
    b.op_context = 11
    outs = []
    for item in (a, c):
        b.stage('square')
        s = item*item
        s = s*s+item
        b.stage('scale')
        s = s*s
        s = s*0.5
        outs.append(s)
    b.op_context = 12
    total = outs[0]*outs[1]
    total = total*total+a
    b.output([total, outs[0]+x])


def brute_force(dag, node_labels, model):
    """Minimum latency over every cut subset under the refresh-all-touched rule."""
    n = len(dag.kind)
    distinct = sorted({node_labels[i] for i in range(n) if dag.kind[i] != 'input'})
    index = {label: i for i, label in enumerate(distinct)}
    m = len(distinct)
    li = [-1 if dag.kind[i] == 'input' else index[node_labels[i]] for i in range(n)]
    users = _users(dag)
    last = [max((m if u == n else li[u] for u in users[i]), default=-2) for i in range(n)]
    best = None
    for size in range(m):
        for cuts in combinations(range(m-1), size):
            refreshed, level, feasible = {}, [0]*n, True
            bounds = (-1,)+cuts+(m-1,)
            for start, end in zip(bounds, bounds[1:]):
                entry = model.input_level if start < 0 else model.boot_out_level
                for i in range(n):
                    if not start < li[i] <= end:
                        continue
                    current = min(entry if li[a] <= start else level[a] for a in dag.args[i])
                    if dag.kind[i] in MUL:
                        current -= 1
                        if current < model.boot_in_level:
                            feasible = False
                    level[i] = current
                if end == m-1:
                    break
                for i in range(n):
                    if start < li[i] <= end and level[i] < model.boot_out_level and last[i] > end:
                        refreshed[i] = {u for u in users[i] if (m if u == n else li[u]) > end}
            if not feasible:
                continue
            result = simulate(dag, refreshed, model)
            if result is None:
                continue
            total = sum(result[1])+len(refreshed)*model.boot_us
            best = total if best is None else min(best, total)
    return best, m


class LabelTests(unittest.TestCase):
    def test_labels_order_and_repeated_stage_names(self):
        inner, recorder, _ = record_dag(staged, 64, frontend_normalize=True)
        node_labels = labels(inner, recorder)
        for nid, args in enumerate(inner.args):
            for arg in args:
                self.assertLessEqual(node_labels[arg], node_labels[nid])
        second = {stage for (statement, stage) in recorder.node_stage if statement == 1}
        self.assertEqual(second, {1, 2})

    def test_stage_marks_leave_the_dag_unchanged(self):
        program = parse(SOFTMAX.format(header=''))
        plain = CostDag(program.header.slots)
        lower(program, {}, plain)
        plain.finalize()
        inner, _, _ = record_dag(lambda b: lower(program, {}, b), program.header.slots, frontend_normalize=True)
        self.assertEqual((plain.kind, plain.args), (inner.kind, inner.args))


class StageCutTests(unittest.TestCase):
    def test_matches_brute_force(self):
        model = model_for()
        inner, recorder, _ = record_dag(staged, 64, frontend_normalize=True)
        node_labels = labels(inner, recorder)
        expected, cuts = brute_force(inner, node_labels, model)
        self.assertLessEqual(cuts, 13)
        placement, stats = stage_cut(inner, node_labels, model)
        self.assertAlmostEqual(placement.latency_us, expected, places=3)
        plan = express(inner, recorder, placement)
        self.assertEqual(verify(staged, 64, plan)['boots'], placement.boots)

    def test_prune_and_select(self):
        model = model_for()
        inner, recorder, _ = record_dag(staged, 64, frontend_normalize=True)
        depth = depth_dp_estimate(inner)
        found = candidates(inner, recorder, depth, model)
        for name in ('depth-dp', 'depth-dp+prune', 'stage-cut', 'stage-cut+prune'):
            self.assertIn('placement', found[name], name)
            refreshed = {v: set(u) for v, u in found[name]['placement'].refreshed_uses.items()}
            self.assertIsNotNone(simulate(inner, refreshed, model))
        self.assertLessEqual(found['depth-dp+prune']['latency_us'], found['depth-dp']['latency_us'])
        self.assertLessEqual(found['stage-cut+prune']['latency_us'], found['stage-cut']['latency_us'])
        chosen = select(found)
        self.assertEqual(found[chosen]['latency_us'],
                         min(found[name]['latency_us'] for name in found if not name.startswith('_')))
        self.assertAlmostEqual(found['depth-dp']['latency_us'], depth.latency_us, places=3)

    def test_prune_removes_an_unneeded_bootstrap(self):
        model = model_for()
        def chain(b):
            x = b.input(0)
            y = x*x
            z = y+x
            b.output([z*z])
        inner, recorder, _ = record_dag(chain, 64, frontend_normalize=False)
        depth = depth_dp_estimate(inner)
        extra = {v: set(u) for v, u in depth.refreshed_uses.items()}
        extra[1] = {2}
        from pt_agent.stage_plan import as_placement
        padded = as_placement(inner, extra, model, cuts=[], max_depth=depth.max_depth)
        pruned, stats = prune(inner, padded, model)
        self.assertEqual(stats['removed'], 1)
        self.assertLess(pruned.latency_us, padded.latency_us)


class MergeTests(unittest.TestCase):
    def test_refreshed_negation_and_source_share_one_bootstrap(self):
        def both(b):
            x, y = b.input(0), b.input(1)
            y = y*y
            early = x*x-y
            late = x
            for _ in range(10):
                late = late*late
            keep = y*1.0
            late = late-y
            b.output([early*late, keep*late])
        inner, recorder, _ = record_dag(both, 64, frontend_normalize=True)
        depth = depth_dp_estimate(inner)
        refreshed = {v: set(u) for v, u in depth.refreshed_uses.items()}
        negation = next(n for n, k in enumerate(inner.kind) if k == 'neg')
        source = inner.args[negation][0]
        refreshed.setdefault(negation, set())
        refreshed.setdefault(source, set())
        users = [u for u in range(len(inner.kind)) if source in inner.args[u] and u != negation]
        refreshed[source].update(users)
        refreshed[negation].update(u for u in range(len(inner.kind)) if negation in inner.args[u])
        from pt_agent.stage_plan import as_placement
        placement = as_placement(inner, refreshed, model_for(), cuts=[], max_depth=depth.max_depth)
        plan = express(inner, recorder, placement)
        self.assertEqual(plan['merged'], [negation])
        self.assertEqual(plan['boots'], placement.boots-1)
        self.assertEqual(verify(both, 64, plan)['boots'], plan['boots'])


class ProgramPlanTests(unittest.TestCase):
    def test_mlp_program_plan(self):
        weights = mlp_weights()
        agent = parse(MLP.format(header=PLAN))
        reference = estimate(parse(MLP.format(header='')), weights)
        predicted = estimate(agent, weights)
        self.assertEqual(predicted['placement_model'], 'program-plan')
        self.assertLessEqual(predicted['latency_pred_us'], reference['latency_pred_us'])
        self.assertEqual(predicted['refresh_plan']['replay']['boots'], predicted['boots_pred'])
        plan, _, _ = plan_program(agent, weights)
        self.assertIn(plan['placement'], plan['candidates'])
        arrays = [np.random.default_rng(1).normal(size=(2, 64))]
        outputs = []
        for program, active_plan in ((parse(MLP.format(header='')), None), (agent, plan)):
            inner = MockBackend(program.header.slots, batch=2)
            inner.bind(pack_inputs(program, arrays))
            active = RefreshBackend(inner, policy='plan', plan=active_plan, **levels()) if active_plan else inner
            values, _ = lower(program, weights, active)
            outputs.append([(v.value if isinstance(v, RValue) else v).value for v in values])
        for expected, actual in zip(*outputs):
            np.testing.assert_array_equal(actual, expected)
        from pt_agent.export import export_package
        with tempfile.TemporaryDirectory() as directory:
            package = export_package(MLP.format(header=PLAN), agent, weights, Path(directory))
            saved = json.loads((package/'config/refresh_plan.json').read_text())
            self.assertEqual(saved['boots'], plan['boots'])
            self.assertTrue(saved['source'].endswith('package call sequence identical'))
            self.assertEqual(saved['replay']['boots'], plan['boots'])

    def test_token_softmax_program_plan(self):
        agent = parse(SOFTMAX.format(header=PLAN))
        predicted = estimate(agent, {})
        reference = estimate(parse(SOFTMAX.format(header='')), {})
        self.assertLessEqual(predicted['latency_pred_us'], reference['latency_pred_us'])
        self.assertEqual(predicted['refresh_plan']['replay']['boots'], predicted['boots_pred'])


if __name__ == '__main__':
    unittest.main()
