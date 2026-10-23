from collections import Counter
import copy
import json
from pathlib import Path
import unittest

from pt_agent.dsl.ops_spec import RESERVED_NAMES
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake import Graph
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import require


R3 = pt_result('r3-resnet20-20261020T131658')


@require(R3)
class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(R3 / 'graph')
        cls.calibration = json.loads((R3 / 'calibration.json').read_text())

    def test_plan(self):
        text = plan(self.graph, self.calibration, deg=31, margin=1.25)
        program = parse(text)
        self.assertTrue(check(program, {key: value.shape for key, value in self.graph.weights.items()}).ok)
        self.assertEqual(Counter(op.kind for op in program.ops),
                         dict(pack=1, conv2d=19, act=19, add=9, slice=4, pad=2, mean=1, relayout=1, linear=1))
        self.assertEqual(len(program.ops), 57)
        first = next(node for node in self.graph.nodes if node.op == 'act')
        domain = self.calibration[first.inputs[0]]
        self.assertEqual(next(op for op in program.ops if op.kind == 'act').attrs['domain'],
                         (domain['min']*1.25, domain['max']*1.25))
        self.assertTrue(all(op.attrs['bn'] is not None for op in program.ops if op.kind == 'conv2d'))
        for op in program.ops:
            for name in op.outputs:
                self.assertNotIn('__', name)
                self.assertNotIn(name, RESERVED_NAMES)

    def test_reject_unsupported_and_small_margin(self):
        graph = copy.deepcopy(self.graph)
        node = next(node for node in graph.nodes if node.op == 'act')
        node.op = 'softmax'
        with self.assertRaisesRegex(ValueError, node.id):
            plan(graph, self.calibration)
        with self.assertRaises(ValueError):
            plan(self.graph, self.calibration, margin=1.0)
