import json
from pathlib import Path
import unittest

import numpy as np

from pt_agent.backends import CountBackend, DagBackend, MockBackend
from pt_agent.cost import PROFILES, estimate, load_latency
from pt_agent.data.cifar import calibration_set
from pt_agent.dsl.parser import parse
from pt_agent.intake import Graph
from pt_agent.lower import lower, pack_inputs
from pt_agent.placement import PlacementError, depth_dp
from pt_agent.paths import pt_result, work_result
from pt_agent.tests._artifacts import DATA, WORK_RESULTS, require


R3 = pt_result('r3-resnet20-20261020T131658')
PROGRAMS = (
    (work_result('pt-agent-r6b-127-1p5-compiler-20261021T025533-1843/program.ptdsl'), 383, 101340690),
    (work_result('pt-agent-r6b-63-1p4-compiler-20261021T025558-09cd/program.ptdsl'), 288, 76809672),
)


def place(dag):
    config = PROFILES['qwen48']
    return depth_dp(dag, last_boot_level=config['boot_out_level']-config['boot_in_level'],
                    boot_in_level=config['boot_in_level'], boot_out_level=config['boot_out_level'],
                    latency_table=load_latency('qwen48'))


@require(DATA, WORK_RESULTS, R3)
class DagTests(unittest.TestCase):
    def test_cipher_operands_context_and_count_keys(self):
        dag = DagBackend(16)
        x = dag.input(0)
        dag.op_context = 17
        y = x*x
        z = (y+.5).rotate(1)
        dag.output([z])
        counter = CountBackend(16)
        cx = counter.input(0)
        counter.output([((cx*cx)+.5).rotate(1)])
        self.assertEqual(dag.kind, ['input', 'mul_cc', 'add', 'rot'])
        self.assertEqual(dag.args, [(), (0, 0), (1,), (2,)])
        self.assertEqual(dag.prog_op, [None, 17, 17, 17])
        self.assertEqual(dag.stats()['nodes'], 4)
        self.assertEqual({key: value for key, value in dag.stats().items() if key != 'nodes'}, counter.stats())

    def test_mock_node_watch_noise_and_full_slots(self):
        data = np.arange(16, dtype=np.float64)
        mock = MockBackend(16, watch={0, 1}, noise=(1e-3, 0))
        mock.bind([data])
        x = mock.input(0)
        expected_input = data+np.random.default_rng(0).normal(0, 1e-3, data.shape)
        self.assertTrue(np.array_equal(x.value, expected_input))
        y = x*.5
        expected = expected_input*.5+np.random.default_rng(1).normal(0, 1e-3, data.shape)
        self.assertTrue(np.array_equal(y.value, expected))
        self.assertEqual(mock._nid, 2)
        self.assertEqual(mock.watch_max[0], 15.0)
        self.assertEqual(mock.watch_max[1], float(np.max(np.abs(expected_input*.5))))

    def test_mock_without_optional_arguments_is_unchanged(self):
        data = np.arange(16, dtype=np.float64)
        outputs = []
        for options in ({}, dict(watch=set(), noise=None), dict(watch={0, 1, 2, 3}, noise=None)):
            mock = MockBackend(16, **options)
            mock.bind([data])
            x = mock.input(0)
            y = -(x*.5+x.rotate(1))
            mock.output([y])
            outputs.append(mock.outputs[0])
        for actual in outputs[1:]:
            self.assertTrue(np.array_equal(actual, outputs[0]))


@require(DATA, WORK_RESULTS, R3)
class PlacementTests(unittest.TestCase):
    def test_chain(self):
        dag = DagBackend(16)
        value = dag.input(0)
        for _ in range(8):
            value = value*.5
        dag.output([value])
        result = place(dag)
        self.assertEqual(result.cuts, [4, 7])
        self.assertEqual((result.barriers, result.boots, result.max_depth), (2, 1, 8))

    def test_fork(self):
        dag = DagBackend(16)
        y = dag.input(0)*.5
        p, q = y, y.rotate(1)
        watched = []
        for index in range(4):
            p, q = p*.5, q*.25
            if index == 2:
                watched = [p.nid, q.nid]
        z = p+q
        for _ in range(2):
            z = z*.5
        dag.output([z])
        result = place(dag)
        self.assertEqual(result.cuts, [4])
        self.assertEqual((result.barriers, result.boots, result.max_depth), (1, 2, 7))
        self.assertEqual(result.boot_nodes, watched)

    def test_explicit_refresh_is_rejected(self):
        dag = DagBackend(16)
        dag.output([dag.input(0).bootstrap()])
        with self.assertRaisesRegex(PlacementError, 'requires a program without explicit refresh'):
            place(dag)


@require(DATA, WORK_RESULTS, R3)
class CompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = Graph.load(R3/'graph')
        cls.programs = []
        for path, boots, latency in PROGRAMS:
            if not path.is_file():
                raise FileNotFoundError(path)
            cls.programs.append(parse(path.read_text()))

    def test_compiler_snapshots(self):
        for program, (_, boots, latency) in zip(self.programs, PROGRAMS):
            with self.subTest(boots=boots):
                result = estimate(program, self.graph.weights)
                self.assertEqual(result['boots_pred'], boots)
                self.assertEqual(result['barriers'], 76)
                self.assertEqual(result['counts']['boot'], boots)
                self.assertEqual(result['counts']['rot'], 2624)
                self.assertEqual(result['placement_model'], 'depth-dp')
                self.assertTrue(result['placement_exact'])
                error = abs(result['latency_pred_us']/latency-1)
                self.assertLessEqual(error, .02)
                print('R8c placement='+json.dumps(dict(boots=result['boots_pred'], barriers=result['barriers'],
                      latency_pred_us=result['latency_pred_us'], latency_plan_us=latency, rel_err=error,
                      boot_share=result['boot_share'], hotspots=result['hotspots'])), flush=True)

    def test_node_ids_and_watch(self):
        program = self.programs[1]
        dag = DagBackend(16384)
        lower(program, self.graph.weights, dag)
        placement = place(dag)
        images, _ = calibration_set()
        packed = pack_inputs(program, [images[:2]])
        values = []
        for options in ({}, dict(watch=set(placement.boot_nodes), noise=None)):
            mock = MockBackend(16384, batch=2, **options)
            mock.bind(packed)
            outputs, _ = lower(program, self.graph.weights, mock)
            self.assertEqual(len(dag.kind), mock._nid)
            if options:
                self.assertEqual(set(mock.watch_max), set(placement.boot_nodes))
                self.assertGreater(max(mock.watch_max.values()), 0)
            values.append(outputs[0].value)
        self.assertTrue(np.array_equal(values[0], values[1]))

    def test_hotspots_and_boot_share(self):
        program = self.programs[1]
        result = estimate(program, self.graph.weights)
        hotspots = result['hotspots']
        self.assertEqual(len(hotspots), 5)
        shares = [row['latency_share'] for row in hotspots]
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertLessEqual(sum(shares), 1.0)
        valid_lines = {op.lineno for op in program.ops}
        self.assertTrue(all(row['line'] in valid_lines for row in hotspots))
        self.assertGreater(result['boot_share'], .8)
        self.assertLess(result['boot_share'], 1.0)

    def test_placement_mode_selection(self):
        text = PROGRAMS[1][0].read_text()
        lazy = estimate(self.programs[1], self.graph.weights, placement_model='lazy')
        self.assertEqual(lazy['boots_pred'], 382)
        self.assertFalse(lazy['placement_exact'])
        self.assertEqual(lazy['boot_nodes'], [])
        self.assertEqual(lazy['hotspots'], [])
        agent = parse(text.replace('refresh="compiler"', 'refresh="agent", l_max=5, boot_out_level=5', 1))
        self.assertTrue(estimate(agent, self.graph.weights)['placement_exact'])
        fallback = parse(text.replace('placement="depth-dp"', 'placement="optimized"', 1))
        other = estimate(fallback, self.graph.weights)
        self.assertEqual(other['placement_model'], 'lazy')
        self.assertFalse(other['placement_exact'])


if __name__ == '__main__':
    unittest.main()
