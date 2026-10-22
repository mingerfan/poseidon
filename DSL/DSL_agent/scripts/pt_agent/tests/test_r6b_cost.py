import inspect
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import tokenize
import unittest

from pt_agent import toolchain
from pt_agent.cost import estimate, load_latency, plan_latency
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake import Graph
from pt_agent.planners.rule import plan
from pt_agent.paths import pt_result


R3 = pt_result('r3-resnet20-20261020T131658')


class CostTests(unittest.TestCase):
    def test_inner_does_not_read_registry(self):
        from pt_agent.experiments import r6b_refresh

        source = inspect.getsource(r6b_refresh._inside)
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                self.assertNotIn('registry', token.string)
                self.assertNotIn('PT_RESULTS', token.string)

    def test_resnet20_estimates(self):
        graph = Graph.load(R3 / 'graph')
        calibration = json.loads((R3 / 'calibration.json').read_text())
        for deg, margin, boots in ((127, 1.5, 383), (63, 1.4, 288)):
            with self.subTest(deg=deg):
                text = plan(graph, calibration, deg=deg, margin=margin)
                agent = plan(graph, calibration, deg=deg, margin=margin, refresh='agent')
                self.assertEqual(text.splitlines()[1:], agent.splitlines()[1:])
                program = parse(agent)
                self.assertEqual((program.header.refresh, program.header.placement,
                                  program.header.l_max, program.header.boot_out_level), ('agent', 'depth-dp', 5, 5))
                self.assertTrue(check(program, {name: value.shape for name, value in graph.weights.items()}).ok)
                started = time.monotonic()
                result = estimate(parse(text), graph.weights)
                seconds = time.monotonic()-started
                self.assertEqual(result['boots_pred'], boots)
                self.assertEqual(result['counts']['boot'], boots)
                lazy = estimate(parse(text), graph.weights, placement_model='lazy')
                self.assertEqual(lazy['boots_pred'], 461 if deg == 127 else 382)
                self.assertEqual(lazy['counts']['boot'], 461 if deg == 127 else 382)
                self.assertEqual(result['counts']['rot'], 2624)
                if deg == 127:
                    self.assertEqual(result['counts']['mul_cp'], 6016)
                self.assertEqual(result['const_bytes'], result['distinct_consts']*16384*8)
                self.assertLess(seconds, 10)
                print('R6b estimate='+json.dumps(dict(deg=deg, margin=margin, seconds=seconds, **result)), flush=True)

    def test_plan_latency(self):
        table = load_latency('qwen48')
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'plan.json'
            path.write_text(json.dumps(dict(values=[dict(id='x', level=5), dict(id='y', level=1)], execution=[
                dict(op='mul_cc', inputs=['x', 'x'], output='a'),
                dict(op='boot', inputs=['y'], output='b'),
                dict(op='relinearize', inputs=['x'], output='c')
            ])))
            result = plan_latency(path)
        self.assertEqual(result['latency_us'], table['mul_double'][5]+table['bootstrap_single'][1])
        self.assertEqual(result['by_op'], dict(mul_cc=table['mul_double'][5], boot=table['bootstrap_single'][1]))
        self.assertEqual(result['untimed'], dict(relinearize=1))
        self.assertEqual((result['boots'], result['steps']), (1, 3))

    def test_prune(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            removed = ('candidate_trace.mlir', 'constants.npz', 'lowered.mlir', 'lowered.earth.mlir', 'lowered.ckks.mlir')
            retained = ('lowered._hecate_golden.runtime-plan.json', 'compile.log', 'trace.log', 'candidate_trace.py',
                        'candidate_trace.sandbox.py', 'boot-compiler-config.json', 'other.npz', 'other.mlir')
            for name in removed+retained:
                (directory / name).write_text(name)
            (directory / 'lowered._hecate_golden.bundle').mkdir()
            (directory / 'lowered._hecate_golden.bundle/constant.bin').write_text('constant')
            result = toolchain.prune(directory)
            self.assertEqual(set(result['deleted']), set(removed)|{'lowered._hecate_golden.bundle/'})
            self.assertTrue(all(not (directory / name).exists() for name in removed))
            self.assertFalse((directory / 'lowered._hecate_golden.bundle').exists())
            self.assertTrue(all((directory / name).is_file() for name in retained))
            note = (directory / 'PRUNED.txt').read_text()
            self.assertIn('User approved', note)
            self.assertIn(result['ts'], note)
            self.assertTrue(all(name in note for name in removed))
