"""Independent real-polynomial and physical schedule tests, never GPU claims."""
import math
import unittest
import numpy as np

from cerium_lowering import CompactGraph, MultiPrimeLowering
from cerium_packing import Config, _Arithmetic, build_model, create_weights, pack_inputs
from equivalent_rewrite import rewrite_compact_graph
from polynomial_rewrite import reassociate_polynomials
from test_cerium_lowering import CONTEXT, PROFILE, check_arithmetic


class PolynomialRewriteTests(unittest.TestCase):
    def cubic(self, graph, x, *, coefficient=None):
        scalar = lambda value: graph.plain(np.full(graph.slots, value))
        square, cube = x * x, (x * x) * x
        return x * scalar(-.5) + scalar(1) + square * scalar(3 / 8) - cube * (
            scalar(5 / 16) if coefficient is None else coefficient)

    def test_exact_existing_polynomials_match_independent_coefficients(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        coefficient = 1 / math.sqrt(2 * math.pi)
        for name, factory, coeffs, expected in (
            ('exp', lambda g, x: _Arithmetic(g, config).exponential(x), [1, 1, .5, 1 / 6, 1 / 24], 'exp4_balanced'),
            ('gelu', lambda g, x: _Arithmetic(g, config).gelu(x), [0, .5, coefficient, 0, -coefficient / 6], 'gelu4_factored'),
            ('rsqrt', lambda g, x: self.cubic(g, x), [1, -.5, 3 / 8, -5 / 16], 'rsqrt3_horner')):
            with self.subTest(name=name):
                graph = CompactGraph(32)
                x = graph.input('x')
                output = factory(graph, x)
                target, outputs, stats = reassociate_polynomials(graph, [output])
                data = np.linspace(-1.1, 1.3, 32)
                actual = target.evaluate(outputs, dict(x=data))[0]
                reference = np.polynomial.polynomial.polyval(data, coeffs)
                np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
                self.assertEqual(stats['rewrite_counts'], {expected: 1})
                old, new = [MultiPrimeLowering(g, CONTEXT, PROFILE, initial_level=12,
                    normalize_shared_rotations=True) for g in (graph, target)]
                old_result, new_result = old.lower([output]), new.lower(outputs)
                check_arithmetic(self, old);check_arithmetic(self, new)
                self.assertEqual(old_result[0].level, new_result[0].level)
                self.assertEqual(sum(r.get('op') == 'mul_cc' for r in old.execution),
                                 sum(r.get('op') == 'mul_cc' for r in new.execution))
                self.assertLess(sum(r.get('op') == 'rescale' for r in new.execution),
                                sum(r.get('op') == 'rescale' for r in old.execution))

    def test_short_masks_and_changed_coefficients_are_not_recognized(self):
        for values in (np.full(8, 5 / 16), np.full(32, .31)):
            graph = CompactGraph(32)
            output = self.cubic(graph, graph.input('x'), coefficient=graph.plain(values))
            _, _, stats = reassociate_polynomials(graph, [output])
            self.assertEqual(stats['rewrite_counts'], {})

    def test_externally_used_powers_do_not_add_horner_cipher_products(self):
        graph = CompactGraph(32)
        x = graph.input('x')
        square, cube = x * x, (x * x) * x
        output = self.cubic(graph, x)
        for extra in (square, cube):
            _, _, stats = reassociate_polynomials(graph, [output, extra])
            self.assertNotIn('rsqrt3_horner', stats['rewrite_counts'])
        fourth = square * square
        gelu = _Arithmetic(graph, Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)).gelu(x)
        _, _, stats = reassociate_polynomials(graph, [gelu, fourth])
        self.assertNotIn('gelu4_factored', stats['rewrite_counts'])

    def test_shared_quadratic_partial_sum_cannot_keep_old_square_plus_horner(self):
        graph = CompactGraph(32)
        x = graph.input('x')
        output = self.cubic(graph, x)
        prefix = output.args[0]
        for extra in (prefix, prefix + graph.plain(np.full(32, .125))):
            source_outputs = [output, extra]
            target, outputs, stats = reassociate_polynomials(graph, source_outputs)
            self.assertNotIn('rsqrt3_horner', stats['rewrite_counts'])
            for g, values in ((graph, source_outputs), (target, outputs)):
                reachable = g.reachable(values)
                cc = sum(node.op == 'mul' and all(a.op != 'plain' for a in node.args)
                         for node in reachable)
                self.assertEqual(cc, 2)
                lowerer = MultiPrimeLowering(g, CONTEXT, PROFILE, initial_level=12)
                lowerer.lower(values)
                check_arithmetic(self, lowerer)
                self.assertEqual(sum(r.get('op') == 'mul_cc' for r in lowerer.execution), 2)

    def test_two_layers_dense_qk_newton_and_boot_labels_preserved(self):
        config = Config(tokens=4, hidden=24, heads=12, tile=8, slots=32, intermediate=48, layers=2)
        source = CompactGraph(32, prepared_qk=True)
        outputs = build_model(source, [source.input(str(i)) for i in range(3)], create_weights(config, 32), config)
        source, outputs, _ = rewrite_compact_graph(source, outputs, tokens=4)
        target, rewritten, stats = reassociate_polynomials(source, outputs)
        self.assertEqual(stats['rewrite_counts'], dict(exp4_balanced=12, gelu4_factored=12, rsqrt3_horner=4))
        packed = pack_inputs(np.random.default_rng(43).normal(0, .2, (4, 24)), config)
        data = {str(i): value for i, value in enumerate(packed)}
        np.testing.assert_allclose(source.evaluate(outputs, data), target.evaluate(rewritten, data), rtol=1e-11, atol=1e-12)
        self.assertEqual(target.qk_specs, source.qk_specs)
        self.assertEqual([n.data for n in source.reachable(outputs) if n.op == 'boot'],
                         [n.data for n in target.reachable(rewritten) if n.op == 'boot'])
        for label, matrix in source.matrices.items():
            self.assertEqual(matrix['weight'].tobytes(), target.matrices[label]['weight'].tobytes())
            self.assertIsNot(matrix['weight'], target.matrices[label]['weight'])
        old, new = [MultiPrimeLowering(g, CONTEXT, PROFILE, initial_level=12,
                    normalize_shared_rotations=True) for g in (source, target)]
        a, b = old.lower(outputs), new.lower(rewritten)
        check_arithmetic(self, old);check_arithmetic(self, new)
        self.assertEqual([v.level for v in a], [v.level for v in b])
        self.assertEqual(sum(r.get('op') == 'mul_cc' for r in old.execution),
                         sum(r.get('op') == 'mul_cc' for r in new.execution))
        self.assertLessEqual(sum(r.get('op') == 'boot' for r in new.execution),
                             sum(r.get('op') == 'boot' for r in old.execution))

    def test_disabled_flags_and_foreign_outputs(self):
        graph = CompactGraph(32)
        x = graph.input('x')
        output = self.cubic(graph, x)
        target, outputs, stats = reassociate_polynomials(graph, [output], exponential=False, gelu=False, rsqrt=False)
        self.assertEqual(stats['rewrite_counts'], {})
        self.assertEqual(len(graph.reachable([output])), len(target.reachable(outputs)))
        with self.assertRaises(ValueError):
            reassociate_polynomials(graph, [CompactGraph(32).input('foreign')])


if __name__ == '__main__':
    unittest.main()
