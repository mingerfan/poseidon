"""Small analytic/protocol checks; no original logit or model data."""

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_monotone_selector as monotone


class MonotoneSelectorCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = monotone.base._numeric()
        from pt_agent.kernels import encrypted_generation
        cls.selector = encrypted_generation

    def fixture(self, root):
        np = self.np
        own = root / 'own/result.json'
        own.parent.mkdir()
        (own.parent / 'worker').mkdir()
        rows = []
        for index in range(6):
            filename = 'own-output-' + str(index) + '-logits.npy'
            np.save(own.parent / 'worker' / filename,
                np.asarray([.9, -.8, .4, -.5, .2], dtype=np.float64).reshape(1, 1, 5), allow_pickle=False)
            rows.append(dict(sample_id='train:' + str(index // 2), step=index % 2,
                logits_file=filename, original_greedy_token=0, approximate_greedy_token=0))
        monotone.base.save(own, dict(status='ok', job='quality-own-tensor', worker=dict(status='ok',
            own_output_generation_completed=True, tensor_forward_completed=6, state_update_completed=3,
            candidate_frozen=False, sample_rows=rows)))
        prepare = root / 'prepare'
        prepare.mkdir()
        sources = monotone.base.source_rows(own)
        artifact = monotone.prepare_artifact(sources, np)
        monotone.base.save(prepare / 'coefficients.json', artifact)
        monotone.base.save(prepare / 'result.json', dict(status='ok', phase='prepare',
            config=monotone.config(), source_rows=sources, own_output_result=str(own), budget_started_ns=time.perf_counter_ns()))
        out = root / 'evaluate'
        out.mkdir()
        return SimpleNamespace(out=out, phase='evaluate', own_result=own,
            coefficients=prepare / 'coefficients.json', supervisor_pid=os.getpid(), budget_started_ns=time.perf_counter_ns()), artifact

    def test_exact_cubic_algebra_boundaries_monotonicity_and_sign_symmetry(self):
        np = self.np
        stages = monotone.analytic_coefficients(np)
        self.assertEqual(len(stages), 52)
        self.assertTrue(all(not value.flags.writeable for value in stages))
        np.testing.assert_array_equal(stages[0], [0., 1.125, 0., -.125])
        np.testing.assert_array_equal(stages[-1], [0., .5625, 0., -.0625])
        grid = np.linspace(-1., 1., 4097)
        polynomial = np.polynomial.chebyshev.chebval(grid, stages[0])
        algebraic = (3. * grid - grid ** 3) / 2.
        self.assertTrue(np.all(np.abs(polynomial - algebraic) <= 1e-12 + 1e-12 * np.abs(algebraic)))
        self.assertTrue(np.all(np.diff(polynomial) >= 0))
        self.assertGreaterEqual(float(polynomial.min()), -1.)
        self.assertLessEqual(float(polynomial.max()), 1.)
        np.testing.assert_array_equal(polynomial, -polynomial[::-1])
        np.testing.assert_array_equal(np.polynomial.chebyshev.chebval(np.asarray([-1., 0., 1.]), stages[0]), [-1., 0., 1.])

    def test_analytic_artifact_roundtrip_and_final_half_factor_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            args, artifact = self.fixture(Path(directory))
            stages, loaded, _ = monotone.load_artifact(args.coefficients, args.own_result, self.np)
            self.assertEqual(loaded, artifact)
            self.assertEqual(len(stages), 52)
            self.assertEqual(artifact['coefficient_fits'], 0)
            self.assertGreater(artifact['scalar_gap_observations'][49]['value'], .9999999)
            changed = copy.deepcopy(artifact)
            changed['stages'][-1][1] *= 2.
            monotone.base.save(args.coefficients, changed)
            with self.assertRaises(ValueError):
                monotone.load_artifact(args.coefficients, args.own_result, self.np)

    def test_observer_retains_original_arithmetic_and_all_fifty_two_stages(self):
        np = self.np
        values = np.asarray([.9, -.8, .4, -.5, .2], dtype=np.float64)
        coefficients = monotone.analytic_coefficients(np)
        before = np.polynomial.chebyshev.chebval
        reference, contract = self.selector.numpy_selection(values, bound=32., normalized_gap=monotone.base.GAP,
                                                            coefficients=coefficients)
        rows = []
        with monotone.observe_rounds(np, len(values), rows, lambda: None):
            observed, measured = self.selector.numpy_selection(values, bound=32., normalized_gap=monotone.base.GAP,
                                                               coefficients=coefficients)
        self.assertIs(np.polynomial.chebyshev.chebval, before)
        np.testing.assert_array_equal(observed, reference)
        self.assertEqual(contract, measured)
        self.assertEqual(sum(row['actual_pairs'] for row in rows), 4)
        self.assertTrue(all(len(row['stages']) == 52 for row in rows))
        self.assertEqual(int(observed.argmax()), int(values.argmax()))

    def test_actual_tie_is_observed_and_keeps_domain_rejection(self):
        np = self.np
        values = np.asarray([.5, .5], dtype=np.float64)
        rows = []
        with monotone.observe_rounds(np, len(values), rows, lambda: None):
            selected, contract = self.selector.numpy_selection(values, bound=32., normalized_gap=monotone.base.GAP,
                coefficients=monotone.analytic_coefficients(np))
        np.testing.assert_array_equal(selected, [.5, .5])
        self.assertFalse(contract['comparison_gap_contract_passed'])
        self.assertEqual(rows[0]['comparisons_below_gap'], 1)
        self.assertEqual(monotone.base.one_hot_quality(values, selected, np)['max_absolute_one_hot_error'], .5)

    def test_six_synthetic_rows_use_actual_vectors_without_fit_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            with mock.patch.object(monotone.base, 'VOCABULARY', 5), \
                    mock.patch.object(monotone.base, 'resident_guard', return_value=nullcontext()):
                result = monotone.worker(args)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual((result['selection_attempts'], result['selection_completed']), (6, 6))
            self.assertEqual(result['coefficient_fits'], 0)
            self.assertTrue(result['greedy_policy_passed'])
            self.assertTrue(all(row['real_comparisons_observed'] == 4 for row in result['rows']))
            for row in result['rows']:
                selected = self.np.load(args.out / row['one_hot_file'], allow_pickle=False)
                self.assertEqual(int(selected.argmax()), row['quality']['actual_selected_token'])
            self.assertIsNone(result['selector_one_hot_quality_accepted'])
            self.assertTrue(all(result[key] == 0 for key in monotone.base.ZERO_FIELDS))

    def test_first_runtime_error_stops_and_preserves_original_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            original = self.np.polynomial.chebyshev.chebval
            with mock.patch.object(monotone.base, 'VOCABULARY', 5), \
                    mock.patch.object(monotone.base, 'resident_guard', return_value=nullcontext()), \
                    mock.patch.object(self.selector, 'numpy_selection', side_effect=FloatingPointError('planned')) as call:
                result = monotone.worker(args)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(result['status'], 'error')
            self.assertEqual((result['selection_attempts'], result['selection_completed']), (1, 0))
            self.assertIs(self.np.polynomial.chebyshev.chebval, original)
            self.assertIsNone(result['CKKS_latency'])


if __name__ == '__main__':
    unittest.main()
