"""Small synthetic protocol/arithmetic checks for the real-logit diagnostic."""

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import qwen_selector_validation as validation


class SelectorValidationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = validation._numeric()
        from pt_agent.kernels import encrypted_generation
        cls.selector = encrypted_generation

    def coefficients(self):
        result = []
        for degree in validation.DEGREES:
            value = self.np.zeros(degree + 1, dtype=self.np.float64)
            value[1] = .9
            result.append(value)
        result[-1] *= .5
        return tuple(result)

    def reports(self):
        domain = [validation.GAP, 1.]
        rows = []
        for degree in validation.DEGREES:
            rows.append(dict(degree=degree, domain=domain, iterations=1, max_error=.1))
            domain = [.9, 1.1]
        return rows

    def fixture(self, root):
        np = self.np
        own = root / 'own/result.json'
        own.parent.mkdir()
        (own.parent / 'worker').mkdir()
        rows = []
        for index in range(6):
            values = np.asarray([.9, -.8, .4, -.5, .2], dtype=np.float64).reshape(1, 1, 5)
            filename = 'own-output-' + str(index) + '-logits.npy'
            np.save(own.parent / 'worker' / filename, values, allow_pickle=False)
            rows.append(dict(sample_id='train:' + str(index // 2), step=index % 2,
                logits_file=filename, original_greedy_token=0, approximate_greedy_token=0))
        record = dict(status='ok', job='quality-own-tensor', worker=dict(status='ok',
            own_output_generation_completed=True, tensor_forward_completed=6, state_update_completed=3,
            candidate_frozen=False, sample_rows=rows))
        own.write_text(json.dumps(record))
        fit = root / 'fit'
        fit.mkdir()
        artifact = dict(format=1, config=validation.fixed_config(),
            stages=[stage.tolist() for stage in self.coefficients()], reports=self.reports(),
            fit_stage_attempts=3, fit_stage_completed=3, final_stage_half_sign_factor_already_applied=True)
        validation.save(fit / 'coefficients.json', artifact)
        validation.save(fit / 'result.json', dict(status='ok', phase='fit', config=validation.fixed_config(),
            own_output_result=str(own), source_rows=validation.source_rows(own),
            started_ns=time.perf_counter_ns(), budget_started_ns=time.perf_counter_ns()))
        out = root / 'evaluate'
        out.mkdir()
        args = SimpleNamespace(out=out, phase='evaluate', own_result=own,
            coefficients=fit / 'coefficients.json', supervisor_pid=os.getpid(), budget_started_ns=time.perf_counter_ns())
        return args, record, artifact

    def test_source_order_identity_and_original_directory_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, record, _ = self.fixture(Path(directory))
            rows = validation.source_rows(args.own_result)
            self.assertEqual([(row['sample_id'], row['step']) for row in rows],
                             [('train:' + str(i // 2), i % 2) for i in range(6)])
            changed = copy.deepcopy(record)
            changed['worker']['sample_rows'][0]['logits_file'] = '../other.npy'
            args.own_result.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                validation.source_rows(args.own_result)
            for out in (args.own_result.parent, args.own_result.parent / 'new', args.coefficients.parent / 'new'):
                with self.assertRaises(ValueError):
                    validation.check_out(out, args.own_result, args.coefficients)

    def test_fit_uses_each_original_stage_once_and_restores_function_on_failure(self):
        np = self.np
        reports = []
        def exchange(target, degree, *, odd, interval):
            self.assertTrue(odd)
            c = np.zeros(degree + 1, dtype=np.float64)
            c[1] = 1.
            return c, dict(iterations=1, max_error=.1)
        approx = SimpleNamespace(_exchange=exchange)
        def composite(degrees, gap):
            self.assertEqual((degrees, gap), (validation.DEGREES, validation.GAP))
            values = []
            domain = (gap, 1.)
            for degree in degrees:
                c, report = approx._exchange(lambda x: np.ones_like(x), degree, odd=True, interval=domain)
                reports.append(dict(report, domain=list(domain), degree=degree))
                values.append(c)
                domain = (.9, 1.1)
            values[-1] *= .5
            return tuple(values)
        approx.composite_report = lambda degrees, gap: reports
        selector = SimpleNamespace(composite_half_sign=composite)
        result = {}
        with mock.patch.object(validation, '_fit_modules', return_value=(approx, selector)):
            artifact = validation.fit_once(np, result, lambda: None)
        self.assertIs(approx._exchange, exchange)
        self.assertEqual([row['attempted'] for row in result['fit_stages']], [1, 1, 1])
        self.assertEqual([row['completed'] for row in result['fit_stages']], [1, 1, 1])
        self.assertEqual(artifact['stages'][-1][1], .5)
        def fail(*args, **kwargs):
            raise ArithmeticError('planned failure')
        approx._exchange = fail
        with mock.patch.object(validation, '_fit_modules', return_value=(approx, selector)):
            with self.assertRaises(ArithmeticError):
                validation.fit_once(np, {}, lambda: None)
        self.assertIs(approx._exchange, fail)

    def test_round_observer_preserves_arithmetic_and_excludes_odd_carry(self):
        np = self.np
        values = np.asarray([.9, -.8, .4, -.5, .2], dtype=np.float64)
        coefficients = self.coefficients()
        original = np.polynomial.chebyshev.chebval
        reference, contract = self.selector.numpy_selection(values, bound=32., normalized_gap=validation.GAP,
                                                            coefficients=coefficients)
        rows = []
        with validation.observe_rounds(np, len(values), self.reports(), rows, lambda: None):
            observed, observed_contract = self.selector.numpy_selection(values, bound=32., normalized_gap=validation.GAP,
                                                                        coefficients=coefficients)
        self.assertIs(np.polynomial.chebyshev.chebval, original)
        np.testing.assert_array_equal(observed, reference)
        self.assertEqual(observed_contract, contract)
        self.assertEqual([row['width'] for row in rows], [5, 3, 2])
        self.assertEqual(sum(row['actual_pairs'] for row in rows), 4)
        self.assertEqual([row['odd_carry_excluded'] for row in rows], [True, True, False])
        self.assertTrue(any(row['stages'][1]['actual_pairs_outside_stage_domain'] for row in rows))

    def test_round_observer_restores_original_function_on_numerical_error(self):
        np = self.np
        original = np.polynomial.chebyshev.chebval
        huge = np.zeros(16, dtype=np.float64)
        huge[1] = 1e308
        rows = []
        with self.assertRaises(FloatingPointError), np.errstate(over='raise', invalid='raise'):
            with validation.observe_rounds(np, 2, self.reports(), rows, lambda: None):
                self.selector.numpy_selection(np.asarray([16., -16.]), bound=32., normalized_gap=validation.GAP,
                    coefficients=(huge, huge, self.coefficients()[-1]))
        self.assertIs(np.polynomial.chebyshev.chebval, original)
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(len(rows[0]['stages']), 1)

    def test_saved_fit_contract_is_checked_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, artifact = self.fixture(Path(directory))
            coefficients, _, _ = validation.load_coefficients(args.coefficients, args.own_result, self.np)
            self.assertEqual([len(stage) for stage in coefficients], [16, 16, 28])
            self.assertTrue(all(not stage.flags.writeable for stage in coefficients))
            changed = copy.deepcopy(artifact)
            changed['reports'][1]['domain'][0] = .8
            validation.save(args.coefficients, changed)
            with self.assertRaises(ValueError):
                validation.load_coefficients(args.coefficients, args.own_result, self.np)

    def test_actual_vector_quality_is_separate_from_exact_one_hot(self):
        np = self.np
        logits = np.asarray([.1, .5, -.1], dtype=np.float64)
        selected = np.asarray([.1, .8, .1], dtype=np.float64)
        quality = validation.one_hot_quality(logits, selected, np)
        self.assertTrue(quality['token_agreement'])
        self.assertEqual(quality['actual_selected_token'], 1)
        self.assertAlmostEqual(quality['l1_one_hot_error'], .4)
        self.assertIsNone(quality['one_hot_quality_acceptance'])
        self.assertFalse(quality['exact_discrete_one_hot_claimed'])

    def test_six_real_loop_calls_and_first_runtime_error_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            with mock.patch.object(validation, 'VOCABULARY', 5), mock.patch.object(validation, 'resident_guard', return_value=nullcontext()):
                result = validation.worker(args)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual((result['selection_attempts'], result['selection_completed']), (6, 6))
            self.assertTrue(all((args.out / row['one_hot_file']).is_file() for row in result['rows']))
            self.assertEqual(result['fresh_fit_calls'], 0)
            with mock.patch.object(validation, 'VOCABULARY', 5), mock.patch.object(validation, 'resident_guard', return_value=nullcontext()), \
                    mock.patch.object(self.selector, 'numpy_selection', side_effect=FloatingPointError('planned first error')) as call:
                failed = validation.worker(args)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(failed['status'], 'error')
            self.assertEqual((failed['selection_attempts'], failed['selection_completed']), (1, 0))
            self.assertEqual(len(failed['rows']), 1)
            self.assertIsNone(failed['CKKS_latency'])

    def test_resident_guard_is_subtree_bounded_and_restores_timer(self):
        from pt_agent.experiments import qwen_own_output_packed
        original = signal.getsignal(signal.SIGALRM)
        record = {}
        with mock.patch.object(qwen_own_output_packed, '_resident_snapshot', side_effect=(1, 600 * 1024)):
            with self.assertRaises(MemoryError):
                with validation.resident_guard(os.getpid(), record):
                    signal.raise_signal(signal.SIGALRM)
        self.assertTrue(record['stopped'])
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_registry_records_saved_failure_once_with_eight_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'case/evaluate'
            out.mkdir(parents=True)
            validation.save(out / 'result.json', dict(status='error', selection_attempts=1,
                selection_completed=0, error=dict(type='FloatingPointError', message='planned'), fit_stages=[],
                **{key: 0 for key in validation.ZERO_FIELDS}))
            validation.save(out / 'supervision.json', dict(observation=dict(exit_code=1,
                timed_out=False, remaining_pids=[])))
            registry = root / 'registry.jsonl'
            row = validation.record_run(out, 'evaluate', registry, revision='test', registration='saved-result')
            second = validation.record_run(out, 'evaluate', registry, revision='test', registration='saved-result')
            self.assertEqual(row, second)
            self.assertEqual(len(registry.read_text().splitlines()), 1)
            self.assertEqual(set(row), {'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            self.assertEqual(row['metrics']['status'], 'error')
            self.assertEqual(row['metrics']['selection_completed'], 0)
            self.assertIsNone(row['metrics']['greedy_policy_passed'])


if __name__ == '__main__':
    unittest.main()
