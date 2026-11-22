import json
from pathlib import Path
import tempfile
import unittest

from summarize_aligned import add_encoder_variant, runtime_summary


def fixture():
    return dict(completed=True, warmup=1, native_boot_instructions_per_plan=29,
        native_boot_calls_measured=87, native_boot_calls_total_including_warmup=116,
        context={}, actual_q_moduli=[], actual_p_moduli=[], timing_scope='test',
        outputs=[dict(finite_slots=32, nonfinite_slots=0, reference_error_valid=False)],
        runs=[dict(phase=phase, online_execution_ms=value, wall_ms=value + 2, native_boot_calls=29)
              for phase, value in [('warmup', 99), ('measured', 3), ('measured', 1), ('measured', 2)]])


class SummaryTests(unittest.TestCase):
    def summarize(self, report):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            return runtime_summary(path)

    def test_only_completed_measured_samples_are_used(self):
        result = self.summarize(fixture())
        self.assertEqual(result['online_median_ms'], 2)
        self.assertEqual(result['native_boots_measured'], 87)
        self.assertIsNone(result['max_reference_error'])

    def test_incomplete_attempt_has_no_inferred_latency(self):
        report = fixture()
        report['completed'] = False
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            self.summarize(report)

    def test_instrumented_ranges_and_multi_plan_diagnostics_are_not_formal(self):
        for kind in ('root', 'run', 'explicit_nonsteady'):
            report = fixture()
            if kind == 'root':
                report['instrumented_diagnostic'] = True
            elif kind == 'run':
                report['runs'][1]['instrumented_diagnostic'] = True
            else:
                report['runs'][1]['formal_steady_sample'] = False
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'not a formal benchmark'):
                self.summarize(report)

    def test_encoder_variant_has_shape_scope_and_cerium_ratio(self):
        report = fixture()
        report['model'] = dict(scope='encoder', config=dict(tokens=128, hidden=768, heads=12,
                                                          intermediate=3072, layers=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'variant.json'
            path.write_text(json.dumps(report))
            result = dict(cerium=dict(encoder_internal_median_ms=4))
            add_encoder_variant(result, 'v6', path)
            self.assertEqual(result['encoder_variants']['v6']['latency_percent_of_cerium'], 50)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                add_encoder_variant(result, 'v6', path)

    def test_variants_reject_multilayer_sublayer_wrong_shape_and_bool_dimension(self):
        for key, value in [('layers', 12), ('scope', 'ffn-up'), ('tokens', 16), ('layers', True)]:
            report = fixture()
            report['model'] = dict(scope='encoder', config=dict(tokens=128, hidden=768, heads=12,
                                                              intermediate=3072, layers=1))
            if key == 'scope':
                report['model']['scope'] = value
            else:
                report['model']['config'][key] = value
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'wrong.json'
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'full T128/H768 single-block'):
                    add_encoder_variant(dict(cerium=dict(encoder_internal_median_ms=4)), 'bad', path)

    def test_executed_boots_must_match_the_plan(self):
        report = fixture()
        report['runs'][1]['native_boot_calls'] = 28
        with self.assertRaisesRegex(ValueError, 'Boot count differs'):
            self.summarize(report)

    def test_cold_sample_is_not_called_warm_steady_state(self):
        report = fixture()
        report['warmup'] = 0
        report['runs'] = [report['runs'][1]]
        report['native_boot_calls_measured'] = 29
        report['native_boot_calls_total_including_warmup'] = 29
        result = self.summarize(report)
        self.assertEqual(result['measurement_class'], 'single cold iteration')


if __name__ == '__main__':
    unittest.main()
