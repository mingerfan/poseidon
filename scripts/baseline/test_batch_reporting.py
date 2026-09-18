import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from run_model_batch import summarize


class ReportingTests(unittest.TestCase):
    def test_running_extended_batch_keeps_full_planned_denominator(self):
        items = [dict(status="passed", compiled=True)] * 64
        self.assertEqual(summarize(items, 96)["metrics"]["compiled"],
                         dict(numerator=64, denominator=96, rate=64/96))
        self.assertEqual(summarize([], 96)["metrics"]["compiled"]["denominator"], 96)
        with self.assertRaises(ValueError):
            summarize(items, 63)

    def test_denominator_keeps_unsupported_and_infrastructure_failures(self):
        items = [dict(status="passed", translated=True, traced=True, compiled=True, executed=True, numerically_correct=True),
                 dict(status="failed", failure_layer="fx_translation", failure_category="input"),
                 dict(status="not_run", failure_layer="environment_or_key_setup", failure_category="infrastructure")]
        report = summarize(items)
        for metric in report["metrics"].values():
            self.assertEqual(metric, dict(numerator=1, denominator=3, rate=1/3))
        self.assertEqual(report["agent_calls"], 0)
        self.assertEqual(report["failure_categories"], {"input": 1, "infrastructure": 1})

    def test_partial_pipeline_is_not_numerical_success(self):
        summary = summarize([dict(status="failed", translated=True, compiled=True, failure_layer="seal_runtime")])
        self.assertEqual(summary["metrics"]["compiled"]["numerator"], 1)
        self.assertEqual(summary["metrics"]["numerically_correct"]["numerator"], 0)
        self.assertIsNone(summarize([])["metrics"]["compiled"]["rate"])


RESULTS = os.environ.get("POSEIDON_FX_BATCH_RESULTS")


@unittest.skipUnless(RESULTS, "requires selected real 48-case rule-translator run")
class BatchEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_all_48_and_holdout_accounting(self):
        report = self.report
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 48)
        self.assertEqual(sum(c["holdout_for_agent"] for c in report["cases"]), 12)
        self.assertEqual(report["summary"], summarize(report["cases"]))
        self.assertEqual(report["agent_calls"], 0)
        self.assertFalse(report["poseidon_gpu_validated"])
        for metric in report["summary"]["metrics"].values():
            self.assertEqual(metric["numerator"], 48)
            self.assertEqual(metric["denominator"], 48)

    def test_integrity_actual_execution_and_frozen_tolerances(self):
        self.assertEqual(self.report["parameters"]["modulus_bits"], [60] * 14)
        self.assertEqual(self.report["parameters"]["security_check"], "tc128")
        for case in self.report["cases"]:
            folder = self.root / case["folder"]
            for name, expected in case["frozen_hashes"].items():
                self.assertEqual(digest(folder / name), expected)
            self.assertEqual(case["execution_exit_code"], 0)
            self.assertTrue(case["execution"]["encrypted_execution"])
            self.assertFalse(case["execution"]["bootstrap_executed"])
            comparison = case["comparison"]
            self.assertEqual(comparison["atol"], 1e-5)
            self.assertEqual(comparison["rtol"], 1e-4)
            for actuals, refs in zip(comparison["actual"], comparison["reference"]):
                for actual, ref in zip(actuals, refs):
                    self.assertLessEqual(abs(actual-ref), 1e-5 + 1e-4 * abs(ref))
            self.assertNotIn("10", case["artifact_gate"]["opcode_counts"])
