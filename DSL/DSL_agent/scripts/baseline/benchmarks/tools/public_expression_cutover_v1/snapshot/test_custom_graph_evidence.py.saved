"""Opt-in checks of recorded REAL encrypted runs, separate from static tests."""
import hashlib
import json
import os
from pathlib import Path
import unittest


class CustomEncryptedEvidenceTests(unittest.TestCase):
    def report(self, variable):
        value = os.environ.get(variable)
        if not value:
            self.skipTest("set " + variable + " to a real execution evidence directory")
        folder = Path(value)
        report = json.loads((folder / "report.json").read_text())
        self.assertEqual(report["agent_calls"], 0)
        self.assertEqual(report["backend"], "upstream_SEAL_HEVM_CPU")
        self.assertFalse(report["poseidon_gpu_validated"])
        return folder, report

    def assert_frozen(self, folder, hashes):
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)

    def test_custom_rule_graph_real_execution(self):
        folder, report = self.report("POSEIDON_CUSTOM_RULE_RESULTS")
        self.assertEqual(report["status"], "passed")
        case = report["cases"][0]
        self.assertEqual(case["descriptor"]["schema"], 2)
        self.assertTrue(case["execution"]["encrypted_execution"])
        self.assertEqual(case["execution"]["input_batches"], 4)
        self.assertFalse(case["execution"]["bootstrap_executed"])
        self.assertEqual(case["comparison"]["compared_values"], 8)
        self.assertTrue(case["comparison"]["passed"])
        self.assertEqual(case["artifact_gate"]["opcode_counts"]["2"], 1)
        self.assert_frozen(folder / case["folder"], case["frozen_hashes"])

    def test_manual_golden_real_isolated_execution(self):
        folder, report = self.report("POSEIDON_CUSTOM_GOLDEN_RESULTS")
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["llm_generation_validated"])
        self.assertIn("manual golden", report["scenario"])
        attempt = report["attempts"][0]
        self.assertTrue(attempt["checked"] and attempt["compiled"] and attempt["executed"])
        self.assertTrue(attempt["execution"]["encrypted_execution"])
        self.assertTrue(attempt["comparison"]["passed"])
        self.assert_frozen(folder, report["frozen_hashes"])
        self.assert_frozen(folder / "attempt-00/output", attempt["artifact_hashes"])

    def test_wrong_reduction_fails_after_real_execution(self):
        folder, report = self.report("POSEIDON_CUSTOM_COUNTEREXAMPLE_RESULTS")
        self.assertNotEqual(report["status"], "passed")
        attempt = report["attempts"][0]
        self.assertTrue(attempt["parsed"] and attempt["checked"] and attempt["compiled"] and attempt["executed"])
        self.assertTrue(attempt["execution"]["encrypted_execution"])
        self.assertEqual(attempt["failure_layer"], "numerical_comparison")
        self.assertFalse(attempt["comparison"]["passed"])
        self.assertEqual(attempt["category"], "candidate")
        self.assert_frozen(folder, report["frozen_hashes"])


if __name__ == "__main__":
    unittest.main()
