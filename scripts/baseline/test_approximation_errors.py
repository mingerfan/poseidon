import hashlib
import json
import math
import os
from pathlib import Path
import unittest

from approximation_errors import decompose, relu_and_quadratic
from hecate_contract import validate_function
from model_graph import evaluate_reference, validate_graph

BASE = Path(__file__).parent
CASE = BASE / "cases/relu-quadratic-explicit.json"
GOLDEN = BASE / "golden_cases/approximation"


class ApproximationTests(unittest.TestCase):
    def test_hand_values_domain_endpoints_and_analytic_bound(self):
        original, polynomial = relu_and_quadratic([-1, -.5, 0, .5, 1])
        self.assertEqual(original, [0, 0, 0, .5, 1])
        self.assertEqual(polynomial, [0, -.125, 0, .375, 1])
        grid = [i/1000 for i in range(-1000, 1001)]
        f, p = relu_and_quadratic(grid)
        self.assertEqual(max(abs(a-b) for a, b in zip(f, p)), .125)
        for value in (-1.001, 1.001, float("nan"), True):
            with self.assertRaises(ValueError):
                relu_and_quadratic([value])

    def test_perfect_polynomial_execution_is_not_original_relu(self):
        f, p = relu_and_quadratic([-.5, .5, 0, 1])
        report = decompose(f, p, p)
        self.assertTrue(report["ckks_passed"])
        self.assertEqual(report["ckks_execution_error"]["max_absolute_error"], 0)
        self.assertEqual(report["approximation_error"]["max_absolute_error"], .125)
        self.assertFalse(report["original_target_threshold_passed"])
        self.assertFalse(report["original_semantic_equivalence_verified"])
        self.assertIsNone(report["approximation_acceptance_budget"])
        self.assertIn("semantic", report["execution_error_interpretation"])

    def test_wrong_implementation_and_signed_error_cancellation_are_visible(self):
        # Approximation and execution errors may cancel; final output alone is insufficient.
        report = decompose([0], [.125], [0])
        self.assertTrue(report["original_target_threshold_passed"])
        self.assertFalse(report["ckks_passed"])
        self.assertEqual(report["approximation_error"]["signed_error"], [.125])
        self.assertEqual(report["ckks_execution_error"]["signed_error"], [-.125])
        self.assertEqual(report["total_error"]["signed_error"], [0])
        self.assertEqual(report["decomposition_residual"], [0])
        self.assertIsNone(report["total_error"]["relative_error"][0])
        self.assertIsNone(report["total_error"]["cosine_similarity"])

    def test_shapes_finite_values_and_tolerance_rejected(self):
        for args in (([], [], []), ([1], [1, 2], [1]), ([1], [1], [float("inf")]),
                     ([True], [1], [1])):
            with self.assertRaises(ValueError):
                decompose(*args)
        for kwargs in (dict(atol=-1), dict(rtol=float("nan")), dict(atol=True)):
            with self.assertRaises(ValueError):
                decompose([1], [1], [1], **kwargs)

    def test_explicit_graph_reference_and_manual_syntax_without_allowing_relu(self):
        data = json.loads(CASE.read_text())
        validate_graph(data)
        xs = [-1, -.5, .5, 1]
        self.assertEqual(evaluate_reference(data, xs), relu_and_quadratic(xs)[1])
        for name in ("quadratic", "quadratic-wrong-sign"):
            source = (GOLDEN / (name+".py")).read_text()
            check = validate_function(source, {"c0": [.5]}, contract="hecate-function-v1")
            self.assertTrue(check["syntax_type_layout_checked"])
        source = '@hc.func("c")\ndef golden(x):\n    return x.relu()\n'
        with self.assertRaises(ValueError):
            validate_function(source, {}, contract="hecate-function-v1")


@unittest.skipUnless(os.environ.get("POSEIDON_APPROXIMATION_RESULTS"), "Requires real approximation/counterexample execution")
class ApproximationEvidenceTests(unittest.TestCase):
    def test_two_real_candidates_and_three_independent_output_comparisons(self):
        root = Path(os.environ["POSEIDON_APPROXIMATION_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["agent_calls"], 0)
        self.assertFalse(report["poseidon_gpu_validated"])
        self.assertFalse(report["original_semantic_equivalence_verified"])
        self.assertEqual(len(report["cases"]), 2)
        for path, expected in report["source_hashes"].items():
            snapshot = (root / report["source_snapshots"][path]).resolve()
            self.assertTrue(snapshot.is_relative_to(root.resolve()))
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), expected)
        for row in report["cases"]:
            self.assertTrue(row["matched_expected"])
            self.assertTrue(row["execution"]["encrypted_execution"])
            self.assertEqual(row["execution"]["input_batches"], 4)
            self.assertFalse(row["execution"]["bootstrap_executed"])
            self.assertNotIn("10", row["artifact_gate"]["opcode_counts"])
            f, p = relu_and_quadratic([x for batch in row["inputs"] for x in batch])
            errors = row["errors"]
            self.assertEqual(errors, decompose(f, p, errors["decrypted"]))
            self.assertEqual(errors["compared_values"], 16)
            self.assertEqual(errors["approximation_error"]["max_absolute_error"], .125)
            self.assertFalse(errors["original_target_threshold_passed"])
            self.assertEqual(errors["ckks_passed"], not row["counterexample"])
            self.assertLess(max(abs(v) for v in errors["decomposition_residual"]), 1e-14)
            run = Path(row["run"])
            self.assertEqual(hashlib.sha256((run / "report.json").read_bytes()).hexdigest(), row["candidate_report_sha256"])
            self.assertEqual(hashlib.sha256((run / "attempt-00/output/decrypted.npy").read_bytes()).hexdigest(), row["decrypted_sha256"])
            if row["counterexample"]:
                self.assertEqual(row["failure_layer"], "numerical_comparison")


if __name__ == "__main__":
    unittest.main()
