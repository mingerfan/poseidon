"""Optional GPU evidence tests; equivalence does not waive the HE stage guard."""
import math
import os
from pathlib import Path
import unittest

from bench.resnet20_s2c_first.summarize_performance import fields, one, summarize


ACCURACY_LOG = os.environ.get("POSEIDON_RELU_LEAF_ACCURACY_LOG")
OFF_LOG = os.environ.get("POSEIDON_RELU_LEAF_OFF_LOG")
ON_LOG = os.environ.get("POSEIDON_RELU_LEAF_ON_LOG")


@unittest.skipUnless(ACCURACY_LOG, "set POSEIDON_RELU_LEAF_ACCURACY_LOG after GPU validation")
class ReluLeafExactEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = Path(ACCURACY_LOG).read_text().splitlines()

    def test_nonzero_secret_and_adversarial_api_cases(self):
        secret = fields(one(self.lines, "SECRET "))
        for key, value in {"h": "192", "positive": "96", "negative": "96",
                           "source": "OS_random", "nonzero": "true"}.items():
            self.assertEqual(secret[key], value)
        api = fields(one(self.lines, "RELU_LEAF_API "))
        self.assertEqual(api["result"], "PASS")
        self.assertEqual(api["term_counts"], "1,2,3,4")
        self.assertEqual(api["rejected"], "9")
        for key in ("zero_and_negative_coefficients", "mixed_q_prefixes",
                    "immutable_inputs", "alias_safe"):
            self.assertEqual(api[key], "true")

    def test_leaf_and_full_relu_are_bitwise_identical(self):
        exact = fields(one(self.lines, "RELU_LEAF_EXACT "))
        expected = {"result": "PASS", "leaf_checks": "14",
                    "leaf_residues": "43122688", "full_relu_residues": "1179648",
                    "coefficient_terms": "30", "same_ciphertext": "true",
                    "same_keys": "true", "output_q": "9"}
        for key, value in expected.items():
            self.assertEqual(exact[key], value)
        self.assertAlmostEqual(float(exact["output_log_scale"]), 40, places=8)

    def test_final_precision_and_stage_failures_remain_visible(self):
        result = fields(one(self.lines, "RESULT "))
        final = fields(one(self.lines, "CHECK ReLU_final_vs_original_polynomial "))
        self.assertEqual(result["he_tolerance"], "1e-5")
        self.assertEqual(result["actual_Q_consumed"], "22")
        self.assertEqual(result["final_output"], "PASS")
        self.assertLessEqual(float(final["max_abs"]), 1e-5)
        stage_errors = [float(fields(one(self.lines, f"CHECK {name} "))["max_abs"])
                        for name in ("P15_stage1", "P15_stage2", "P27_stage3")]
        self.assertTrue(all(math.isfinite(error) for error in stage_errors))
        expected_guard = "PASS" if max(stage_errors) <= 1e-5 else "FAIL"
        self.assertEqual(result["stage_guard"], expected_guard)
        self.assertEqual(result["precision"], expected_guard)


@unittest.skipUnless(OFF_LOG and ON_LOG, "set both ReLU leaf full-network A/B logs")
class ReluLeafNetworkEvidence(unittest.TestCase):
    def test_continuous_graphs_and_exact_kernel_count_reduction(self):
        off = summarize(Path(OFF_LOG).read_text())
        on = summarize(Path(ON_LOG).read_text())
        self.assertEqual(off["relu_fused_leaf_calls"], 0)
        self.assertEqual(on["relu_fused_leaf_calls"], 266)
        self.assertEqual(on["relu_fused_leaf_terms"], 570)
        # Per ReLU: 30 two-component PMults + 16 fused adds -> 14 CAccums.
        self.assertEqual(off["activities"] - on["activities"], 19 * (30 * 2 + 16 - 14))
        for run in (off, on):
            self.assertEqual(run["gpu_prediction"], 3)
            self.assertLessEqual(run["max_logit_error"], 0.1)
            self.assertEqual(run["relu_q_prefix_source_views"], 1463)
            self.assertEqual(run["relu_materialized_moddrops"], 19)


if __name__ == "__main__":
    unittest.main()
