"""Exact modular basis fusion: reporting contracts and optional GPU evidence."""
import math
import os
from pathlib import Path
import unittest

from bench.resnet20_s2c_first.summarize_performance import fields, one, summarize
from bench.resnet20_s2c_first.test_summarize_performance import GOOD


def basis_log(enabled, leaf=False):
    flag = str(enabled).lower()
    log = GOOD.replace(" encrypted_logits_q=4",
                       f" relu_basis_fusion={flag} encrypted_logits_q=4")
    plain_calls = (95 if leaf else 665) - (95 if enabled else 0)
    log = log.replace("prefix_multiply_plain_calls=665",
                      f"prefix_multiply_plain_calls={plain_calls}")
    if leaf:
        log = log.replace(" encrypted_logits_q=4", " relu_leaf_fusion=true encrypted_logits_q=4")
        log += ("RELU_LEAF_FUSION enabled=true calls=266 terms=570 adds_removed=304 "
                "exact_checks=0 online_observer=false coefficients_changed=false rescale_changed=false\n")
    for preparation in (True, False):
        prefix = "RELU_BASIS_FUSION_PREPARED" if preparation else "RELU_BASIS_FUSION"
        log += (f"{prefix} enabled={flag} calls={285 if enabled else 0} "
                f"constants={190 if enabled else 0} products={95 if enabled else 0} "
                f"exact_checks={285 if enabled and preparation else 0} "
                f"exact_residues={824311808 if enabled and preparation else 0} "
                "online_observer=false coefficients_changed=false rescale_changed=false\n")
    return log


class BasisFusionReporting(unittest.TestCase):
    def test_independent_leaf_and_basis_switches_and_historical_logs(self):
        self.assertEqual(summarize(GOOD)["relu_fused_basis_calls"], 0)
        for enabled in (False, True):
            for leaf in (False, True):
                result = summarize(basis_log(enabled, leaf))
                self.assertEqual(result["relu_fused_basis_calls"], 285 if enabled else 0)
                self.assertEqual(result["relu_fused_leaf_calls"], 266 if leaf else 0)

    def test_rejects_incomplete_checks_pruning_and_online_observation(self):
        log = basis_log(True)
        for old, new in (("calls=285", "calls=284"),
                         ("constants=190", "constants=189"),
                         ("products=95", "products=94"),
                         ("exact_checks=285", "exact_checks=284"),
                         ("exact_residues=824311808", "exact_residues=0"),
                         ("exact_checks=0", "exact_checks=1"),
                         ("rescale_changed=false", "rescale_changed=true"),
                         ("prefix_multiply_plain_calls=570", "prefix_multiply_plain_calls=0"),
                         (" relu_basis_fusion=true", "")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                summarize(log.replace(old, new))
        for prefix in ("RELU_BASIS_FUSION_PREPARED ", "RELU_BASIS_FUSION "):
            with self.assertRaises(ValueError):
                summarize("\n".join(row for row in log.splitlines() if not row.startswith(prefix)))


ACCURACY = os.environ.get("POSEIDON_RELU_BASIS_ACCURACY_LOG")
OFF = os.environ.get("POSEIDON_RELU_BASIS_OFF_LOG")
ON = os.environ.get("POSEIDON_RELU_BASIS_ON_LOG")


class BasisFusionGpuEvidence(unittest.TestCase):
    @unittest.skipUnless(ACCURACY, "set POSEIDON_RELU_BASIS_ACCURACY_LOG")
    def test_api_basis_nodes_full_relu_and_unchanged_precision_guard(self):
        lines = Path(ACCURACY).read_text().splitlines()
        secret = fields(one(lines, "SECRET "))
        for key, value in {"h": "192", "positive": "96", "negative": "96",
                           "source": "OS_random", "nonzero": "true"}.items():
            self.assertEqual(secret[key], value)
        api = fields(one(lines, "RELU_BASIS_API "))
        for key in ("constants_and_products", "zero_negative_coefficients", "mixed_q_and_scales",
                    "source_alias", "correction_alias", "immutable_inputs"):
            self.assertEqual(api[key], "true")
        self.assertEqual(api["result"], "PASS")
        self.assertEqual(api["checks"], "22")
        self.assertEqual(api["exact_residues"], "83755008")
        self.assertEqual(api["rejected"], "9")
        exact = fields(one(lines, "RELU_BASIS_EXACT "))
        for key, value in {"result": "PASS", "calls": "15", "constants": "10", "products": "5",
                           "exact_checks": "15", "exact_residues": "43384832",
                           "full_relu_residues": "1179648", "same_ciphertext": "true",
                           "same_keys": "true", "output_q": "9"}.items():
            self.assertEqual(exact[key], value)
        self.assertAlmostEqual(float(exact["output_log_scale"]), 40, places=8)
        result = fields(one(lines, "RESULT "))
        final_error = float(fields(one(lines, "CHECK ReLU_final_vs_original_polynomial "))["max_abs"])
        stage_errors = [float(fields(one(lines, f"CHECK {stage} "))["max_abs"])
                        for stage in ("P15_stage1", "P15_stage2", "P27_stage3")]
        self.assertTrue(all(math.isfinite(e) for e in stage_errors + [final_error]))
        self.assertEqual(result["he_tolerance"], "1e-5")
        self.assertEqual(result["actual_Q_consumed"], "22")
        self.assertLessEqual(final_error, 1e-5)
        self.assertEqual(result["final_output"], "PASS")
        guard = "PASS" if max(stage_errors) <= 1e-5 else "FAIL"
        self.assertEqual(result["stage_guard"], guard)
        self.assertEqual(result["precision"], guard)

    @unittest.skipUnless(OFF and ON, "set both ReLU basis full-network A/B logs")
    def test_full_network_and_exact_activity_reduction(self):
        off, on = (summarize(Path(path).read_text()) for path in (OFF, ON))
        self.assertEqual(off["relu_fused_basis_calls"], 0)
        self.assertEqual(on["relu_fused_basis_calls"], 285)
        # Constant correction: (add + sub c0 + copy c1) -> one kernel.
        # Product correction: (add + 2 PMults + 2 subtracts) -> one kernel.
        self.assertEqual(off["activities"] - on["activities"], 19 * (10 * 2 + 5 * 4))
        for path, run in ((OFF, off), (ON, on)):
            secret = fields(one(Path(path).read_text().splitlines(), "SECRET "))
            self.assertEqual(secret["h"], "192")
            self.assertEqual(secret["nonzero"], "true")
            self.assertEqual(run["gpu_prediction"], 3)
            self.assertLessEqual(run["max_logit_error"], 0.1)
            self.assertEqual(run["relu_fused_leaf_calls"], 266)
            self.assertEqual(run["conv_plain_batch_calls"], 480)
            self.assertEqual(run["relu_q_prefix_source_views"], 1463)


if __name__ == "__main__":
    unittest.main()
