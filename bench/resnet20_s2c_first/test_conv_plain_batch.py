"""Fail-closed reporting tests and optional real GPU Conv batch evidence."""
import os
from pathlib import Path
import unittest

from bench.resnet20_s2c_first.summarize_performance import fields, one, summarize
from bench.resnet20_s2c_first.test_summarize_performance import GOOD


def batch_log(enabled):
    flag = str(enabled).lower()
    log = GOOD.replace(" encrypted_logits_q=4",
                       f" conv_plain_batch={flag} encrypted_logits_q=4")
    for prepared in (True, False):
        prefix = "CONV_PLAIN_BATCH_PREPARED" if prepared else "CONV_PLAIN_BATCH"
        log += (f"{prefix} enabled={flag} chains={160 if enabled else 0} "
                f"terms={1440 if enabled else 0} calls={480 if enabled else 0} "
                f"exact_checks={480 if enabled and prepared else 0} "
                f"exact_residues={566231040 if enabled and prepared else 0} "
                "pending=0 online_observer=false coefficients_changed=false "
                "rescale_changed=false\n")
    return log


class ConvBatchReporting(unittest.TestCase):
    def test_enabled_disabled_and_historical_logs(self):
        self.assertEqual(summarize(GOOD)["conv_plain_batch_calls"], 0)
        for enabled in (False, True):
            result = summarize(batch_log(enabled))
            self.assertEqual(result["conv_plain_batch_terms"], 1440 if enabled else 0)
            self.assertEqual(result["conv_plain_batch_calls"], 480 if enabled else 0)

    def test_rejects_missing_terms_incomplete_checks_and_online_observation(self):
        log = batch_log(True)
        for old, new in (("terms=1440", "terms=1439"),
                         ("exact_checks=480", "exact_checks=479"),
                         ("exact_residues=566231040", "exact_residues=566231039"),
                         ("exact_checks=0", "exact_checks=1"),
                         ("pending=0", "pending=1"),
                         ("coefficients_changed=false", "coefficients_changed=true"),
                         ("rescale_changed=false", "rescale_changed=true"),
                         (" conv_plain_batch=true", "")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                summarize(log.replace(old, new))
        for prefix in ("CONV_PLAIN_BATCH_PREPARED ", "CONV_PLAIN_BATCH "):
            missing = "\n".join(row for row in log.splitlines() if not row.startswith(prefix))
            with self.assertRaises(ValueError):
                summarize(missing)


EXACT_LOG = os.environ.get("POSEIDON_CONV_BATCH_EXACT_LOG")
OFF_LOG = os.environ.get("POSEIDON_CONV_BATCH_OFF_LOG")
ON_LOG = os.environ.get("POSEIDON_CONV_BATCH_ON_LOG")


class ConvBatchGpuEvidence(unittest.TestCase):
    @unittest.skipUnless(EXACT_LOG, "set POSEIDON_CONV_BATCH_EXACT_LOG after GPU validation")
    def test_adversarial_exact_equivalence(self):
        lines = Path(EXACT_LOG).read_text().splitlines()
        secret = fields(one(lines, "SECRET "))
        self.assertEqual(secret["h"], "192")
        self.assertEqual(secret["positive"], "96")
        self.assertEqual(secret["negative"], "96")
        self.assertEqual(secret["nonzero"], "true")
        self.assertEqual(secret["source"], "OS_random")
        exact = fields(one(lines, "CONV_PLAIN_BATCH_EXACT "))
        for key in ("capture_and_replay", "zero_negative_masks", "copy_on_write",
                    "mixed_q_prefixes", "alias_safe", "same_ciphertext"):
            self.assertEqual(exact[key], "true")
        self.assertEqual(exact["result"], "PASS")
        self.assertEqual(exact["lengths"], "1..10")
        self.assertEqual(exact["checks"], "65")
        self.assertEqual(exact["exact_residues"], str(65 * 9 * 65536 * 2))
        self.assertEqual(exact["rejected"], "4")

    @unittest.skipUnless(OFF_LOG and ON_LOG, "set both Conv batch full-network A/B logs")
    def test_full_network_inventory_accuracy_and_kernel_count(self):
        for path in (OFF_LOG, ON_LOG):
            secret = fields(one(Path(path).read_text().splitlines(), "SECRET "))
            for key, value in {"h": "192", "positive": "96", "negative": "96",
                               "source": "OS_random", "nonzero": "true"}.items():
                self.assertEqual(secret[key], value)
        off = summarize(Path(OFF_LOG).read_text())
        on = summarize(Path(ON_LOG).read_text())
        self.assertEqual(off["conv_plain_batch_calls"], 0)
        self.assertEqual(on["conv_plain_batch_calls"], 480)
        self.assertEqual(on["conv_plain_batch_terms"], 1440)
        # Per group: one two-component PMult + eight CAccums -> three batches.
        self.assertEqual(off["activities"] - on["activities"], 160 * (2 + 8 - 3))
        for run in (off, on):
            self.assertEqual(run["gpu_prediction"], 3)
            self.assertLessEqual(run["max_logit_error"], 0.1)
            self.assertEqual(run["relu_fused_leaf_calls"], 266)


if __name__ == "__main__":
    unittest.main()
