"""Audit actual Poseidon native outputs; metadata-only tests are not GPU evidence."""
import hashlib
import json
import math
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get("POSEIDON_MODSWITCH_SEMANTICS_RESULTS"), "Requires native Poseidon primitive/convention results")
class ModswitchEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["POSEIDON_MODSWITCH_SEMANTICS_RESULTS"])
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_hashes_and_native_success_without_gpu_claim(self):
        report = self.report
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["gpu_executed"])
        self.assertFalse(report["hevm_execution"])
        self.assertEqual(report["agent_calls"], 0)
        for path, expected in report["source_hashes"].items():
            target = (self.root / path).resolve()
            self.assertTrue(target.is_relative_to(self.root.resolve()))
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), expected)
        for name, value in report["checks"].items():
            self.assertEqual(value["exit_code"], 0)
            for suffix in ("stdout", "stderr"):
                raw = (self.root / (name+"."+suffix)).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), value[suffix+"_sha256"])
        self.assertEqual(json.loads((self.root / "primitive.stdout").read_text()),
                         report["checks"]["primitive"]["result"])
        self.assertIn("mgpu HEVM plaintext encoding tests passed",
                      (self.root / "level_convention.stdout").read_text())

    def test_all_physical_q_prefixes_aliases_scale_and_decode(self):
        data = self.report["checks"]["primitive"]["result"]
        self.assertEqual(data["backend"], "poseidon_software_primitive")
        self.assertEqual(data["security_request"], "tc128")
        self.assertEqual(data["degree"], 16384)
        self.assertEqual(data["slots"], 8192)
        self.assertEqual(data["scale"], 2**40)
        self.assertEqual([q.bit_length() for q in data["q_moduli"]], [48]*8)
        self.assertEqual([p.bit_length() for p in data["p_moduli"]], [50])
        self.assertEqual(data["source_q_count"], 8)
        self.assertEqual(data["source_poseidon_level"], 7)
        self.assertEqual(len(data["cases"]), 8)
        original = data["cases"][0]
        for index, row in enumerate(data["cases"]):
            self.assertEqual(row["drop_count"], index)
            self.assertEqual(row["target_q_count"], 8-index)
            self.assertEqual(row["poseidon_level"], 7-index)
            for field in ("prefix_exact", "alias_matches", "sequential_matches", "scale_unchanged", "ntt_unchanged"):
                self.assertTrue(row[field])
            self.assertTrue(math.isfinite(row["max_absolute_error"]))
            self.assertLess(row["max_absolute_error"], data["atol"])
            self.assertEqual(row["mae"], original["mae"])
            self.assertEqual(row["max_absolute_error"], original["max_absolute_error"])
        self.assertTrue(data["rescale_changes_scale"])
        self.assertTrue(data["empty_rejected"])
        self.assertNotEqual(data["rescaled_scale"], data["scale"])
        self.assertAlmostEqual(data["rescaled_scale"], data["scale"]/data["q_moduli"][-1], places=15)


if __name__ == "__main__":
    unittest.main()
