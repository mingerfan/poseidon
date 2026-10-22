"""Validate captured real compiler outputs; not encrypted-execution tests.

Set POSEIDON_NATIVE_COMPILER_RESULTS to a native_compiler_smoke.py result dir
after running the Poseidon dump command from the native compiler runbook.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest

RESULTS = os.environ.get("POSEIDON_NATIVE_COMPILER_RESULTS")


@unittest.skipUnless(RESULTS, "requires explicitly selected real native compiler outputs")
class NativeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_compilation_scope_and_artifact_integrity(self):
        self.assertEqual(self.report["status"], "compiled")
        self.assertFalse(self.report["python_dsl_tracing_validated"])
        self.assertFalse(self.report["encrypted_execution_validated"])
        self.assertEqual({x["case"] for x in self.report["cases"]}, {"add", "mul_plain"})
        for case in self.report["cases"]:
            self.assertEqual(case["compiler_exit_code"], 0)
            for name, recorded in case["artifacts"].items():
                data = (self.root / case["case"] / name).read_bytes()
                self.assertEqual(len(data), recorded["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), recorded["sha256"])

    def test_fixed_public_constant_payload(self):
        add = (self.root / "add/_hecate_add.cst").read_bytes()
        plain = (self.root / "mul_plain/_hecate_mul_plain.cst").read_bytes()
        self.assertEqual(struct.unpack("<q", add), (0,))
        self.assertEqual(struct.unpack("<qq4d", plain), (1, 4, 1.5, -2.0, 0.25, 3.0))

    def test_characterize_current_modswitch_adapter_blocker(self):
        # Current compatibility failure, NOT a desired permanent runtime rule.
        # Replace this expectation when a verified ModswitchC mapping is added.
        for case in ("add", "mul_plain"):
            report = json.loads((self.root / case / "poseidon-preflight.json").read_text())
            gate = report["execution_gate"]
            self.assertFalse(gate["ok"])
            self.assertTrue(gate["checks"]["artifacts_loaded"])
            self.assertFalse(gate["checks"]["schedule_built"])
            self.assertFalse(gate["checks"]["readiness_ok"])
            opcodes = report["hevm_opcode_summary"]["opcode_counts"]
            modswitch = next(x for x in opcodes if x["name"] == "ModswitchC")
            self.assertEqual(modswitch["opcode"], 4)
            self.assertEqual(modswitch["count"], 1)
            self.assertFalse(modswitch["supported"])
            self.assertFalse(any(x["name"] == "BootstrapC" for x in opcodes))
            self.assertIn("ckks.modswitchc", (self.root / case / "lowered.ckks.mlir").read_text())


if __name__ == "__main__":
    unittest.main()
