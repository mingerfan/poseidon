"""Offline dependency guards and opt-in real Python compiler artifact checks."""
import copy
import json
import os
from pathlib import Path
import struct
import unittest

from hecate_python_env import LOCK, LIMIT, PINS, digest, validate_lock


class PythonWheelLockTests(unittest.TestCase):
    def setUp(self):
        self.lock = json.loads(LOCK.read_text())

    def test_exact_cpu_closure_within_budget(self):
        wheels = validate_lock(self.lock)
        self.assertEqual(len(wheels), 9)
        self.assertLess(sum(w["bytes"] for w in wheels), LIMIT)
        self.assertEqual(next(w for w in wheels if w["name"] == "torch")["version"], "2.0.1+cpu")

    def test_missing_or_extra_dependency_rejected(self):
        for remove in (True, False):
            lock = copy.deepcopy(self.lock)
            if remove:
                lock["wheels"].pop()
            else:
                lock["wheels"].append(dict(name="triton", version="2.0.0"))
            with self.assertRaises(ValueError):
                validate_lock(lock)

    def test_unapproved_version_rejected(self):
        self.lock["wheels"][0]["version"] = "2.0.0"
        with self.assertRaises(ValueError):
            validate_lock(self.lock)

    def test_budget_and_unsafe_download_rejected(self):
        for key, value in (("bytes", LIMIT), ("url", "http://files.pythonhosted.org/x"),
                           ("url", "https://example.org/x"), ("filename", "../x.whl"),
                           ("sha256", "invalid")):
            lock = copy.deepcopy(self.lock)
            lock["wheels"][0][key] = value
            with self.assertRaises(ValueError):
                validate_lock(lock)

    def test_pins_have_upstream_requirements_evidence(self):
        requirements = (LOCK.parents[4] / "third_party/dacapo/requirements.txt").read_text()
        for name, version in PINS.items():
            self.assertIn(f"{name}=={version.removesuffix('+cpu')}", requirements)


RESULTS = os.environ.get("POSEIDON_PYTHON_COMPILER_RESULTS")


@unittest.skipUnless(RESULTS, "requires explicitly selected real Hecate Python compiler artifacts")
class PythonCompilerArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_real_trace_compile_integrity_without_execution_claim(self):
        self.assertTrue(self.report["python_dsl_tracing_validated"])
        self.assertFalse(self.report["encrypted_execution_validated"])
        self.assertEqual({c["case"] for c in self.report["cases"]}, {"add", "mul_plain", "linear4x2"})
        for case in self.report["cases"]:
            self.assertEqual(case["trace_exit_code"], 0)
            self.assertEqual(case["compiler_exit_code"], 0)
            for name, expected in case["artifacts"].items():
                path = self.root / case["case"] / name
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(digest(path), expected["sha256"])
            self.assertEqual(case["comparison"]["status"], "not_run")
            self.assertIsNone(case["comparison"]["mae"])

    def test_frozen_reference_and_non_elementwise_linear(self):
        case = next(c for c in self.report["cases"] if c["case"] == "linear4x2")
        self.assertEqual(case["reference"][0], [0.125, -0.25])
        self.assertEqual(case["reference"][1], [0.5, 0.75])
        self.assertEqual(case["comparison"]["atol"], 1e-5)
        self.assertEqual(case["comparison"]["rtol"], 1e-4)
        earth = (self.root / "linear4x2/trace_golden.mlir").read_text()
        self.assertIn("earth.rotate", earth)
        self.assertIn("earth.mul", earth)
        self.assertIn("earth.add", earth)
        self.assertEqual(earth.count('"earth.rotate"'), 4)

    def test_real_linear_cst_contains_exact_weights_and_bias(self):
        data = (self.root / "linear4x2/_hecate_golden.cst").read_bytes()
        count, = struct.unpack_from("<q", data)
        self.assertEqual(count, 4)
        offset, constants = 8, []
        for _ in range(count):
            size, = struct.unpack_from("<q", data, offset)
            offset += 8
            constants.append(list(struct.unpack_from(f"<{size}d", data, offset)))
            offset += 8 * size
        self.assertEqual(offset, len(data))
        self.assertEqual(constants, [[1.0, -2.0, 0.5, 3.0], [0.125],
                                     [-0.75, 0.25, 2.0, -1.5], [-0.25]])

    def test_real_constant_encoding_is_separate_from_encrypted_execution(self):
        case = next(c for c in self.report["cases"] if c["case"] == "linear4x2")
        diagnostic = case["constant_payload_encoding"]
        self.assertEqual(diagnostic["exit_code"], 0)
        self.assertFalse(diagnostic["uses_artifact_execution_parameters"])
        self.assertFalse(diagnostic["encrypted_execution_validated"])
        self.assertEqual(diagnostic["data_level"], 7)
        self.assertIn("repeat", case["layout"]["constant_policy"])
        self.assertFalse(case["layout"]["backend_layout_validated"])
        self.assertIn("real Linear CST: constants=4", (self.root / "linear4x2/constant-encoding.log").read_text())

    def test_characterize_adapter_rejection_not_runtime_success(self):
        # Replace this characterization after ModswitchC semantics are verified.
        self.assertEqual(self.report["status"], "adapter_blocked")
        for case in self.report["cases"]:
            self.assertNotEqual(case["adapter_exit_code"], 0)
            self.assertFalse(case["execution_gate"]["ok"])
            self.assertTrue(case["execution_gate"]["checks"]["artifacts_loaded"])
            self.assertFalse(case["execution_gate"]["checks"]["schedule_built"])
            report = json.loads((self.root / case["case"] / "poseidon-preflight.json").read_text())
            opcodes = report["hevm_opcode_summary"]["opcode_counts"]
            self.assertTrue(any(op["name"] == "ModswitchC" and not op["supported"] for op in opcodes))
            self.assertFalse(any(op["name"] == "BootstrapC" for op in opcodes))


if __name__ == "__main__":
    unittest.main()
