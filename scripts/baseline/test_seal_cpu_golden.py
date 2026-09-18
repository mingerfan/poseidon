"""Offline negative gates plus opt-in real SEAL encrypted-result evidence."""
import json
import math
import os
from pathlib import Path
import struct
import unittest

from hecate_python_env import digest
from seal_artifact_gate import inspect_artifacts


def artifact(ops=((6, 0, 0, 0),), result_level=13, result_register=0):
    return (struct.pack("<IIQQ", 0x4845564D, 24, 1, 1) +
            struct.pack("<5Q", 80, len(ops), 2, 2, 13) +
            struct.pack("<5Q", 40, 13, 40, result_level, result_register) +
            b"".join(struct.pack("<4H", *op) for op in ops))


EMPTY_CST = struct.pack("<q", 0)


class SealGateTests(unittest.TestCase):
    def test_add_and_real_modswitch(self):
        gate = inspect_artifacts(artifact(), EMPTY_CST)
        self.assertFalse(gate["execution_validated"])
        gate = inspect_artifacts(artifact(((4, 0, 0, 11), (6, 0, 0, 0)), result_level=2), EMPTY_CST)
        self.assertEqual(gate["res_level"], [2])

    def test_bootstrap_upscale_and_unknown_rejected_even_at_end(self):
        for opcode in (10, 5, 11, 42, 65534):
            with self.subTest(opcode=opcode), self.assertRaisesRegex(ValueError, "Forbidden opcode"):
                inspect_artifacts(artifact(((6, 0, 0, 0), (opcode, 0, 0, 0))), EMPTY_CST)

    def test_alloc_unspecified_operands_ignored(self):
        inspect_artifacts(artifact(((65535, 60000, 50000, 40000), (6, 0, 0, 0))), EMPTY_CST)

    def test_truncated_and_trailing_hevm(self):
        for data in (artifact()[:-1], artifact() + b"x", b"", b"bad" * 40):
            with self.assertRaises(ValueError):
                inspect_artifacts(data, EMPTY_CST)

    def test_resource_count_bound(self):
        data = bytearray(artifact())
        struct.pack_into("<Q", data, 40, 1000000)
        with self.assertRaisesRegex(ValueError, "Resource limit"):
            inspect_artifacts(data, EMPTY_CST)

    def test_malformed_constants(self):
        invalid = [b"", EMPTY_CST + b"x", struct.pack("<q", -1), struct.pack("<q", 1),
                   struct.pack("<qq", 1, 0), struct.pack("<qq", 1, -3),
                   struct.pack("<qqd", 1, 1, float("nan"))]
        for cst in invalid:
            with self.assertRaises(ValueError):
                inspect_artifacts(artifact(), cst)

    def test_uninitialized_register_and_bad_result(self):
        for data in (artifact(((6, 0, 1, 0),)), artifact(result_register=1),
                     artifact(result_register=2), artifact(result_level=2)):
            with self.assertRaises(ValueError):
                inspect_artifacts(data, EMPTY_CST)

    def test_modswitch_zero_negative_overdrop(self):
        for drop in (0, 13, 65535):
            with self.assertRaisesRegex(ValueError, "modswitch"):
                inspect_artifacts(artifact(((4, 0, 0, drop),)), EMPTY_CST)

    def test_rotation_key_allowlist(self):
        for step in (1, 2):
            gate = inspect_artifacts(artifact(((1, 0, 0, step),)), EMPTY_CST)
            self.assertEqual(gate["rotation_steps"], [step])
        for step in (3, 65535):
            with self.assertRaisesRegex(ValueError, "Rotation"):
                inspect_artifacts(artifact(((1, 0, 0, step),)), EMPTY_CST)

    def test_encode_and_eager_plain_reuse_rejected(self):
        encode = (0, 0, 0, (13 << 10) | 40)
        cst = struct.pack("<qqd", 1, 1, 2)
        inspect_artifacts(artifact((encode, (9, 0, 0, 0))), cst)
        with self.assertRaisesRegex(ValueError, "Repeated plain"):
            inspect_artifacts(artifact((encode, encode)), cst)


RESULTS = os.environ.get("POSEIDON_SEAL_GOLDEN_RESULTS")


@unittest.skipUnless(RESULTS, "requires explicitly selected real SEAL encrypted run")
class SealEncryptedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())

    def test_backend_boundary_and_parameters(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertTrue(self.report["encrypted_execution_validated"])
        self.assertFalse(self.report["poseidon_gpu_execution_validated"])
        self.assertEqual(self.report["backend"], "upstream_SEAL_HEVM_CPU")
        params = self.report["parameters"]
        self.assertEqual(params["security_check"], "tc128")
        self.assertTrue(params["parameters_set"])
        self.assertEqual(params["polynomial_degree"], 32768)
        self.assertEqual(params["modulus_bits"], [60] * 14)
        self.assertEqual(params["rotation_steps"], [1, 2])

    def test_artifact_integrity_and_no_bootstrap(self):
        expected = self.report.get("selected_cases", ["add", "mul_plain", "linear4x2"])
        self.assertEqual([c["case"] for c in self.report["cases"]], expected)
        for case in self.report["cases"]:
            folder = self.root / case["case"]
            for name, expected in case["input_artifact_hashes"].items():
                self.assertEqual(digest(folder / name), expected)
            gate = inspect_artifacts((folder / "lowered._hecate_golden.hevm").read_bytes(),
                                     (folder / "_hecate_golden.cst").read_bytes())
            self.assertEqual(gate, case["gate"])
            self.assertEqual(case["execution_exit_code"], 0)
            self.assertTrue(case["execution"]["encrypted_execution"])
            self.assertFalse(case["execution"]["bootstrap_executed"])

    def test_frozen_tolerance_independent_reference_and_recompute_metrics(self):
        for case in self.report["cases"]:
            comp = case["comparison"]
            self.assertEqual(comp["atol"], 1e-5)
            self.assertEqual(comp["rtol"], 1e-4)
            self.assertEqual(comp["reference"], case["reference"])
            self.assertEqual(len(comp["actual"]), 4)
            errors = []
            for actuals, refs in zip(comp["actual"], comp["reference"]):
                self.assertEqual(len(actuals), len(refs))
                for actual, ref in zip(actuals, refs):
                    self.assertTrue(math.isfinite(actual))
                    error = abs(actual - ref)
                    self.assertLessEqual(error, 1e-5 + 1e-4 * abs(ref))
                    errors.append(error)
            self.assertEqual(max(errors), comp["max_absolute_error"])
            self.assertAlmostEqual(sum(errors) / len(errors), comp["mae"], places=15)
        linear = next((c for c in self.report["cases"] if c["case"] == "linear4x2"), None)
        if linear is not None:
            self.assertEqual(linear["reference"][0], [0.125, -0.25])
            self.assertEqual(linear["reference"][1], [0.5, 0.75])
            self.assertEqual(linear["gate"]["rotation_steps"], [1, 2])


EXTENDED_RESULTS = os.environ.get("POSEIDON_SEAL_SEMANTICS_RESULTS")


@unittest.skipUnless(EXTENDED_RESULTS, "requires real MLP and primitive semantics run")
class SealSemanticEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(EXTENDED_RESULTS)
        cls.report = json.loads((cls.root / "report.json").read_text())
        cls.cases = {c["case"]: c for c in cls.report["cases"]}

    def test_extended_suite_cannot_silently_drop_cases(self):
        self.assertEqual(self.report["status"], "passed")
        self.assertTrue({"rotate1", "rotate2", "square", "quartic", "mlp4x4x2"} <= self.cases.keys())
        for name in ("rotate1", "rotate2", "square", "quartic", "mlp4x4x2"):
            self.assertTrue(self.cases[name]["comparison"]["passed"])
            self.assertEqual(self.cases[name]["execution_exit_code"], 0)

    def test_asymmetric_rotation_sign_and_stride(self):
        x = [0.5, -1.0, 0.25, -0.75]
        for step in (1, 2):
            case = self.cases[f"rotate{step}"]
            self.assertEqual(case["reference"][1], x[step:] + x[:step])
            self.assertEqual(case["gate"]["rotation_steps"], [step])
            for actual, ref in zip(case["comparison"]["actual"][1], case["reference"][1]):
                self.assertLessEqual(abs(actual - ref), 1e-5 + 1e-4 * abs(ref))

    def test_square_quartic_and_relinearization(self):
        for name, power in (("square", 2), ("quartic", 4)):
            case = self.cases[name]
            self.assertEqual(case["reference"][1], [x**power for x in [0.5, -1, 0.25, -0.75]])
            self.assertGreater(case["gate"]["opcode_counts"].get("8", 0), 0)
            if power == 4:
                self.assertEqual(case["gate"]["opcode_counts"].get("3"), 2)
            for sample in case["execution"]["ciphertext_metadata"]:
                self.assertTrue(all(o["polynomials"] == 2 for o in sample["outputs"]))

    def test_observed_levels_and_scales_agree_with_compiler(self):
        for case in self.cases.values():
            for sample in case["execution"]["ciphertext_metadata"]:
                self.assertEqual(sample["input"]["data_modulus_count"], 13)
                self.assertEqual(sample["input"]["log2_scale"], 40)
                for i, observed in enumerate(sample["outputs"]):
                    self.assertEqual(observed["data_modulus_count"], case["gate"]["res_level"][i])
                    self.assertAlmostEqual(observed["log2_scale"], case["gate"]["res_scale"][i], delta=1e-6)

    def test_mlp_fixed_reference_and_real_graph_coverage(self):
        case = self.cases["mlp4x4x2"]
        self.assertEqual(case["reference"][0], [0.064453125, -0.26171875])
        self.assertEqual(case["reference"][1], [3.44873046875, -2.47509765625])
        self.assertEqual(case["reference"][3], [19.970703125, -7.57421875])
        counts = case["gate"]["opcode_counts"]
        for opcode, count in ((1, 8), (3, 8), (8, 4), (9, 12)):
            self.assertEqual(counts[str(opcode)], count)
        self.assertNotIn("10", counts)
        fx = (self.root / "mlp4x4x2/reference-fx.txt").read_text()
        self.assertIn("call_module[target=hidden]", fx)
        self.assertIn("call_function[target=torch.square]", fx)
        self.assertIn("call_module[target=output]", fx)
