"""Rotation semantics and key policy: static tests separate from encrypted evidence."""
import ast
import hashlib
import json
import os
import struct
from pathlib import Path
import unittest

from candidate_contract import make_request, request_rotations, validate_candidate
from candidate_trace import evaluate_tree
from hecate_contract import CONTRACT_ROTATIONS, rotation_literal, validate_function
from model_graph import evaluate_reference, validate_graph
from seal_artifact_gate import inspect_artifacts

HERE = Path(__file__).resolve().parent
EMPTY_CST = struct.pack("<q", 0)


def artifact(ops=((6, 0, 0, 0),)):
    # Minimal data-only fixture: one input/output, level13, scale40, no constants.
    return (struct.pack("<IIQQ", 0x4845564D, 24, 1, 1) +
            struct.pack("<5Q", 80, len(ops), 2, 2, 13) +
            struct.pack("<5Q", 40, 13, 40, 13, 0) +
            b"".join(struct.pack("<4H", *op) for op in ops))


class RotationContractTests(unittest.TestCase):
    def test_signed_literal_is_not_eval(self):
        for text, expected in (("-3", -3), ("3", 3), ("-1", -1)):
            self.assertEqual(rotation_literal(ast.parse(text, mode="eval").body), expected)
        for text in ("True", "1+1", "int('1')", "x", "--1", "-True", "1.0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                rotation_literal(ast.parse(text, mode="eval").body)

    def test_v2_steps_and_legacy_rejection(self):
        for step in (-3, -2, -1, 1, 2, 3):
            source = f'@hc.func("c")\ndef golden(x):\n    return x.rotate({step})\n'
            check = validate_function(source, {}, contract="hecate-function-v2")
            self.assertEqual(check["rotation_steps"], [step])
            if step not in (1, 2):
                for contract in ("hecate-function-v0", "hecate-function-v1"):
                    with self.assertRaises(ValueError):
                        validate_function(source, {}, contract=contract)
        for step in (0, 4, -4, 16384):
            with self.assertRaises(ValueError):
                validate_function(f'@hc.func("c")\ndef golden(x):\n    return x.rotate({step})\n', {}, contract="hecate-function-v2")

    def test_interpreter_passes_signed_step_unchanged(self):
        class Cipher:
            def rotate(self, step):
                return step
        source = '@hc.func("c")\ndef golden(x):\n    return x.rotate(-3)\n'
        validate_function(source, {}, contract="hecate-function-v2")
        self.assertEqual(evaluate_tree(source, {}, Cipher()), -3)

    def test_reference_direction_by_explicit_expected_vectors(self):
        expected = {-3: [2., 3., 4., 1.], -2: [3., 4., 1., 2.], -1: [4., 1., 2., 3.],
                    1: [2., 3., 4., 1.], 2: [3., 4., 1., 2.], 3: [4., 1., 2., 3.]}
        for step, result in expected.items():
            suffix = ("minus" if step < 0 else "plus") + str(abs(step))
            case = json.loads((HERE / f"cases/rotate-{suffix}.json").read_text())
            validate_graph(case)
            self.assertEqual(evaluate_reference(case, [1., 2., 3., 4.]), result)

    def test_artifact_signed_encoding_and_missing_policy_keys(self):
        for step in (-3, -2, -1, 1, 2, 3):
            raw = artifact(((1, 0, 0, step & 65535),))
            gate = inspect_artifacts(raw, EMPTY_CST, rotation_steps=CONTRACT_ROTATIONS["hecate-function-v2"])
            self.assertEqual(gate["rotation_steps"], [step])
            if step not in (1, 2):
                with self.assertRaisesRegex(ValueError, "Rotation"):
                    inspect_artifacts(raw, EMPTY_CST)

    def test_malformed_key_policy_and_invalid_model_steps(self):
        for steps in ([], [True], [0], [4], [1, 1], "1"):
            with self.assertRaises(ValueError):
                inspect_artifacts(artifact(), EMPTY_CST, rotation_steps=steps)
        case = json.loads((HERE / "cases/rotate-minus1.json").read_text())
        for step in (True, 0, 4, -4, "-1", 1.0):
            case["nodes"][0]["step"] = step
            with self.assertRaises(ValueError):
                validate_graph(case)

    def test_request_version_and_key_policy_bound_together(self):
        payload = dict(fx_graph="rotate", public_constants={}, constant_origins={},
                       layout=dict(output_ciphertexts=1), static_check=dict(contract="hecate-function-v2"))
        request = make_request(payload, {"schema": 2}, "a" * 64)
        self.assertEqual(request_rotations(request), (-3, -2, -1, 1, 2, 3))
        candidate = dict(schema=1, request_id=request["request_id"],
                         hecate_source='@hc.func("c")\ndef golden(x):\n    return x.rotate(-2)\n')
        self.assertEqual(validate_candidate(candidate, request)["rotation_steps"], [-2])
        request["task"] = "hecate-function-synthesis-v0"
        with self.assertRaises(ValueError):
            request_rotations(request)


class RotationEncryptedEvidenceTests(unittest.TestCase):
    def test_actual_missing_key_probe(self):
        selected = os.environ.get("POSEIDON_ROTATION_MISSING_KEY_REPORT")
        if not selected:
            self.skipTest("set POSEIDON_ROTATION_MISSING_KEY_REPORT to a real key-file probe")
        report = json.loads(Path(selected).read_text())
        self.assertTrue(report["actual_key_file_checked"] and report["matched"])
        self.assertEqual(report["required_steps"], [-1])
        self.assertEqual(report["result_code"], 2)
        self.assertEqual(report["expected_result_code"], 2)
        self.assertFalse(report["encrypted_execution"])

    def test_all_six_rotations_and_wrong_direction_real(self):
        selected = os.environ.get("POSEIDON_ROTATION_BATCH_RESULTS")
        if not selected:
            self.skipTest("set POSEIDON_ROTATION_BATCH_RESULTS to a real seven-case rotation run")
        report = json.loads((Path(selected) / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 7)
        self.assertEqual([c["step"] for c in report["cases"] if not c["counterexample"]], [-3, -2, -1, 1, 2, 3])
        for case in report["cases"]:
            self.assertTrue(case["matched_expected"])
            run = Path(case["run"])
            data = json.loads((run / "report.json").read_text())
            self.assertEqual(data["agent_calls"], 0)
            self.assertFalse(data["poseidon_gpu_validated"])
            self.assertEqual(data["parameters"]["rotation_steps"], [-3, -2, -1, 1, 2, 3])
            self.assertEqual(data["parameters"]["security_check"], "tc128")
            self.assertEqual(data["parameters"]["modulus_bits"], [60] * 14)
            attempt = data["attempts"][0]
            self.assertTrue(attempt["execution"]["encrypted_execution"])
            self.assertTrue(attempt["execution"]["rotation_key_check"]["actual_key_file_verified"])
            self.assertEqual(attempt["execution"]["input_batches"], 4)
            expected_step = 1 if case["counterexample"] else case["step"]
            self.assertEqual(attempt["artifact_gate"]["rotation_steps"], [expected_step])
            self.assertEqual(attempt["execution"]["rotation_key_check"]["required_steps"], [expected_step])
            self.assertEqual(attempt["comparison"]["passed"], not case["counterexample"])
            for name, expected in data["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), expected)
            for name, expected in attempt["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((run / "attempt-00/output" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
