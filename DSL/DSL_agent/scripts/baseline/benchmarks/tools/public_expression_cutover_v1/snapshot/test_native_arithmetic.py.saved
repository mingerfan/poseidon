"""Versioned native Hecate arithmetic; symbolic tests never claim FHE execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from candidate_contract import canonical, make_request, validate_candidate
from candidate_trace import evaluate_tree
from deepseek_provider import ProviderError, public_request
from hecate_contract import validate_function
from test_candidate_pipeline import request_fixture


def fragment(expression):
    return '@hc.func("c")\ndef golden(x):\n    return ' + expression + '\n'


def v1_request():
    base = request_fixture()
    return make_request(dict(fx_graph=base["fx_graph"], public_constants=base["public_constants"],
                             constant_origins=base["constant_origins"], layout=base["layout"]),
                        {"schema": 2, "id": "public-model"}, "a" * 64)


class NativeArithmeticTests(unittest.TestCase):
    def test_v0_does_not_silently_change(self):
        for expression in ("-x", "x - x", "x - w"):
            with self.assertRaises(ValueError):
                validate_function(fragment(expression), {"w": 1})
        self.assertEqual(request_fixture()["task"], "hecate-function-synthesis-v0")

    def test_v1_cipher_negation_subtraction_and_public_subtraction(self):
        for expression, kind in (("-x", "negate"), ("x-x", "subtract"), ("x-w", "subtract")):
            result = validate_function(fragment(expression), {"w": [1, 2, 3, 4]}, contract="hecate-function-v1")
            self.assertEqual(result["operator_counts"][kind], 1)
            self.assertEqual(result["outputs"][0]["slot_period"], 4)
            self.assertFalse(result["encrypted_correctness_checked"])

    def test_order_and_negation_preserved_in_symbolic_dispatch(self):
        class Symbol:
            def __init__(self, value):
                self.value = value
            def __neg__(self):
                return Symbol(("negate", self.value))
            def __sub__(self, other):
                return Symbol(("subtract", self.value, other.value if isinstance(other, Symbol) else other))
        for expression, expected in (("-x-w", ("subtract", ("negate", "cipher"), "public")),
                                     ("x-(-x)", ("subtract", "cipher", ("negate", "cipher")))):
            source = fragment(expression)
            validate_function(source, {"w": 1}, contract="hecate-function-v1")
            self.assertEqual(evaluate_tree(source, {"w": "public"}, Symbol("cipher")).value, expected)

    def test_public_only_arithmetic_literals_and_other_unary_ops_rejected(self):
        for expression in ("-w", "w-x", "w-w", "-1", "+x", "~x", "not x", "x / w",
                           "x - True", "x.__neg__()", "x.rotate(-1)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                validate_function(fragment(expression), {"w": 1}, contract="hecate-function-v1")

    def test_native_arithmetic_does_not_enable_mutation(self):
        source = '@hc.func("c")\ndef golden(x):\n    x -= x\n    return x\n'
        with self.assertRaises(ValueError):
            validate_function(source, {}, contract="hecate-function-v1")

    def test_unknown_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_function(fragment("x"), {}, contract="hecate-function-v999")

    def test_v1_request_rules_are_hashed_and_provider_compatible(self):
        request = v1_request()
        self.assertEqual(request["task"], "hecate-function-synthesis-v1")
        self.assertIn("Subtraction is ordered", request["rules"])
        self.assertEqual(public_request(request), request)
        candidate = dict(schema=1, request_id=request["request_id"], hecate_source=fragment("-x-w"))
        self.assertEqual(validate_candidate(candidate, request)["contract"], "hecate-function-v1")

    def test_request_cannot_mix_version_and_rules(self):
        request = v1_request()
        request["rules"] = request_fixture()["rules"]
        request["request_id"] = hashlib.sha256(canonical({k: v for k, v in request.items() if k != "request_id"})).hexdigest()
        with self.assertRaises(ProviderError):
            public_request(request)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_candidate(dict(schema=1, request_id=request["request_id"], hecate_source=fragment("-x")), request)

    def test_negation_counts_toward_resource_limit(self):
        request = v1_request()
        lines = ['@hc.func("c")', 'def golden(x):']
        for i in range(257):
            lines.append(f'    v{i} = -' + ('x' if i == 0 else f'v{i-1}'))
        lines.append('    return v256')
        with self.assertRaisesRegex(ValueError, "operation budget"):
            validate_candidate(dict(schema=1, request_id=request["request_id"], hecate_source='\n'.join(lines)), request)


class NativeArithmeticEvidenceTests(unittest.TestCase):
    def report(self, variable):
        value = os.environ.get(variable)
        if not value:
            self.skipTest("set " + variable + " to a real native-arithmetic candidate run")
        folder = Path(value)
        report = json.loads((folder / "report.json").read_text())
        request = json.loads((folder / "request.json").read_text())
        self.assertEqual(request["task"], "hecate-function-synthesis-v1")
        self.assertEqual(report["agent_calls"], 0)
        self.assertEqual(report["backend"], "upstream_SEAL_HEVM_CPU")
        self.assertFalse(report["poseidon_gpu_validated"])
        self.assertFalse(report["llm_generation_validated"])
        for name, expected in report["frozen_hashes"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected)
        attempt = report["attempts"][0]
        self.assertTrue(attempt["execution"]["encrypted_execution"])
        self.assertFalse(attempt["execution"]["bootstrap_executed"])
        self.assertEqual(attempt["execution"]["input_batches"], 4)
        self.assertTrue(attempt["parsed"] and attempt["checked"] and attempt["compiled"] and attempt["executed"])
        for name, expected in attempt["artifact_hashes"].items():
            self.assertEqual(hashlib.sha256((folder / "attempt-00/output" / name).read_bytes()).hexdigest(), expected)
        return report, attempt

    def test_native_negate_and_plain_subtract_real(self):
        report, attempt = self.report("POSEIDON_NATIVE_NEGATE_RESULTS")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(attempt["static_check"]["operator_counts"]["negate"], 1)
        self.assertEqual(attempt["static_check"]["operator_counts"]["subtract"], 1)
        self.assertGreater(int(attempt["artifact_gate"]["opcode_counts"].get("2", 0)), 0)
        self.assertTrue(attempt["comparison"]["passed"])

    def test_native_cipher_subtract_real(self):
        report, attempt = self.report("POSEIDON_NATIVE_SUBTRACT_RESULTS")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(attempt["comparison"]["compared_values"], 16)
        self.assertEqual(attempt["static_check"]["operator_counts"]["subtract"], 1)
        self.assertGreater(int(attempt["artifact_gate"]["opcode_counts"].get("2", 0)), 0)
        self.assertTrue(attempt["comparison"]["passed"])

    def test_swapped_subtraction_detected_after_decryption(self):
        report, attempt = self.report("POSEIDON_NATIVE_SUBTRACT_WRONG_RESULTS")
        self.assertNotEqual(report["status"], "passed")
        self.assertEqual(attempt["failure_layer"], "numerical_comparison")
        self.assertEqual(attempt["category"], "candidate")
        self.assertFalse(attempt["comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
