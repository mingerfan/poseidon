"""Multi-input permission, binding, reference and opt-in encrypted evidence tests."""
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest

from candidate_trace import evaluate_tree
from hecate_contract import validate_function
from multi_input_fixtures import CASES, NAMES, CONSTANTS, fixture_inputs, reference
from seal_artifact_gate import inspect_artifacts

SOURCES = Path(__file__).parent / "golden_cases/multi_input"


def artifact(arity, rhs=1):
    """Minimal AddCC artifact with independent initial registers, no fake execution."""
    nresults, nops, ncipher = 1, 1, arity+1
    config = [40+8*(2*arity+3), nops, ncipher, 0, 13]
    metadata = [40]*arity + [13]*arity + [40, 13, arity]
    return (struct.pack("<IIQQ", 0x4845564D, 24, arity, nresults) +
            struct.pack("<"+"Q"*(len(config)+len(metadata)), *(config+metadata)) +
            struct.pack("<4H", 6, arity, 0, rhs))


class MultiInputTests(unittest.TestCase):
    def test_all_golden_signatures_and_no_legacy_relaxation(self):
        for case, (arity, _) in CASES.items():
            source = (SOURCES / (case+".py")).read_text()
            check = validate_function(source, CONSTANTS, 2 if case == "dual_outputs" else 1,
                                      contract="hecate-function-v3", input_names=NAMES[:arity])
            self.assertEqual([v["name"] for v in check["inputs"]], list(NAMES[:arity]))
            self.assertNotIn("input", check)
            self.assertFalse(check["encrypted_correctness_checked"])
            for version in ("hecate-function-v0", "hecate-function-v1", "hecate-function-v2"):
                with self.assertRaises(ValueError):
                    validate_function(source, CONSTANTS, contract=version, input_names=NAMES[:arity])

    def test_bad_signatures_decorator_defaults_aliases_and_constants(self):
        source = (SOURCES / "dual_add.py").read_text()
        for replacement in (source.replace('"c,c"', '"c,p"'), source.replace("x, y", "y, x"),
                            source.replace("x, y", "x, y=0"), source.replace("x, y", "*args"),
                            source.replace("x, y", "x, x"), source.replace("return x + y", "y = x\n    return y")):
            with self.assertRaises(ValueError):
                validate_function(replacement, {}, contract="hecate-function-v3", input_names=("x", "y"))
        for names in (("x",), ("x", "x"), ("x", "hc"), ("x", "y", "z", "t", "u"), (True, "y")):
            with self.assertRaises(ValueError):
                validate_function(source, {}, contract="hecate-function-v3", input_names=names)
        with self.assertRaises(ValueError):
            validate_function(source, {"y": 1.0}, contract="hecate-function-v3", input_names=("x", "y"))

    def test_ast_preserves_distinct_input_objects_and_order(self):
        class Expr:
            def __init__(self, name):
                self.name = name
            def __sub__(self, other):
                return (self.name, "minus", other.name)
        source = (SOURCES / "ordered_subtract.py").read_text()
        x, y = Expr("encrypted-x"), Expr("encrypted-y")
        self.assertEqual(evaluate_tree(source, {}, encrypted_inputs={"x": x, "y": y}),
                         ("encrypted-x", "minus", "encrypted-y"))
        for values in ({"x": x}, {"x": x, "y": y, "z": x}):
            with self.assertRaises(ValueError):
                evaluate_tree(source, {}, encrypted_inputs=values)

    def test_hand_calculated_references_and_input_independence(self):
        x, y = [1., 2., 3., 4.], [4., 1., -1., 0.]
        self.assertEqual(reference("ordered_subtract", [x, y], CONSTANTS), [-3., 1., 4., 4.])
        self.assertEqual(reference("dual_product", [x, y], CONSTANTS), [4., 2., -3., 0.])
        self.assertEqual(reference("dot_difference", [x, y], CONSTANTS), [2.375])
        self.assertEqual(reference("dual_outputs", [x, y], CONSTANTS), [5., -3.])
        self.assertEqual(reference("triple_merge", [x, y, x], CONSTANTS), y)
        self.assertEqual(reference("quad_merge", [x, y, y, x], CONSTANTS), [0.]*4)
        inputs = fixture_inputs(4)
        self.assertEqual(inputs, fixture_inputs(4))
        self.assertNotEqual(inputs[1][0], inputs[1][1])
        inputs[0][0][0] = 1.0
        self.assertEqual(inputs[0][1][0], 0.0)

    def test_artifact_requires_explicit_arity_and_all_operands_initialized(self):
        cst = struct.pack("<q", 0)
        for arity in (2, 3, 4):
            gate = inspect_artifacts(artifact(arity), cst, expected_inputs=arity)
            self.assertEqual(len(gate["arg_level"]), arity)
            with self.assertRaises(ValueError):
                inspect_artifacts(artifact(arity), cst)
            with self.assertRaises(ValueError):
                inspect_artifacts(artifact(arity, rhs=arity), cst, expected_inputs=arity)
        for count in (True, 0, 5, "2"):
            with self.assertRaises(ValueError):
                inspect_artifacts(artifact(2), cst, expected_inputs=count)

    def test_wrong_order_counterexample_differs_on_asymmetric_inputs(self):
        batch = fixture_inputs(2)[1]
        forward = reference("ordered_subtract", batch, CONSTANTS)
        reverse = reference("ordered_subtract", batch[::-1], CONSTANTS)
        self.assertNotEqual(forward, reverse)
        self.assertGreater(max(abs(a-b) for a, b in zip(forward, reverse)), 0.1)


class MultiInputTorchTests(unittest.TestCase):
    def test_independent_reference_matches_all_torch_cases(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Run in pinned Torch environment")
        from multi_input_fixtures import torch_model
        for case, (arity, _) in CASES.items():
            model = torch_model(case, CONSTANTS)
            for batch in fixture_inputs(arity):
                actual = model(*[torch.tensor(v, dtype=torch.float64) for v in batch])
                expected = torch.tensor(reference(case, batch, CONSTANTS), dtype=torch.float64)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12), case)


@unittest.skipUnless(os.environ.get("POSEIDON_MULTI_INPUT_RESULTS"), "Requires real multi-input evidence")
class MultiInputEvidenceTests(unittest.TestCase):
    def test_real_arity_metadata_immutability_and_counterexample(self):
        root = Path(os.environ["POSEIDON_MULTI_INPUT_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["agent_calls"], 0)
        self.assertFalse(report["poseidon_gpu_validated"])
        self.assertEqual(report["parameters"]["security_check"], "tc128")
        self.assertEqual(report["parameters"]["modulus_bits"], [60]*14)
        self.assertEqual(report["parameters"]["rotation_steps"], [1, 2])
        self.assertEqual(len(report["cases"]), 9)
        self.assertEqual({x["case"] for x in report["cases"] if not x["counterexample"]}, set(CASES))
        for item in report["cases"]:
            self.assertTrue(item["matched_expected"])
            self.assertEqual(item["comparison"]["passed"], not item["counterexample"])
            execution = item["execution"]
            arity = CASES[item["case"]][0]
            self.assertEqual(execution["backend"], "upstream_SEAL_HEVM_CPU")
            self.assertTrue(execution["encrypted_execution"])
            self.assertFalse(execution["bootstrap_executed"])
            self.assertEqual(execution["encrypted_input_count"], arity)
            self.assertEqual(execution["input_batches"], 4)
            self.assertTrue(execution["rotation_key_check"]["actual_key_file_verified"])
            self.assertEqual(item["comparison"]["atol"], 1e-5)
            self.assertEqual(item["comparison"]["rtol"], 1e-4)
            self.assertEqual(len(item["artifact_gate"]["arg_level"]), arity)
            for observation in execution["ciphertext_metadata"]:
                self.assertEqual(len(observation["inputs"]), arity)
                self.assertTrue(all(m["polynomials"] == 2 for m in observation["inputs"]+observation["outputs"]))
            directory = root / item["directory"]
            for name, expected in item["immutable_hashes"].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(report["cases"][-1]["failure_layer"], "numerical_comparison")


if __name__ == "__main__":
    unittest.main()
