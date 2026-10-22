"""Wider hidden neurons are scalar ciphertexts, not a changed packing/security profile."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from model_graph import validate_graph, evaluate_reference, LINEAR_WIDTH_LIMIT

BASE = Path(__file__).parent


def case(width=8):
    return json.loads((BASE / f"cases/custom-wide-mlp-{width}.json").read_text())


class WideLinearTests(unittest.TestCase):
    def test_four_explicit_hidden_widths_and_user_weights(self):
        self.assertEqual(LINEAR_WIDTH_LIMIT, 8)
        for width in (5, 6, 7, 8):
            data = case(width)
            self.assertEqual(validate_graph(data)["output_shape"], [2])
            original = evaluate_reference(data, [.5, -1, .25, -.75])
            data["id"] = "arbitrary-user-graph"
            self.assertEqual(evaluate_reference(data, [.5, -1, .25, -.75]), original)
            data["constants"]["weight_in"][-1][0] += .17
            self.assertNotEqual(evaluate_reference(data, [.5, -1, .25, -.75]), original)

    def test_width_9_bad_bias_broadcast_packing_and_output_rejected(self):
        invalid = []
        wide = case()
        wide["constants"]["weight_in"].append([.1]*4)
        wide["constants"]["bias_in"].append(.1)
        invalid.append(wide)
        for key, value in (("bias_in", [0]*7), ("offset", [0]*4), ("weight_out", [[.1]*7]*2)):
            data = case()
            data["constants"][key] = value
            invalid.append(data)
        data = case()
        data["output"] = "hidden"
        invalid.append(data)
        data = case()
        data["nodes"].insert(1, dict(id="invalid", op="add", inputs=["hidden", "x"]))
        invalid.append(data)
        data = case()
        data["input_shape"] = [8]
        invalid.append(data)
        for data in invalid:
            with self.assertRaises(ValueError):
                validate_graph(data)

    def test_hand_zero_input_reference_uses_all_eight_neurons(self):
        data = case()
        values = [b*b+o for b, o in zip(data["constants"]["bias_in"], data["constants"]["offset"])]
        expected = [sum(w*v for w, v in zip(row, values))+b
                    for row, b in zip(data["constants"]["weight_out"], data["constants"]["bias_out"])]
        self.assertEqual(evaluate_reference(data, [0]*4), expected)
        self.assertNotEqual(values[6], values[7])


try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires pinned Torch")
class WideLinearTorchTests(unittest.TestCase):
    def test_reference_torch_rule_and_manual_golden(self):
        from model_catalog import build_model, test_inputs
        from fx_to_hecate import translate
        from hecate_contract import validate_function
        from test_fx_to_hecate import interpret_fragment
        from candidate_contract import make_request
        from deepseek_provider import public_request
        for width in (5, 6, 7, 8):
            data = case(width)
            model, shape = build_model(data)
            payload = translate(model, shape)
            request = make_request(payload, data, "a"*64)
            self.assertEqual(public_request(request), request)
            self.assertNotIn("reference", request)
            self.assertNotIn("hecate_source", request)
            self.assertEqual(payload["layout"]["input_slot_period"], 4)
            self.assertEqual(payload["layout"]["output_ciphertexts"], 2)
            for x in test_inputs(shape):
                expected = evaluate_reference(data, x.tolist())
                np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(interpret_fragment(payload, x), expected, atol=1e-12, rtol=1e-12)
            if width == 8:
                self.assertEqual(len(payload["public_constants"]), 42)
                self.assertEqual(payload["public_constants"]["c14"], data["constants"]["weight_in"][7])
                self.assertEqual(payload["public_constants"]["c23"], data["constants"]["offset"][7])
                self.assertEqual(payload["public_constants"]["c31"], data["constants"]["weight_out"][0][7])
                for name, wrong in (("wide8", False), ("wide8-wrong-neuron", True)):
                    manual = copy.deepcopy(payload)
                    manual["hecate_source"] = (BASE / f"golden_cases/wide_linear/{name}.py").read_text()
                    validate_function(manual["hecate_source"], manual["public_constants"], 2)
                    differences = []
                    for x in test_inputs(shape):
                        actual, expected = interpret_fragment(manual, x), evaluate_reference(data, x.tolist())
                        differences.append(not np.allclose(actual, expected, atol=1e-12, rtol=1e-12))
                    self.assertEqual(any(differences), wrong)


@unittest.skipUnless(os.environ.get("POSEIDON_WIDE_RULE_RESULTS"), "Requires real width5..8 encrypted cohort")
class WideLinearRuleEvidenceTests(unittest.TestCase):
    def test_all_hidden_widths_real_encrypted_execution(self):
        root = Path(os.environ["POSEIDON_WIDE_RULE_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_descriptors"], [case(w) for w in (5, 6, 7, 8)])
        self.assertEqual(report["agent_calls"], 0)
        self.assertFalse(report["poseidon_gpu_validated"])
        self.assertEqual(report["parameters"]["modulus_bits"], [60]*14)
        self.assertEqual(report["parameters"]["security_check"], "tc128")
        for row in report["cases"]:
            self.assertEqual(row["status"], "passed")
            self.assertTrue(row["execution"]["encrypted_execution"])
            self.assertEqual(row["execution"]["input_batches"], 4)
            self.assertFalse(row["execution"]["bootstrap_executed"])
            self.assertTrue(row["comparison"]["passed"])
            self.assertEqual((row["comparison"]["atol"], row["comparison"]["rtol"]), (1e-5, 1e-4))
            for name, expected in row["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((root / row["folder"] / name).read_bytes()).hexdigest(), expected)


@unittest.skipUnless(os.environ.get("POSEIDON_WIDE_GOLDEN_RESULTS"), "Requires actual wide manual candidate and negative")
class WideLinearGoldenEvidenceTests(unittest.TestCase):
    def test_manual_and_wrong_neuron_through_isolated_candidate_path(self):
        report = json.loads((Path(os.environ["POSEIDON_WIDE_GOLDEN_RESULTS"]) / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 2)
        for row in report["cases"]:
            self.assertTrue(row["matched_expected"])
            root = Path(row["run"])
            candidate = json.loads((root / "report.json").read_text())
            self.assertEqual(candidate["agent_calls"], 0)
            self.assertFalse(candidate["llm_generation_validated"])
            self.assertFalse(candidate["poseidon_gpu_validated"])
            self.assertEqual(json.loads((root / "model.json").read_text()), case())
            attempt = candidate["attempts"][0]
            self.assertTrue(attempt["execution"]["encrypted_execution"])
            self.assertFalse(attempt["execution"]["bootstrap_executed"])
            self.assertEqual(attempt["comparison"]["passed"], not row["counterexample"])
            self.assertNotIn("10", attempt["artifact_gate"]["opcode_counts"])
            for name, expected in candidate["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
            for name, expected in attempt["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((root / "attempt-00/output" / name).read_bytes()).hexdigest(), expected)
            if row["counterexample"]:
                self.assertEqual(attempt["failure_layer"], "numerical_comparison")


if __name__ == "__main__":
    unittest.main()
