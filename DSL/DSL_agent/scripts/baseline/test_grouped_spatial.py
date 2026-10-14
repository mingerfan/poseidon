"""Grouped/dilated windows: hand arithmetic, independent Torch and real CKKS evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from model_graph import validate_graph, evaluate_reference
from spatial_ops import nested
from run_grouped_spatial_goldens import CASES

BASE = Path(__file__).parent
MATRICES = (
    [[.5, 0, -.25, 0], [0, .5, 0, -.25]],
    [[.25, 0, 0, .5]],
    [[.5, 0, -.25, 0], [0, .5, 0, -.25]],
    [[.5, 0, 0, 0], [0, .5, 0, 0], [0, 0, -.25, 0], [0, 0, 0, -.25]],
    [[.5, -.25, 0, 0], [.25, .5, 0, 0], [0, 0, -.5, .25], [0, 0, .75, -.25]],
    [[0, -.25, 0, 0], [0, 0, 0, .25]],
)
EXPECTED = ([-.125, .125], [2.125], [-.125, .125],
            [.625, 1.125, -.875, -1.125], [.125, 1.125, -.25, 1.], [-.375, .875])


def case(name):
    return json.loads((BASE / f"cases/{name}.json").read_text())


class GroupedSpatialTests(unittest.TestCase):
    def test_hand_results_and_free_user_weights(self):
        for name, expected in zip(CASES, EXPECTED):
            data = case(name)
            self.assertEqual(validate_graph(data)["output_shape"], [len(expected)])
            vector = nested([1., 2., 3., 4.], data["input_shape"])
            self.assertEqual(evaluate_reference(data, vector), expected)
            data["id"] = "user-selected-weights-and-graph"
            self.assertEqual(evaluate_reference(data, vector), expected)
            data["constants"]["b"][0] += .125
            self.assertNotEqual(evaluate_reference(data, vector), expected)

    def test_reject_invalid_groups_dilation_and_extra_fields(self):
        for name, field, value in (
            (CASES[0], "groups", 2), (CASES[3], "groups", 0), (CASES[3], "groups", 3),
            (CASES[3], "groups", True), (CASES[3], "groups", 2.0),
            (CASES[0], "dilation", [0]), (CASES[0], "dilation", [-1]),
            (CASES[0], "dilation", [True]), (CASES[0], "dilation", [5]),
            (CASES[0], "dilation", [1, 2]), (CASES[0], "dilation", None),
            (CASES[0], "dilation", 2), (CASES[0], "dilation", [4]),
            (CASES[0], "padding_mode", "reflect")):
            data = case(name)
            data["nodes"][0][field] = value
            with self.subTest(name=name, field=field, value=value), self.assertRaises(ValueError):
                validate_graph(data)
        data = case(CASES[4])
        data["constants"]["w"].pop()
        data["constants"]["b"].pop()
        with self.assertRaises(ValueError):
            validate_graph(data)  # Cout=3 is not divisible by groups=2.
        data = case(CASES[3])
        data["constants"]["w"][0].append([.5])
        data["constants"]["w"][1].append([.25])
        with self.assertRaises(ValueError):
            validate_graph(data)  # Weight channels must be Cin/groups, not Cin.

    def test_old_implicit_defaults_equal_explicit_defaults(self):
        from test_spatial import CASES as old_cases, load
        for name in old_cases:
            data = load(name)
            explicit = copy.deepcopy(data)
            for node in explicit["nodes"]:
                if node["op"].startswith("conv"):
                    node.update(groups=1, dilation=[1]*(1 if node["op"] == "conv1d" else 2))
            vector = nested([.5, -1., .25, -.75], data["input_shape"])
            self.assertEqual(evaluate_reference(data, vector), evaluate_reference(explicit, vector))


try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires pinned Torch")
class GroupedSpatialTorchTests(unittest.TestCase):
    def test_reference_torch_exact_masks_rule_and_manual_programs(self):
        from model_catalog import build_model, test_inputs
        from fx_to_hecate import translate
        from test_fx_to_hecate import interpret_fragment
        from hecate_contract import validate_function
        from candidate_contract import make_request
        from deepseek_provider import public_request
        for name, masks in zip(CASES, MATRICES):
            data = case(name)
            model, shape = build_model(data)
            payload = translate(model, shape)
            request = make_request(payload, data, "a"*64)
            self.assertEqual(public_request(request), request)
            self.assertEqual(request["model"], data)
            self.assertNotIn("reference", request)
            self.assertNotIn("hecate_source", request)
            self.assertEqual(payload["layout"]["input_slot_period"], 4)
            self.assertEqual(payload["layout"]["output_ciphertexts"], len(masks))
            for index, mask in enumerate(masks):
                self.assertEqual(payload["public_constants"][f"c{2*index}"], mask)
            manual = dict(payload, hecate_source=(BASE / f"golden_cases/grouped_spatial/{name}.py").read_text())
            validate_function(manual["hecate_source"], payload["public_constants"], len(masks))
            for x in test_inputs(shape):
                expected = evaluate_reference(data, x.tolist())
                np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(interpret_fragment(payload, x.reshape(-1)), expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(interpret_fragment(manual, x.reshape(-1)), expected, atol=1e-12, rtol=1e-12)

    def test_wrong_tap_and_wrong_group_pass_ast_but_not_semantics(self):
        from model_catalog import build_model, test_inputs
        from fx_to_hecate import translate
        from test_fx_to_hecate import interpret_fragment
        from hecate_contract import validate_function
        for name, suffix in ((CASES[0], "wrong-tap"), (CASES[3], "wrong-group")):
            data = case(name)
            model, shape = build_model(data)
            payload = translate(model, shape)
            payload["hecate_source"] = (BASE / f"golden_cases/grouped_spatial/{name}-{suffix}.py").read_text()
            validate_function(payload["hecate_source"], payload["public_constants"], len(EXPECTED[CASES.index(name)]))
            self.assertTrue(any(not np.allclose(interpret_fragment(payload, x.reshape(-1)),
                                               evaluate_reference(data, x.tolist()), atol=1e-12, rtol=1e-12)
                                for x in test_inputs(shape)))


@unittest.skipUnless(os.environ.get("POSEIDON_GROUPED_SPATIAL_RESULTS"), "Requires real grouped/dilated candidate evidence")
class GroupedSpatialEvidenceTests(unittest.TestCase):
    def test_six_goldens_two_counterexamples_actual_immutable_ckks(self):
        root = Path(os.environ["POSEIDON_GROUPED_SPATIAL_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual({c["case"] for c in report["cases"] if not c["counterexample"]}, set(CASES))
        self.assertEqual(sum(c["counterexample"] for c in report["cases"]), 2)
        for row in report["cases"]:
            self.assertTrue(row["matched_expected"])
            run = Path(row["run"])
            data = json.loads((run / "report.json").read_text())
            self.assertEqual(data["agent_calls"], 0)
            self.assertFalse(data["poseidon_gpu_validated"])
            self.assertFalse(data["llm_generation_validated"])
            self.assertEqual(data["backend"], "upstream_SEAL_HEVM_CPU")
            self.assertEqual(data["parameters"]["security_check"], "tc128")
            self.assertEqual(data["parameters"]["modulus_bits"], [60]*14)
            attempt = data["attempts"][0]
            self.assertTrue(attempt["execution"]["encrypted_execution"])
            self.assertEqual(attempt["execution"]["input_batches"], 4)
            self.assertFalse(attempt["execution"]["bootstrap_executed"])
            self.assertFalse(attempt["trace"]["candidate_python_executed"])
            self.assertTrue(all(json.loads((run / "probe/probe.json").read_text()).values()))
            self.assertEqual(attempt["comparison"]["passed"], not row["counterexample"])
            self.assertEqual((attempt["comparison"]["atol"], attempt["comparison"]["rtol"]), (1e-5, 1e-4))
            for opcode in ("1", "9"):
                self.assertGreater(attempt["artifact_gate"]["opcode_counts"].get(opcode, 0), 0)
            if row["counterexample"]:
                self.assertEqual(attempt["failure_layer"], "numerical_comparison")
            for name, digest in data["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)
            for name, digest in attempt["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((run / "attempt-00/output" / name).read_bytes()).hexdigest(), digest)


@unittest.skipUnless(os.environ.get("POSEIDON_GROUPED_SPATIAL_RULE_RESULTS"), "Requires actual grouped/dilated rule baseline")
class GroupedSpatialRuleEvidenceTests(unittest.TestCase):
    def test_same_six_graphs_rule_pipeline_real_execution(self):
        root = Path(os.environ["POSEIDON_GROUPED_SPATIAL_RULE_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_descriptors"], [case(name) for name in CASES])
        self.assertEqual(len(report["cases"]), 6)
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
            for name, digest in row["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((root / row["folder"] / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
