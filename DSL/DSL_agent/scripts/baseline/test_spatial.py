"""Spatial operator coverage: direct windows, static rejection and Torch/DSL."""
import copy
import json
import hashlib
import os
from pathlib import Path
import unittest

from model_graph import validate_graph, evaluate_reference
from spatial_ops import OPS, geometry

ROOT = Path(__file__).parent
CASES = ("conv1d-valid", "conv1d-stride-pad", "conv2d-valid", "conv2d-channels",
         "avg1d-pad-count", "avg1d-pad-exclude", "avg2d-valid", "avg2d-wide", "conv-square-pool")


def load(name):
    return json.loads((ROOT / f"cases/{name}.json").read_text())


class SpatialTests(unittest.TestCase):
    def test_every_spatial_operator_has_explicit_case(self):
        from model_graph import OPS as graph_ops
        # These non-spatial operators already have dedicated regression suites;
        # keep this exact inventory current without weakening spatial coverage.
        self.assertEqual(graph_ops, {"add", "multiply", "subtract", "negate", "square", "power", "linear", "flatten", "rotate",
                                     "batch_norm", "reshape", "concat"} | OPS)
        covered = set()
        for name in CASES:
            data = load(name)
            covered.update(set(validate_graph(data)["operators"]) & OPS)
        self.assertEqual(covered, OPS)

    def test_hand_calculated_cross_correlation_and_pool_divisor(self):
        self.assertEqual(evaluate_reference(load("conv1d-valid"), [[1., 2., 3., 4.]]), [.125, -.125, -.375])
        self.assertEqual(evaluate_reference(load("avg1d-pad-count"), [[1., 2., 3., 4.]]), [1., 3.])
        self.assertEqual(evaluate_reference(load("avg1d-pad-exclude"), [[1., 2., 3., 4.]]), [1.5, 3.])
        self.assertEqual(evaluate_reference(load("avg2d-valid"), [[[1., 2.], [3., 4.]]]), [2.5])
        self.assertEqual(evaluate_reference(load("avg2d-wide"), [[[1., 2., 3., 4.]]]), [1.5, 3.5])
        self.assertEqual(evaluate_reference(load("conv-square-pool"), [[1., 2., 3., 4.]]), [.578125, 1.015625])
        self.assertAlmostEqual(evaluate_reference(load("conv2d-valid"), [[[1., 2.], [3., 4.]]])[0], 2.85)
        for actual, expected in zip(evaluate_reference(load("conv2d-channels"), [[[1., 2.]], [[3., 4.]]]), [-1.6, -2.]):
            self.assertAlmostEqual(actual, expected)

    def test_reject_wrong_channels_groups_dilation_stride_padding_and_overflow(self):
        for field, value in (("stride", [0]), ("padding", [-1]), ("groups", 2), ("dilation", 2), ("stride", [True])):
            data = load("conv1d-valid")
            data["nodes"][0][field] = value
            with self.assertRaises(ValueError):
                validate_graph(data)
        data = load("conv2d-valid")
        data["constants"]["w"] = [[[[.5]], [[.25]]]]
        with self.assertRaises(ValueError):
            validate_graph(data)
        for change in ({"padding": [2]}, {"ceil_mode": True}, {"count_include_pad": 1}, {"divisor_override": 1}):
            data = load("avg1d-pad-count")
            data["nodes"][0].update(change)
            with self.assertRaises(ValueError):
                validate_graph(data)
        with self.assertRaises(ValueError):
            geometry("conv1d", (1, 4), (4, 1, 1), (1,), (1,), (0,))

    def test_wrong_kernel_orientation_and_divisor_are_observable(self):
        data = load("conv1d-valid")
        values = [[.5, -1., .25, -.75]]
        correct = evaluate_reference(data, values)
        data["constants"]["w"][0][0].reverse()
        self.assertNotEqual(evaluate_reference(data, values), correct)
        self.assertNotEqual(evaluate_reference(load("avg1d-pad-count"), values),
                            evaluate_reference(load("avg1d-pad-exclude"), values))


try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires pinned Torch")
class SpatialTorchTests(unittest.TestCase):
    def test_all_cases_reference_torch_rule_dsl(self):
        from model_catalog import build_model, test_inputs
        from fx_to_hecate import translate
        from test_fx_to_hecate import interpret_fragment
        from hecate_contract import validate_function
        for name in CASES:
            data = load(name)
            model, shape = build_model(data)
            payload = translate(model, shape)
            for vector in test_inputs(shape):
                expected = evaluate_reference(data, vector.tolist())
                with torch.no_grad():
                    np.testing.assert_allclose(model(torch.from_numpy(vector.copy())).numpy(), expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(interpret_fragment(payload, vector.reshape(-1)), expected, atol=1e-12, rtol=1e-12)
                manual = dict(payload, hecate_source=(ROOT / f"golden_cases/spatial/{name}.py").read_text())
                validate_function(manual["hecate_source"], payload["public_constants"], payload["layout"]["output_ciphertexts"])
                np.testing.assert_allclose(interpret_fragment(manual, vector.reshape(-1)), expected, atol=1e-12, rtol=1e-12)

    def test_sparse_pool_skips_zero_scalars_and_zero_rows_have_explicit_abi(self):
        from model_catalog import build_model
        from fx_to_hecate import translate, UnsupportedModel
        model, shape = build_model(load("conv-square-pool"))
        payload = translate(model, shape)
        self.assertNotIn(0.0, [v for v in payload["public_constants"].values() if type(v) is float])
        data = load("conv1d-valid")
        data["constants"]["w"] = [[[0., 0.]]]
        model, shape = build_model(data)
        zero = translate(model, shape)
        self.assertEqual(zero['static_check']['contract'], 'hecate-function-v4')
        self.assertEqual(zero['layout']['auxiliary_ciphertexts'][0]['dsl_name'], 'zero_ct')


@unittest.skipUnless(os.environ.get("POSEIDON_SPATIAL_RESULTS"), "Requires real spatial candidate evidence")
class SpatialEvidenceTests(unittest.TestCase):
    def test_all_goldens_and_wrong_windows_are_real_and_immutable(self):
        report = json.loads((Path(os.environ["POSEIDON_SPATIAL_RESULTS"]) / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 11)
        self.assertEqual({c["case"] for c in report["cases"] if not c["counterexample"]}, set(CASES))
        for item in report["cases"]:
            self.assertTrue(item["matched_expected"])
            run = Path(item["run"])
            data = json.loads((run / "report.json").read_text())
            self.assertEqual(data["agent_calls"], 0)
            self.assertFalse(data["llm_generation_validated"])
            self.assertFalse(data["poseidon_gpu_validated"])
            self.assertEqual(data["parameters"]["security_check"], "tc128")
            self.assertEqual(data["parameters"]["modulus_bits"], [60]*14)
            self.assertTrue(all(json.loads((run / "probe/probe.json").read_text()).values()))
            attempt = data["attempts"][0]
            self.assertTrue(attempt["execution"]["encrypted_execution"])
            self.assertFalse(attempt["execution"]["bootstrap_executed"])
            self.assertEqual(attempt["execution"]["input_batches"], 4)
            self.assertFalse(attempt["trace"]["candidate_python_executed"])
            self.assertEqual(attempt["comparison"]["passed"], not item["counterexample"])
            self.assertEqual(attempt["comparison"]["atol"], 1e-5)
            self.assertEqual(attempt["comparison"]["rtol"], 1e-4)
            self.assertGreater(attempt["artifact_gate"]["opcode_counts"].get("1", 0), 0)
            self.assertGreater(attempt["artifact_gate"]["opcode_counts"].get("9", 0), 0)
            if item["case"] == "conv-square-pool":
                self.assertGreater(attempt["artifact_gate"]["opcode_counts"].get("8", 0), 0)
            if item["counterexample"]:
                self.assertEqual(attempt["failure_layer"], "numerical_comparison")
            for name, expected in data["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), expected)
            for name, expected in attempt["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((run / "attempt-00/output" / name).read_bytes()).hexdigest(), expected)

    @unittest.skipUnless(os.environ.get("POSEIDON_SPATIAL_REPAIR_RESULTS"), "Requires actual repaired composition")
    def test_exact_zero_elimination_repaired_actual_runtime(self):
        root = Path(os.environ["POSEIDON_SPATIAL_REPAIR_RESULTS"])
        data = json.loads((root / "report.json").read_text())
        self.assertEqual(data["status"], "passed")
        item = data["cases"][0]
        self.assertEqual(item["case"], "conv-square-pool")
        self.assertTrue(item["execution"]["encrypted_execution"])
        self.assertTrue(item["comparison"]["passed"])
        payload = json.loads((root / "case-000/translation.json").read_text())
        self.assertNotIn(0., [v for v in payload["public_constants"].values() if type(v) is float])


if __name__ == "__main__":
    unittest.main()
