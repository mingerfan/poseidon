"""User-defined multi-input graph semantics and isolated pipeline evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from candidate_contract import make_request, request_input_names, validate_candidate
from model_graph import validate_graph, evaluate_reference

ROOT = Path(__file__).parent


def case(name="custom-dual-subtract"):
    return json.loads((ROOT / f"cases/{name}.json").read_text())


class Schema3Tests(unittest.TestCase):
    def test_input_list_is_explicit_not_dictionary_order(self):
        data = case()
        self.assertEqual(validate_graph(data)["schema"], 3)
        values = {"right": [4., 3., 2., 1.], "left": [1., 2., 3., 4.]}
        before = copy.deepcopy(values)
        self.assertEqual(evaluate_reference(data, values), [-3., -1., 1., 3.])
        self.assertEqual(values, before)
        with self.assertRaises(ValueError):
            evaluate_reference(data, {"left": values["left"]})

    def test_reject_invalid_names_counts_shapes_and_signature_fields(self):
        for inputs in ([], [{"name": "left", "shape": [4]}],
                       [{"name": "left", "shape": [4]}]*2,
                       [{"name": "arg"+str(i), "shape": [4]} for i in range(5)],
                       [{"name": "left", "shape": [8]}, {"name": "right", "shape": [4]}],
                       [{"name": "left", "shape": [4], "private": False}, {"name": "right", "shape": [4]}],
                       [{"name": "left();", "shape": [4]}, {"name": "right", "shape": [4]}]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                validate_graph(dict(case(), inputs=inputs))
        with self.assertRaises(ValueError):
            validate_graph(dict(case(), input_shape=[4]))
        with self.assertRaises(ValueError):
            validate_graph(dict(case(), constants={"left": 0.5}))

    def test_legacy_schema_is_not_implicitly_multi_input(self):
        data = case()
        data["schema"] = 2
        with self.assertRaises(ValueError):
            validate_graph(data)

    def test_custom_weights_determine_reference_not_case_name(self):
        data = case("custom-dual-linear")
        data["id"] = "user-chosen-name"
        inputs = {"features": [[1., 0.], [0., 0.]], "residual": [0.]*4}
        expected = evaluate_reference(data, inputs)
        self.assertAlmostEqual(expected[0], 0.16)
        self.assertAlmostEqual(expected[1], 0.36)
        data["constants"]["weights"][0][0] = 0.6
        self.assertAlmostEqual(evaluate_reference(data, inputs)[0], 0.49)


try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires pinned Torch")
class Schema3TorchTests(unittest.TestCase):
    def check(self, data, golden=None):
        from model_catalog import build_model
        from fx_to_hecate import translate
        from multi_input_fixtures import fixture_inputs
        from candidate_trace import evaluate_tree
        from deepseek_provider import public_request
        model, shape = build_model(data)
        payload = translate(model, shape)
        request = make_request(payload, data, "a"*64)
        self.assertEqual(public_request(request), request)
        self.assertEqual(request["task"], "hecate-function-synthesis-v4")
        names = request_input_names(request)
        self.assertEqual([s["name"] for s in payload["layout"]["inputs"]], [s["name"] for s in data["inputs"]])
        source = (ROOT / f"golden_cases/multi_input/{golden}.py").read_text() if golden else payload["hecate_source"]
        validate_candidate(dict(schema=1, request_id=request["request_id"], hecate_source=source), request)

        class Slots:
            # Pure logical-slot test model, NEVER reported as encrypted execution.
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float64)
            def __add__(self, b):
                return Slots(self.values+(b.values if isinstance(b, Slots) else b))
            def __sub__(self, b):
                return Slots(self.values-(b.values if isinstance(b, Slots) else b))
            def __mul__(self, b):
                return Slots(self.values*(b.values if isinstance(b, Slots) else b))
            def rotate(self, step):
                return Slots(np.roll(self.values, -step))

        for flat in fixture_inputs(len(names)):
            logical = [np.asarray(v).reshape(s["shape"]) for v, s in zip(flat, data["inputs"])]
            expected = evaluate_reference(data, {s["name"]: v.tolist() for s, v in zip(data["inputs"], logical)})
            with torch.no_grad():
                np.testing.assert_allclose(model(*[torch.from_numpy(v.copy()) for v in logical]).numpy(), expected,
                                           atol=1e-12, rtol=1e-12)
            outputs = evaluate_tree(source, payload["public_constants"],
                                    encrypted_inputs={n: Slots(v) for n, v in zip(names, flat)})
            if not isinstance(outputs, list):
                outputs = [outputs]
            actual = [outputs[i].values[j] for i, j in payload["layout"]["output_selectors"]]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        return request

    def test_all_user_descriptors_rule_and_manual_golden(self):
        for name, golden in (("custom-dual-subtract", "ordered_subtract"),
                             ("custom-dual-linear", "custom_dual_linear"),
                             ("custom-triple-merge", "triple_merge"), ("custom-quad-merge", "quad_merge")):
            self.check(case(name))
            self.check(case(name), golden)

    def test_new_names_weights_and_graph_are_not_catalog_selection(self):
        data = case("custom-dual-linear")
        data["id"] = "not-a-catalog-name"
        data["constants"]["weights"][0][0] = 0.173
        data["inputs"][0]["name"] = "observations"
        data["nodes"][0]["inputs"] = ["observations"]
        data["nodes"].append({"id": "negated", "op": "negate", "inputs": [data["output"]]})
        data["output"] = "negated"
        self.check(data)

    def test_request_signature_tampering_and_no_reference_exposure(self):
        from deepseek_provider import public_request, ProviderError
        req = self.check(case())
        for field in ("reference", "test_inputs", "private_key", "hecate_source"):
            self.assertNotIn(field, req)
        altered = copy.deepcopy(req)
        altered["layout"]["inputs"].reverse()
        with self.assertRaises(ValueError):
            request_input_names(altered)
        with self.assertRaises(ProviderError):
            public_request(altered)


@unittest.skipUnless(os.environ.get("POSEIDON_SCHEMA3_RESULTS"), "Requires schema-3 isolated execution evidence")
class Schema3EvidenceTests(unittest.TestCase):
    def test_actual_candidate_sandbox_multi_input_and_negative(self):
        root = Path(os.environ["POSEIDON_SCHEMA3_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 5)
        for item in report["cases"]:
            self.assertTrue(item["matched_expected"])
            run = Path(item["run"])
            data = json.loads((run / "report.json").read_text())
            req = json.loads((run / "request.json").read_text())
            # v4 clarified headers without changing the v3 AST semantics.
            self.assertIn(req["task"], ("hecate-function-synthesis-v3", "hecate-function-synthesis-v4"))
            request_input_names(req)  # Enforce the immutable version/rules pairing.
            self.assertEqual(data["agent_calls"], 0)
            self.assertFalse(data["llm_generation_validated"])
            self.assertFalse(data["poseidon_gpu_validated"])
            self.assertEqual(data["parameters"]["security_check"], "tc128")
            self.assertTrue(all(json.loads((run / "probe/probe.json").read_text()).values()))
            attempt = data["attempts"][0]
            self.assertFalse(attempt["trace"]["candidate_python_executed"])
            self.assertTrue(attempt["execution"]["encrypted_execution"])
            self.assertEqual(attempt["execution"]["encrypted_input_count"], len(request_input_names(req)))
            self.assertEqual(len(attempt["artifact_gate"]["arg_level"]), len(request_input_names(req)))
            self.assertEqual(attempt["comparison"]["passed"], not item["counterexample"])
            for name, expected in data["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), expected)
            for name, expected in attempt["artifact_hashes"].items():
                self.assertEqual(hashlib.sha256((run / "attempt-00/output" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
