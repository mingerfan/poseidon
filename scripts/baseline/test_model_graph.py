"""Custom graphs: independent arithmetic and rejection gates, not FHE evidence."""
import copy
import json
from pathlib import Path
import unittest

from model_graph import OPS, build_graph_model, evaluate_reference, validate_graph

ROOT = Path(__file__).resolve().parent


def example():
    return json.loads((ROOT / "cases/custom-subneg-linear.json").read_text())


class GraphContractTests(unittest.TestCase):
    def test_custom_graph_is_not_a_catalog_selector(self):
        data = example()
        self.assertNotIn("family", data)
        self.assertNotIn("configuration", data)
        report = validate_graph(data)
        self.assertEqual(report["output_shape"], [2])
        self.assertFalse(report["compiled"])
        self.assertFalse(report["encrypted_execution"])

    def test_reference_hand_calculated(self):
        # offset - x = [-.375,.75,.25,1.5], then row dot products + bias.
        self.assertEqual(evaluate_reference(example(), [.5, -1., .25, -.75]), [.90625, .0625])

    def test_weights_are_user_values_not_generated_seeds(self):
        data = example()
        before = evaluate_reference(data, [0., 0., 0., 0.])
        data["constants"]["weight"][0][0] = .2
        after = evaluate_reference(data, [0., 0., 0., 0.])
        self.assertNotEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])

    def test_arbitrary_graph_connections_and_ordered_subtraction(self):
        data = example()
        data["nodes"] = [
            {"id": "a", "op": "negate", "inputs": ["x"]},
            {"id": "b", "op": "subtract", "inputs": ["x", "a"]}]
        data["output"] = "b"
        self.assertEqual(evaluate_reference(data, [1., -1., .5, 0.]), [2., -2., 1., 0.])

    def test_no_python_or_runtime_override_fields(self):
        for key in ("python", "command", "inputs", "atol", "security", "reference", "path"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_graph(dict(example(), **{key: "override"}))

    def test_reject_nonfinite_bool_ragged_oversized_constants(self):
        for value in (float("nan"), float("inf"), True, 1025, 10**1000, [], [[1], [1, 2]], [[[1]]], [1] * 129):
            data = example()
            data["constants"]["offset"] = value
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                validate_graph(data)

    def test_reject_input_shapes_not_yet_encrypted_verified(self):
        for shape in ([8], [2, 3], [], [True, 4], [4, 1]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                validate_graph(dict(example(), input_shape=shape))

    def test_reject_unsupported_ops_without_approximation(self):
        for op in ("relu", "silu", "conv1d", "conv2d", "avg_pool", "max_pool", "batch_norm", "concat",
                   "reshape", "rotate", "bootstrap", "eval", "__import__"):
            data = example()
            data["nodes"][0]["op"] = op
            with self.subTest(op=op), self.assertRaises(ValueError):
                validate_graph(data)

    def test_reject_cycles_forward_refs_duplicate_names_and_arity(self):
        for changes in ({"inputs": ["logits", "offset"]}, {"id": "x"}, {"id": "offset"},
                        {"id": "../escape"}, {"inputs": ["x"]}, {"extra": 1}):
            data = example()
            data["nodes"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_graph(data)

    def test_reject_bad_linear_and_broadcast(self):
        for field, value in (("weight", [[1, 2]]), ("bias", [1, 2, 3]), ("offset", [1, 2])):
            data = example()
            data["constants"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_graph(data)

    def test_reject_cipher_layout_merge(self):
        data = example()
        data["nodes"].append({"id": "bad", "op": "add", "inputs": ["x", "logits"]})
        data["output"] = "bad"
        with self.assertRaisesRegex(ValueError, "packing"):
            validate_graph(data)

    def test_reference_does_not_mutate_definition_or_input(self):
        data, vector = example(), [.1, -.5, .8, -.2]
        before, input_before = copy.deepcopy(data), list(vector)
        evaluate_reference(data, vector)
        self.assertEqual(data, before)
        self.assertEqual(vector, input_before)

    def test_power_and_limits(self):
        for power in (2, 4):
            data = dict(example(), nodes=[{"id": "y", "op": "power", "inputs": ["x"], "exponent": power}], output="y")
            self.assertEqual(evaluate_reference(data, [0., 1., -1., .5]), [0., 1., 1., .5 ** power])
        for power in (True, 3, -1, 1000):
            data["nodes"][0]["exponent"] = power
            with self.assertRaises(ValueError):
                validate_graph(data)
        with self.assertRaises(ValueError):
            validate_graph(dict(example(), nodes=[]))


try:
    import torch
    import numpy as np
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires pinned Torch: run_model_batch.py --unit-tests")
class GraphTranslationTests(unittest.TestCase):
    def check_graph(self, data):
        from fx_to_hecate import translate
        from model_catalog import build_model, test_inputs, validate_descriptor
        from test_fx_to_hecate import interpret_fragment
        self.assertEqual(validate_descriptor(data), ("custom_graph", None))
        model, shape = build_model(data)
        payload = translate(model, shape)
        for vector in test_inputs(shape):
            expected = evaluate_reference(data, vector.tolist())
            with torch.no_grad():
                np.testing.assert_allclose(model(torch.from_numpy(vector.copy())).numpy(), expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(interpret_fragment(payload, vector.reshape(-1)), expected, atol=1e-12, rtol=1e-12)
        return payload

    def test_subneg_linear_and_manual_golden(self):
        from hecate_contract import validate_function
        from model_catalog import test_inputs
        from test_fx_to_hecate import interpret_fragment
        data = example()
        payload = self.check_graph(data)
        # Reviewed separately from the emitter: x-offset, -(x-offset), two row dots.
        self.assertEqual(payload["public_constants"], {
            "c0": [-.125, .25, -.5, -.75], "c1": [-1.],
            "c2": [.5, -.25, .125, .75], "c3": .125,
            "c4": [-.5, .25, .5, -.125], "c5": -.25})
        payload["hecate_source"] = (ROOT / "golden_cases/subneg_linear/golden.py").read_text()
        validate_function(payload["hecate_source"], payload["public_constants"], 2)
        for vector in test_inputs([4]):
            np.testing.assert_allclose(interpret_fragment(payload, vector), evaluate_reference(data, vector.tolist()), atol=1e-12)

    def test_cipher_subtract_and_non_dyadic_weights(self):
        data = example()
        data["constants"]["weight"][0][0] = .173
        data["nodes"].insert(1, {"id": "difference", "op": "subtract", "inputs": ["x", "centered"]})
        data["nodes"][2]["inputs"] = ["difference"]
        self.check_graph(data)

    def test_every_declared_graph_operator(self):
        data = example()
        nodes = [
            {"id": "flat", "op": "flatten", "inputs": ["x"]},
            {"id": "mul", "op": "multiply", "inputs": ["flat", "offset"]},
            {"id": "plus", "op": "add", "inputs": ["mul", "flat"]},
            {"id": "sq", "op": "square", "inputs": ["plus"]},
            {"id": "pow4", "op": "power", "inputs": ["sq"], "exponent": 4},
            {"id": "sub", "op": "subtract", "inputs": ["pow4", "flat"]},
            {"id": "neg", "op": "negate", "inputs": ["sub"]},
            {"id": "out", "op": "linear", "inputs": ["neg"], "weight": "weight", "bias": "bias"}]
        data.update(input_shape=[2, 2], nodes=nodes, output="out")
        from spatial_ops import OPS as spatial_ops
        # BatchNorm is tested separately with explicit N,C,... layouts.
        self.assertEqual({n["op"] for n in nodes}, OPS - {"rotate","batch_norm","reshape","concat"} - spatial_ops)
        self.check_graph(data)

    def test_caller_mutation_cannot_change_constructed_graph(self):
        data = example()
        model, _ = build_graph_model(data)
        expected = model(torch.zeros(4, dtype=torch.float64)).numpy().copy()
        data["constants"]["bias"][0] = 999
        data["nodes"][0]["op"] = "add"
        np.testing.assert_array_equal(model(torch.zeros(4, dtype=torch.float64)).numpy(), expected)

    def test_all_rotation_steps_reference_torch_fx_and_golden(self):
        from hecate_contract import validate_function
        from test_fx_to_hecate import interpret_fragment
        from model_catalog import test_inputs
        for step in (-3, -2, -1, 1, 2, 3):
            suffix = ("minus" if step < 0 else "plus") + str(abs(step))
            data = json.loads((ROOT / f"cases/rotate-{suffix}.json").read_text())
            payload = self.check_graph(data)
            self.assertEqual(payload["static_check"]["contract"], "hecate-function-v2")
            payload["hecate_source"] = (ROOT / f"golden_cases/rotations/{suffix}.py").read_text()
            validate_function(payload["hecate_source"], {}, contract="hecate-function-v2")
            for vector in test_inputs([4]):
                np.testing.assert_array_equal(interpret_fragment(payload, vector), evaluate_reference(data, vector.tolist()))

    def test_custom_request_accepted_by_provider_without_reference(self):
        from candidate_contract import make_request
        from deepseek_provider import public_request
        data = example()
        payload = self.check_graph(data)
        request = make_request(payload, data, "a" * 64)
        self.assertEqual(public_request(request), request)
        self.assertNotIn("reference", request)
        self.assertNotIn("hecate_source", request)


if __name__ == "__main__":
    unittest.main()
