"""Rule-translator semantics tests; plaintext interpreter is NOT FHE evidence."""
import ast
from collections import Counter
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


def interpret_fragment(payload, flat_input):
    """Independent tiny plaintext slot interpreter for unit diagnostics only."""
    values = {name: np.asarray(value, dtype=float) for name, value in payload["public_constants"].items()}
    values["x"] = np.asarray(flat_input, dtype=float)
    function = ast.parse(payload["hecate_source"]).body[0]

    def evaluate(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            return left + right if isinstance(node.op, ast.Add) else left * right
        if isinstance(node, ast.Call):
            from hecate_contract import rotation_literal
            return np.roll(evaluate(node.func.value), -rotation_literal(node.args[0]))
        if isinstance(node, ast.List):
            return [evaluate(n) for n in node.elts]
        raise AssertionError(type(node))

    for statement in function.body[:-1]:
        values[statement.targets[0].id] = evaluate(statement.value)
    output = evaluate(function.body[-1].value)
    if not isinstance(output, list):
        output = [output]
    return np.asarray([output[i][j] for i, j in payload["layout"]["output_selectors"]])


@unittest.skipIf(torch is None, "requires approved pinned CPU Torch environment; use run_model_batch.py --unit-tests")
class FXTranslatorTests(unittest.TestCase):
    def test_catalog_is_48_cases_with_8_families(self):
        from model_catalog import descriptors
        cases = descriptors()
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({c["id"] for c in cases}), 48)
        self.assertEqual(sorted(Counter(c["family"] for c in cases).values()), [6] * 8)

    def test_all_catalog_plaintext_semantics_and_exact_state_reproducibility(self):
        from model_catalog import descriptors, build_model, test_inputs
        from fx_to_hecate import translate
        for descriptor in descriptors():
            with self.subTest(case=descriptor["id"]):
                model, shape = build_model(descriptor)
                second, _ = build_model(descriptor)
                self.assertTrue(all(torch.equal(t, second.state_dict()[k]) for k, t in model.state_dict().items()))
                payload = translate(model, shape)
                self.assertFalse(payload["static_check"]["encrypted_correctness_checked"])
                for vector in test_inputs(shape):
                    with torch.no_grad():
                        reference = model(torch.from_numpy(vector.copy())).numpy()
                    actual = interpret_fragment(payload, vector.reshape(-1))
                    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)

    def test_training_dtype_shape_and_nonfinite_rejected(self):
        from fx_to_hecate import translate, UnsupportedModel
        model = torch.nn.Linear(4, 2, dtype=torch.float64)
        with self.assertRaisesRegex(UnsupportedModel, "eval"):
            translate(model, [4])
        with self.assertRaisesRegex(UnsupportedModel, "float64"):
            translate(torch.nn.Linear(4, 2).eval(), [4])
        with self.assertRaisesRegex(UnsupportedModel, "input shapes"):
            translate(model.eval(), [8])
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(UnsupportedModel, "invalid"):
            translate(model, [4])

    def test_relu_random_and_data_branch_rejected(self):
        from fx_to_hecate import translate, UnsupportedModel

        class Random(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand_like(x)

        class Branch(torch.nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x * x
                return x

        for model in (torch.nn.ReLU(), Random(), Branch()):
            with self.assertRaises(UnsupportedModel):
                translate(model.eval(), [4])

    def test_kwargs_reshape_and_implicit_multidim_linear_rejected(self):
        from fx_to_hecate import translate, UnsupportedModel

        class Alpha(torch.nn.Module):
            def forward(self, x):
                return torch.add(x, x, alpha=2)

        class Reshape(torch.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1)

        for model, shape in ((Alpha(), [4]), (Reshape(), [2, 2]), (torch.nn.Linear(2, 2, dtype=torch.float64), [2, 2])):
            with self.assertRaises(UnsupportedModel):
                translate(model.eval(), shape)

    def test_custom_non_catalog_graph_and_public_left_arithmetic(self):
        from fx_to_hecate import translate

        class Independent(torch.nn.Module):
            def forward(self, x):
                y = 0.125 * x + 0.25
                return torch.square(y) + x

        model = Independent().eval()
        payload = translate(model, [4])
        x = np.array([.1, -.8, .6, .3])
        np.testing.assert_allclose(interpret_fragment(payload, x), model(torch.from_numpy(x)).numpy(), atol=1e-12)

    def test_root_linear_functional_call(self):
        from fx_to_hecate import translate
        with torch.random.fork_rng(devices=[]):
            model = torch.nn.Linear(4, 3, dtype=torch.float64).eval()
        payload = translate(model, [4])
        self.assertEqual(payload["layout"]["output_shape"], [3])
        x = np.array([.1, -.8, .6, .3])
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(interpret_fragment(payload, x), expected, atol=1e-12)

    def test_bad_broadcast_rejected(self):
        from fx_to_hecate import translate, UnsupportedModel

        class BadBroadcast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.ones(2, dtype=torch.float64))

            def forward(self, x):
                return x * self.w

        with self.assertRaisesRegex(UnsupportedModel, "broadcast"):
            translate(BadBroadcast().eval(), [4])

    def test_descriptor_rejects_extra_fields_and_identifier_injection(self):
        from model_catalog import descriptors, validate_descriptor
        base = descriptors()[0]
        for changes in ({"id": "../../tmp"}, {"family": "relu"}, {"configuration": True}, {"model_py": "evil.py"}):
            with self.assertRaises(ValueError):
                validate_descriptor(dict(base, **changes))


if __name__ == "__main__":
    unittest.main()
