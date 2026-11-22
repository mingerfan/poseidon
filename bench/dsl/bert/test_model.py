"""CPU-only slot simulation against independent dense polynomial BERT math.

Run from the repository root with:
    python -B -m unittest discover -s bench/dsl/bert -p test_model.py -v

The simulator intentionally has no scalar or Plain-left arithmetic overloads,
so it checks the minimal backend interface rather than NumPy broadcasting.
No CKKS noise, CUDA, Hecate, downloads or external model package is involved.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

if __package__:
    from .model import BertConfig, build_model, create_weights, linear
else:
    from model import BertConfig, build_model, create_weights, linear


class NumpyPlain:
    """Opaque full-capacity plaintext, with no arithmetic operators."""

    def __init__(self, values):
        self.values = values.copy()


class NumpyExpr:
    """Cyclic slots plus operation-depth instrumentation, not a CKKS emulator."""

    def __init__(self, backend, values, ciphertext_depth=0, multiply_depth=0):
        self.backend = backend
        self.values = np.asarray(values, dtype=np.float64).copy()
        self.ciphertext_depth = ciphertext_depth
        self.multiply_depth = multiply_depth
        backend.max_ciphertext_depth = max(backend.max_ciphertext_depth, ciphertext_depth)
        backend.max_multiply_depth = max(backend.max_multiply_depth, multiply_depth)

    def _binary(self, other, operation, multiply=False):
        if not isinstance(other, (NumpyExpr, NumpyPlain)):
            raise TypeError("only Expr/Plain operands are supported")
        if isinstance(other, NumpyExpr):
            if other.backend is not self.backend:
                raise ValueError("expressions must share a backend")
            ct_depth = max(self.ciphertext_depth, other.ciphertext_depth) + int(multiply)
            depth = max(self.multiply_depth, other.multiply_depth) + int(multiply)
            if multiply:
                self.backend.ciphertext_products += 1
        else:
            ct_depth = self.ciphertext_depth
            depth = self.multiply_depth + int(multiply)
            if multiply:
                self.backend.plaintext_products += 1
        return NumpyExpr(self.backend, operation(self.values, other.values), ct_depth, depth)

    def __add__(self, other):
        return self._binary(other, np.add)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __mul__(self, other):
        return self._binary(other, np.multiply, multiply=True)

    def rotate(self, shift):
        if not isinstance(shift, int):
            raise TypeError("rotation must use a Python integer")
        self.backend.rotations.append(shift)
        return NumpyExpr(self.backend, np.roll(self.values, -shift),
                         self.ciphertext_depth, self.multiply_depth)


class NumpyBackend:
    """Plain factory and explicit identity bootstrap callbacks for CPU tests."""

    def __init__(self, slots, dirty_boot_tail_start=None):
        self.slots = slots
        self.rotations = []
        self.boots = []
        self.plaintext_products = 0
        self.ciphertext_products = 0
        self.max_ciphertext_depth = 0
        self.max_multiply_depth = 0
        self.dirty_boot_tail_start = dirty_boot_tail_start

    def plain(self, values):
        if not isinstance(values, np.ndarray) or values.shape != (self.slots,):
            raise ValueError("plain expects exactly one full-capacity 1D ndarray")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("plain expects finite real values")
        return NumpyPlain(values)

    def encrypt(self, prefix, tail=0.0):
        values = np.full(self.slots, tail, dtype=np.float64)
        values[:len(prefix)] = prefix
        return NumpyExpr(self, values)

    def bootstrap(self, expr, label=""):
        self.boots.append((label, expr.ciphertext_depth, expr.multiply_depth))
        values = expr.values.copy()
        if self.dirty_boot_tail_start is not None:
            values[self.dirty_boot_tail_start:] = 10000.0
        return NumpyExpr(self, values)


def dense_reference(inputs, weights, config):
    """Independent matrix implementation of the *chosen* approximations.

    This reference shares no rotation, masking, BSGS, normalization, polynomial
    or attention helper with model.py. Its semantic tensors contain no tails.
    """
    hidden = np.asarray(inputs, dtype=np.float64).copy()
    traces = []

    def affine(x, layer, name):
        return x @ np.asarray(layer[name + "_weight"], dtype=np.float64).T + layer[name + "_bias"]

    def norm(x, gamma, beta):
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = np.mean(centered ** 2, axis=-1, keepdims=True) + config.ln_epsilon
        u = variance / config.ln_variance_center - 1
        inverse = (1 - u / 2 + 3 * u ** 2 / 8 - 5 * u ** 3 / 16)
        inverse /= math.sqrt(config.ln_variance_center)
        return centered * inverse * gamma + beta

    for layer in weights:
        query = affine(hidden, layer, "q").reshape(config.tokens, config.heads, config.head_dim)
        key = affine(hidden, layer, "k").reshape(config.tokens, config.heads, config.head_dim)
        value = affine(hidden, layer, "v").reshape(config.tokens, config.heads, config.head_dim)
        # [query token, head, key token]: all keys, not a causal prefix.
        scores = np.einsum("thd,khd->thk", query, key) / math.sqrt(config.head_dim)
        centered = scores - np.mean(scores, axis=-1, keepdims=True)
        exponential = 1 + centered + centered ** 2 / 2 + centered ** 3 / 6 + centered ** 4 / 24
        denominator = np.sum(exponential, axis=-1, keepdims=True)
        inverse = np.full_like(denominator, 1 / config.tokens)
        for _ in range(config.reciprocal_iterations):
            inverse = inverse * (2 - denominator * inverse)
        probabilities = exponential * inverse
        context = np.einsum("thk,khd->thd", probabilities, value).reshape(config.tokens, config.hidden)
        attention_residual = hidden + affine(context, layer, "o")
        attended = norm(attention_residual, layer["attention_ln_gamma"], layer["attention_ln_beta"])
        up = affine(attended, layer, "ffn_up")
        activated = up / 2 + up ** 2 / math.sqrt(2 * math.pi) - up ** 4 / (6 * math.sqrt(2 * math.pi))
        ffn_residual = attended + affine(activated, layer, "ffn_down")
        hidden = norm(ffn_residual, layer["ffn_ln_gamma"], layer["ffn_ln_beta"])
        traces.append({"probabilities": probabilities, "attention_residual": attention_residual,
                       "attention_ln": attended, "ffn_up": up, "gelu": activated,
                       "ffn_residual": ffn_residual, "output": hidden.copy()})
    return hidden, traces


def small_config(**overrides):
    values = dict(tokens=2, layers=2, hidden=8, intermediate=16, heads=2,
                  head_dim=4, slots=64, weight_std=0.08)
    values.update(overrides)
    return BertConfig(**values)


def fixture_inputs(config, seed=11):
    values = np.random.default_rng(seed).normal(size=(config.tokens, config.hidden))
    values -= values.mean(axis=-1, keepdims=True)
    values /= np.sqrt(np.mean(values ** 2, axis=-1, keepdims=True))
    return values


def simulate(config, inputs, weights, policy="none", tail=0.0, dirty_boot_tail_start=None):
    backend = NumpyBackend(config.slots, dirty_boot_tail_start)
    encrypted = [backend.encrypt(row, tail=tail) for row in inputs]
    saved = [expr.values.copy() for expr in encrypted]
    outputs = build_model(backend, encrypted, weights, config, boot_policy=policy)
    for expr, before in zip(encrypted, saved):
        np.testing.assert_array_equal(expr.values, before)
    return backend, outputs


class RectangularBSGSTests(unittest.TestCase):
    def test_rectangles_and_baby_reuse(self):
        rng = np.random.default_rng(123)
        # Wide expansion, contraction, non-power-of-two dimensions, awkward
        # baby/chunk sizes, and the physical-ring-size boundary are covered.
        for in_size, out_size, slots, baby, chunk in (
                (8, 16, 64, 2, 8), (5, 17, 64, 3, 5),
                (13, 5, 64, 4, 7), (8, 8, 8, 3, 8),
                (3, 32, 32, 2, 4), (1, 7, 8, 1, 1)):
            with self.subTest(shape=(out_size, in_size), baby=baby, chunk=chunk):
                weight = rng.normal(size=(out_size, in_size))
                bias = rng.normal(size=out_size)
                prefix = rng.normal(size=in_size)
                backend = NumpyBackend(slots)
                expr = backend.encrypt(prefix, tail=9000.0)
                output = linear(backend, expr, weight, bias, slots=slots,
                                baby_step=baby, chunk_size=chunk)
                np.testing.assert_allclose(output.values[:out_size], weight @ prefix + bias,
                                           rtol=2e-13, atol=2e-13)
                np.testing.assert_array_equal(output.values[out_size:], 0)
                padded = 1 << (in_size - 1).bit_length()
                used_baby = min(baby, padded)
                chunks = math.ceil(out_size / min(chunk, padded))
                expected_rotations = (int(padded < slots) + used_baby - 1
                                      + chunks * (math.ceil(padded / used_baby) - 1)
                                      + chunks - 1)
                self.assertEqual(len(backend.rotations), expected_rotations)
                # In particular, babies are NOT recomputed for each chunk.
                for shift in range(1, used_baby):
                    self.assertEqual(backend.rotations.count(shift), 1)

    def test_one_ciphertext_can_hold_3072_outputs(self):
        # Small input makes this an inexpensive actual 3072-slot result test.
        slots = 32768
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(3072, 8))
        prefix = rng.normal(size=8)
        backend = NumpyBackend(slots)
        output = linear(backend, backend.encrypt(prefix, tail=-1000), weight,
                        slots=slots, baby_step=4)
        np.testing.assert_allclose(output.values[:3072], weight @ prefix, rtol=2e-13, atol=2e-13)
        np.testing.assert_array_equal(output.values[3072:], 0)
        self.assertIsInstance(output, NumpyExpr)

    def test_default_chunking_reduces_rotations(self):
        backend = NumpyBackend(128)
        weight = np.random.default_rng(9).normal(size=(64, 16))
        linear(backend, backend.encrypt(np.ones(16)), weight, slots=128)
        self.assertEqual(len(backend.rotations), 19)
        self.assertLess(len(backend.rotations), sum(weight.shape) - 2)


class EncoderArithmeticTests(unittest.TestCase):
    def assert_prefixes(self, outputs, expected, config, tolerance=3e-11):
        self.assertEqual(len(outputs), config.tokens)
        actual = np.stack([expr.values[:config.hidden] for expr in outputs])
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        for expr in outputs:
            np.testing.assert_array_equal(expr.values[config.hidden:], 0)

    def test_dense_polynomial_reference_two_layers(self):
        config = small_config()
        weights = create_weights(config, seed=4)
        inputs = fixture_inputs(config)
        expected, trace = dense_reference(inputs, weights, config)
        backend, outputs = simulate(config, inputs, weights, tail=7654.0)
        self.assert_prefixes(outputs, expected, config)
        self.assertEqual(backend.boots, [])
        self.assertGreater(backend.ciphertext_products, 0)
        self.assertEqual(len(trace), 2)

    def test_padded_token_rows_and_non_power_of_two_head_dim(self):
        # tokens=3 exercises softmax padding; head_dim=3 and hidden=6
        # exercise masked head extraction and non-power-of-two LN reductions.
        config = small_config(tokens=3, hidden=6, heads=2, head_dim=3,
                              intermediate=13, bsgs_baby_step=3, linear_chunk_size=5)
        weights = create_weights(config, seed=31)
        inputs = fixture_inputs(config, seed=5)
        expected, _ = dense_reference(inputs, weights, config)
        _, outputs = simulate(config, inputs, weights, tail=-9876.0)
        self.assert_prefixes(outputs, expected, config)

    def test_bidirectional_future_token_dependency(self):
        config = small_config(layers=1)
        weights = create_weights(config, seed=1)
        inputs = fixture_inputs(config)
        _, original = simulate(config, inputs, weights)
        changed = inputs.copy()
        changed[1] = np.roll(changed[1], 2) + np.arange(config.hidden) / 5
        _, updated = simulate(config, changed, weights)
        self.assertGreater(np.max(np.abs(original[0].values[:8] - updated[0].values[:8])), 1e-5)
        expected, trace = dense_reference(changed, weights, config)
        self.assert_prefixes(updated, expected, config)
        self.assertTrue(np.all(trace[0]["probabilities"][0, :, 1] > 0))
        # A one-token run has no future key and must differ at query zero.
        one = replace(config, tokens=1)
        _, isolated = simulate(one, inputs[:1], weights)
        self.assertGreater(np.max(np.abs(original[0].values[:8] - isolated[0].values[:8])), 1e-5)

    def test_layer_norm_subtracts_mean_not_rmsnorm(self):
        config = small_config(layers=1)
        weights = create_weights(config, seed=2)
        for name, values in weights[0].items():
            values.fill(1 if name.endswith("_gamma") else 0)
        inputs = fixture_inputs(config) + 3.0
        _, outputs = simulate(config, inputs, weights)
        _, shifted = simulate(config, inputs + 4.0, weights)
        expected, _ = dense_reference(inputs, weights, config)
        self.assert_prefixes(outputs, expected, config)
        for output, translated in zip(outputs, shifted):
            self.assertAlmostEqual(output.values[:config.hidden].mean(), 0, places=12)
            self.assertGreater(np.linalg.norm(output.values[:config.hidden]), 1)
            np.testing.assert_allclose(output.values, translated.values, atol=1e-12, rtol=1e-12)

    def test_dirty_input_and_bootstrap_tails(self):
        config = small_config(layers=1, tokens=3)
        weights = create_weights(config, seed=77)
        inputs = fixture_inputs(config)
        _, clean = simulate(config, inputs, weights, policy="stage")
        backend, dirty = simulate(config, inputs, weights, policy="stage", tail=1e5,
                                  dirty_boot_tail_start=32)
        self.assertEqual(len(backend.boots), 15 * config.tokens)
        for expected, actual in zip(clean, dirty):
            np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-12)

    def test_explicit_stage_schedule_and_depth(self):
        for iterations, expected_depth in ((3, 4), (4, 6)):
            with self.subTest(iterations=iterations):
                config = small_config(reciprocal_iterations=iterations)
                weights = create_weights(config, seed=19)
                inputs = fixture_inputs(config)
                expected, _ = dense_reference(inputs, weights, config)
                backend, outputs = simulate(config, inputs, weights, policy="stage", tail=42)
                self.assert_prefixes(outputs, expected, config)
                labels = [boot[0] for boot in backend.boots]
                self.assertEqual(len(labels), 15 * config.tokens * config.layers)
                self.assertEqual(len(labels), len(set(labels)))
                for layer in range(config.layers):
                    for token in range(config.tokens):
                        prefix = f"layer{layer}"
                        expected_labels = {
                            f"{prefix}.{name}.token{token}" for name in
                            ("q", "k", "v", "attention_residual", "ffn_up", "gelu", "ffn_residual")}
                        expected_labels.update(
                            f"{prefix}.attention.token{token}.{suffix}" for suffix in
                            ("scores", "softmax.exp", "softmax.reciprocal", "context"))
                        expected_labels.update(
                            f"{prefix}.{norm}.token{token}.{suffix}"
                            for norm in ("attention_ln", "ffn_ln") for suffix in ("variance", "output"))
                        self.assertTrue(expected_labels.issubset(labels))
                self.assertEqual(backend.max_ciphertext_depth, expected_depth)
                self.assertGreater(backend.max_multiply_depth, backend.max_ciphertext_depth)

    def test_no_boot_policy_needs_no_boot_method(self):
        config = small_config(layers=1)
        implementation = NumpyBackend(config.slots)

        class PlainOnly:
            plain = implementation.plain

        inputs = fixture_inputs(config)
        weights = create_weights(config, seed=12)
        encrypted = [implementation.encrypt(row) for row in inputs]
        outputs = build_model(PlainOnly(), encrypted, weights, config, boot_policy="none")
        expected, _ = dense_reference(inputs, weights, config)
        self.assert_prefixes(outputs, expected, config)
        self.assertEqual(implementation.boots, [])

    def test_full_32768_slot_small_encoder(self):
        config = small_config(layers=1, slots=32768)
        inputs = fixture_inputs(config)
        weights = create_weights(config, seed=9)
        expected, _ = dense_reference(inputs, weights, config)
        _, outputs = simulate(config, inputs, weights, policy="stage", tail=999)
        self.assert_prefixes(outputs, expected, config)

    def test_independent_layer_weights_and_seed(self):
        config = small_config(layers=3)
        weights = create_weights(config, seed=27)
        again = create_weights(config, seed=27)
        for name in weights[0]:
            for index in range(config.layers):
                np.testing.assert_array_equal(weights[index][name], again[index][name])
                self.assertEqual(weights[index][name].dtype, np.float32)
            self.assertFalse(np.shares_memory(weights[0][name], weights[1][name]))
            self.assertFalse(np.array_equal(weights[0][name], weights[1][name]))
        inputs = fixture_inputs(config)
        expected, _ = dense_reference(inputs, weights, config)
        _, outputs = simulate(config, inputs, weights)
        self.assert_prefixes(outputs, expected, config)
        tied = [weights[0]] * config.layers
        _, reused = simulate(config, inputs, tied)
        self.assertGreater(np.max(np.abs(outputs[0].values - reused[0].values)), 1e-4)

    def test_ln_center_and_four_reciprocal_steps(self):
        config = small_config(ln_variance_center=1.25, ln_epsilon=0.003,
                              reciprocal_iterations=4)
        weights = create_weights(config, seed=13)
        inputs = fixture_inputs(config)
        expected, _ = dense_reference(inputs, weights, config)
        _, outputs = simulate(config, inputs, weights, policy="stage")
        self.assert_prefixes(outputs, expected, config)


class ValidationTests(unittest.TestCase):
    def test_bert_base_defaults(self):
        config = BertConfig()
        self.assertEqual((config.hidden, config.intermediate, config.heads, config.head_dim,
                          config.layers, config.slots), (768, 3072, 12, 64, 12, 32768))

    def test_invalid_configurations(self):
        for changes in (dict(hidden=7), dict(slots=63), dict(intermediate=65),
                        dict(tokens=33), dict(layers=0), dict(tokens=True),
                        dict(reciprocal_iterations=2), dict(ln_variance_center=0),
                        dict(ln_epsilon=float("nan")), dict(bsgs_baby_step=0)):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                small_config(**changes)

    def test_invalid_build_arguments(self):
        config = small_config(layers=1)
        backend = NumpyBackend(config.slots)
        weights = create_weights(config, seed=0)
        inputs = [backend.encrypt(row) for row in fixture_inputs(config)]
        for supplied_inputs, supplied_weights, policy in (
                (inputs[:1], weights, "none"), (inputs, [], "none"),
                (inputs, weights, "automatic")):
            with self.assertRaises(ValueError):
                build_model(backend, supplied_inputs, supplied_weights, config, policy)
        wrong_shape = dict(weights[0], q_weight=np.zeros((8, 7)))
        with self.assertRaises(ValueError):
            build_model(backend, inputs, [wrong_shape], config)
        nonfinite = dict(weights[0], o_bias=np.full(8, np.nan))
        with self.assertRaises(ValueError):
            build_model(backend, inputs, [nonfinite], config)
        missing = dict(weights[0])
        del missing["attention_ln_gamma"]
        with self.assertRaises(ValueError):
            build_model(backend, inputs, [missing], config)


if __name__ == "__main__":
    unittest.main()
