"""CPU-only layout and dense-polynomial oracles; no Hecate/GPU dependency.

Run: python -m unittest bench.bert_dsl.test_packed_model -v
These tests validate the selected polynomial workload, not CKKS error or the
accuracy of the polynomials against pretrained BERT's transcendental functions.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

try:
    from . import packed_model as packed
except ImportError:
    import packed_model as packed


class ArrayExpr:
    def __init__(self, values, backend):
        self.values = np.array(values, dtype=np.float64, copy=True)
        self.backend = backend

    def _value(self, other):
        values = other.values if isinstance(other, ArrayExpr) else other
        values = np.asarray(values)
        if values.shape != self.values.shape:
            raise AssertionError("only full-slot Plain or Expr operands are allowed")
        return values

    def __add__(self, other):
        return ArrayExpr(self.values + self._value(other), self.backend)

    def __sub__(self, other):
        return ArrayExpr(self.values - self._value(other), self.backend)

    def __mul__(self, other):
        if isinstance(other, ArrayExpr):
            self.backend.cipher_products += 1
        return ArrayExpr(self.values * self._value(other), self.backend)

    def rotate(self, shift):
        self.backend.rotations += 1
        return ArrayExpr(np.roll(self.values, -shift), self.backend)


class ArrayBackend:
    def __init__(self, config, dirty_boots=False):
        self.config = config
        self.dirty_boots = dirty_boots
        self.labels = []
        self.plains = 0
        self.cipher_products = 0
        self.rotations = 0

    def plain(self, values):
        assert isinstance(values, np.ndarray)
        assert values.shape == (self.config.slots,)
        self.plains += 1
        return values.copy()

    def encrypt(self, values):
        return [ArrayExpr(row, self) for row in values]

    def bootstrap(self, expr, label=""):
        self.labels.append(label)
        values = expr.values.copy()
        if self.dirty_boots:
            width = self.config.intermediate if ".ffn_up." in label or ".gelu." in label else self.config.hidden
            values.reshape(self.config.rows, self.config.row_stride)[:, width:] = 123.0
        return ArrayExpr(values, self)


def dense_attention(queries, keys, values, config):
    """Independent [token,head,feature] math, without cyclic/slot helpers."""
    q = queries.reshape(config.tokens, config.heads, config.head_dim)
    k = keys.reshape(config.tokens, config.heads, config.head_dim)
    v = values.reshape(config.tokens, config.heads, config.head_dim)
    scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(config.head_dim)
    scores -= scores.mean(axis=-1, keepdims=True)
    exp = 1 + scores + scores**2 / 2 + scores**3 / 6 + scores**4 / 24
    denominator = exp.sum(axis=-1, keepdims=True)
    inverse = np.full_like(denominator, 1 / config.tokens)
    for _ in range(config.reciprocal_iterations):
        inverse = inverse * (2 - denominator * inverse)
    # Normalize AFTER the weighted sum, just as the mathematical identity
    # requires; no packed helper is reused to calculate these dense quantities.
    context = np.einsum("hqk,khd->qhd", exp, v)
    context *= inverse[:, :, 0].T[:, :, None]
    return context.reshape(config.tokens, config.hidden)


def dense_layer_norm(values, gamma, beta, config):
    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered**2, axis=-1, keepdims=True) + config.ln_epsilon
    u = variance / config.ln_variance_center - 1
    inverse = (1 - u / 2 + 3 * u**2 / 8 - 5 * u**3 / 16) / math.sqrt(config.ln_variance_center)
    return centered * inverse * gamma + beta


def dense_model(values, weights, config):
    result = np.array(values, dtype=np.float64, copy=True)
    for layer in weights:
        def project(x, name):
            return x @ layer[name + "_weight"].T + layer[name + "_bias"]
        context = dense_attention(project(result, "q"), project(result, "k"), project(result, "v"), config)
        attended = dense_layer_norm(project(context, "o") + result,
                                    layer["attention_ln_gamma"], layer["attention_ln_beta"], config)
        up = project(attended, "ffn_up")
        activated = up / 2 + (up**2 - up**4 / 6) / math.sqrt(2 * math.pi)
        result = dense_layer_norm(project(activated, "ffn_down") + attended,
                                  layer["ffn_ln_gamma"], layer["ffn_ln_beta"], config)
    return result


class PackedModelTests(unittest.TestCase):
    def setUp(self):
        # Two tokens per CT, two CTs. Non-power-of-two hidden/FFN widths test
        # padding, while power-of-two head_dim matches BERT's 64-slot heads.
        self.config = packed.PackedConfig(tokens=4, hidden=6, intermediate=11,
                                          heads=3, slots=32, row_stride=16,
                                          weight_std=0.035)
        self.rng = np.random.default_rng(390)

    def decode(self, expressions, config=None, width=None):
        return packed.unpack_outputs(np.stack([expr.values for expr in expressions]),
                                     self.config if config is None else config, width=width)

    def assert_clean(self, expressions, width, config=None):
        config = self.config if config is None else config
        for expr in expressions:
            tails = expr.values.reshape(config.rows, config.row_stride)[:, width:]
            np.testing.assert_array_equal(tails, np.zeros_like(tails))

    def test_default_128_layout_and_validation(self):
        config = packed.PackedConfig()
        self.assertEqual((config.tokens, config.rows, config.groups, config.head_dim), (128, 8, 16, 64))
        values = np.arange(config.tokens * config.hidden).reshape(config.tokens, config.hidden)
        arrays = packed.pack_inputs(values, config, tail_value=17)
        self.assertEqual(arrays.shape, (16, 32768))
        np.testing.assert_array_equal(packed.unpack_outputs(arrays, config), values)
        for kwargs in ({"tokens": 127}, {"row_stride": 3000}, {"hidden": 769},
                       {"intermediate": 4097}, {"slots": 1024}, {"reciprocal_iterations": 2},
                       {"ln_variance_center": 0}, {"tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                packed.PackedConfig(**kwargs)
        with self.assertRaises(ValueError):
            replace(self.config, hidden=9, heads=3)  # head_dim=3 is unsupported.

    def test_cyclic_gather_cross_group_wrap_and_dirty_tails(self):
        config = self.config
        values = np.arange(config.tokens * config.hidden).reshape(config.tokens, config.hidden) / 10
        backend = ArrayBackend(config)
        inputs = backend.encrypt(packed.pack_inputs(values, config, tail_value=999))
        for shift in range(-config.tokens - 1, config.tokens + 2):
            with self.subTest(shift=shift):
                outputs = packed.shift_tokens(backend, inputs, shift, config=config)
                np.testing.assert_array_equal(self.decode(outputs), np.roll(values, -shift, axis=0))
                self.assert_clean(outputs, config.hidden)
        with self.assertRaises(TypeError):
            packed.shift_tokens(backend, inputs, 0.5, config=config)

    def test_rectangular_bsgs_giant_compensation_and_dirty_rows(self):
        config = self.config
        # Include widths equal to stride: global cyclic wrap must STILL not
        # import an adjacent token, and invalid input tails need not be zero.
        for out_size, in_size in ((11, 6), (6, 11), (16, 5), (5, 16), (1, 16), (16, 1)):
            x = self.rng.normal(size=(config.rows, in_size))
            w = self.rng.normal(size=(out_size, in_size))
            bias = self.rng.normal(size=out_size)
            physical = self.rng.normal(100, 10, size=(config.rows, config.row_stride))
            physical[:, :in_size] = x
            for baby in (1, 3, 7, 50):
                with self.subTest(shape=(out_size, in_size), baby=baby):
                    backend = ArrayBackend(config)
                    expr = ArrayExpr(physical.reshape(-1), backend)
                    output = packed.linear(backend, expr, w, bias, config=config, baby_step=baby)
                    rows = output.values.reshape(config.rows, config.row_stride)
                    np.testing.assert_allclose(rows[:, :out_size], x @ w.T + bias, rtol=2e-13, atol=2e-13)
                    self.assert_clean([output], out_size)
                    self.assertEqual(backend.cipher_products, 0)
                    # Two shared masks/plains (input mask and bias), with one
                    # public diagonal per d, NOT one per token/physical row.
                    self.assertEqual(backend.plains, out_size + in_size + 1)

    def test_attention_dense_oracle_both_newton_counts(self):
        for iterations in (3, 4):
            # Three ciphertext groups and four-feature heads additionally
            # exercise multi-step head reduction and non-power-of-two tokens.
            config = replace(self.config, tokens=6, hidden=8, heads=2,
                             reciprocal_iterations=iterations)
            q, k = self.rng.normal(0, 0.2, size=(2, config.tokens, config.hidden))
            v = self.rng.normal(size=(config.tokens, config.hidden))
            backend = ArrayBackend(config)
            batches = [backend.encrypt(packed.pack_inputs(a, config, tail_value=321)) for a in (q, k, v)]
            outputs = packed.attention(backend, *batches, config=config)
            np.testing.assert_allclose(self.decode(outputs, config), dense_attention(q, k, v, config),
                                       rtol=2e-13, atol=2e-13)
            self.assert_clean(outputs, config.hidden, config)
            self.assertEqual(len(outputs), config.groups)
            self.assertEqual(backend.labels, [])

    def test_uniform_attention_depends_on_future_cross_group_tokens(self):
        config = self.config
        q = np.zeros((config.tokens, config.hidden))
        k = np.zeros_like(q)
        v = np.zeros_like(q)
        v[-1] = np.arange(1, config.hidden + 1)
        backend = ArrayBackend(config)
        batches = [backend.encrypt(packed.pack_inputs(a, config, tail_value=-456)) for a in (q, k, v)]
        outputs = packed.attention(backend, *batches, config=config)
        expected = np.tile(v[-1] / config.tokens, (config.tokens, 1))
        np.testing.assert_allclose(self.decode(outputs), expected, rtol=0, atol=1e-15)
        self.assertTrue(np.all(self.decode(outputs)[0] != 0))  # Not causal/local.

    def test_full_model_dense_polynomials_two_layers_and_dirty_boots(self):
        config = replace(self.config, layers=2, bsgs_baby_step=3)
        x = self.rng.normal(size=(config.tokens, config.hidden))
        x -= x.mean(axis=-1, keepdims=True)
        x /= np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
        # Add substantial row means: RMSNorm or omitted LN centering fails.
        x += np.arange(config.tokens)[:, None] / 3
        weights = packed.create_weights(config, seed=19)
        expected = dense_model(x, weights, config)
        self.assertFalse(np.shares_memory(weights[0]["q_weight"], weights[0]["k_weight"]))
        self.assertFalse(np.array_equal(weights[0]["q_weight"], weights[1]["q_weight"]))
        for policy in ("none", "stage"):
            with self.subTest(policy=policy):
                backend = ArrayBackend(config, dirty_boots=True)
                inputs = backend.encrypt(packed.pack_inputs(x, config, tail_value=500))
                outputs = packed.build_model(backend, inputs, weights, config, boot_policy=policy)
                np.testing.assert_allclose(self.decode(outputs, config), expected, rtol=5e-12, atol=5e-12)
                self.assert_clean(outputs, config.hidden, config)
                self.assertEqual(len(outputs), config.groups)
                expected_boots = 15 * config.groups * config.layers if policy == "stage" else 0
                self.assertEqual(len(backend.labels), expected_boots)
                self.assertEqual(len(backend.labels), len(set(backend.labels)))
                if policy == "stage":
                    self.assertTrue(any("attention.denominator.group" in label for label in backend.labels))
                    self.assertTrue(any("attention_ln.variance.group" in label for label in backend.labels))
                    self.assertFalse(any("token" in label for label in backend.labels))

    def test_more_rows_and_single_group(self):
        for config in (replace(self.config, tokens=8, slots=64),
                       replace(self.config, tokens=4, slots=64),
                       replace(self.config, tokens=4, slots=16)):
            with self.subTest(rows=config.rows, groups=config.groups):
                backend = ArrayBackend(config)
                x = self.rng.normal(0, 0.2, size=(config.tokens, config.hidden))
                weights = packed.create_weights(config, seed=7)
                inputs = backend.encrypt(packed.pack_inputs(x, config, tail_value=202))
                outputs = packed.build_model(backend, inputs, weights, config, boot_policy="none")
                np.testing.assert_allclose(self.decode(outputs, config), dense_model(x, weights, config),
                                           rtol=5e-12, atol=5e-12)

    def test_shared_frontend_graph_compatibility(self):
        if __package__:
            from .backend import Graph
        else:
            from backend import Graph
        config = self.config
        graph = Graph(config.slots)
        inputs = [graph.input(f"group{group}") for group in range(config.groups)]
        weights = packed.create_weights(config, seed=29)
        x = self.rng.normal(0, 0.5, size=(config.tokens, config.hidden))
        physical = packed.pack_inputs(x, config, tail_value=800)
        outputs = packed.build_model(graph, inputs, weights, config, boot_policy="stage")
        evaluated = graph.evaluate(outputs, {expr.data: row for expr, row in zip(inputs, physical)})
        decoded = packed.unpack_outputs(np.stack(evaluated), config)
        np.testing.assert_allclose(decoded, dense_model(x, weights, config), rtol=5e-12, atol=5e-12)
        self.assertEqual(sum(node.op == "boot" for node in graph.reachable(outputs)), 15 * config.groups)

    def test_public_validation(self):
        config = self.config
        backend = ArrayBackend(config)
        x = self.rng.normal(size=(config.tokens, config.hidden))
        inputs = backend.encrypt(packed.pack_inputs(x, config))
        weights = packed.create_weights(config)
        with self.assertRaises(ValueError):
            packed.build_model(backend, inputs[:1], weights, config)
        with self.assertRaises(ValueError):
            packed.build_model(backend, inputs, weights, config, boot_policy="auto")
        with self.assertRaises(ValueError):
            packed.build_model(backend, inputs, [], config)
        bad = [dict(weights[0])]
        bad[0]["q_weight"] = bad[0]["q_weight"].T[:, :-1]
        with self.assertRaises(ValueError):
            packed.build_model(backend, inputs, bad, config)
        bad[0].pop("q_weight")
        with self.assertRaises(ValueError):
            packed.build_model(backend, inputs, bad, config)
        with self.assertRaises(ValueError):
            packed.linear(backend, inputs[0], np.zeros((17, 6)), config=config)
        with self.assertRaises(ValueError):
            packed.attention(backend, inputs, inputs[:1], inputs, config=config)
        with self.assertRaises(ValueError):
            packed.pack_inputs(np.zeros((1, 6)), config)


if __name__ == "__main__":
    unittest.main()
