"""CPU-only compact-layout tests against independent dense polynomial math.

No Hecate, SSH or GPU; only tiny Expr graphs are constructed. Full-size tests
exercise NumPy packing/configuration alone, never allocate real dense weights.
"""

from dataclasses import replace
import math
import unittest

import numpy as np

try:
    from . import cerium_packing as compact
    from . import model
except ImportError:
    import cerium_packing as compact
    import model


class Plain:
    """Intentionally opaque to arithmetic: no operators or array conversion."""
    def __init__(self, values):
        self.values = values.copy()


class ArrayExpr:
    def __init__(self, values, backend):
        self.values = np.array(values, dtype=np.float64, copy=True)
        self.backend = backend

    def _operand(self, other):
        if not isinstance(other, (Plain, ArrayExpr)):
            raise AssertionError("arithmetic requires an opaque Plain or Expr, not a scalar/array")
        if other.values.shape != self.values.shape:
            raise AssertionError("operands must have the full slot shape")
        return other.values

    def __add__(self, other):
        return ArrayExpr(self.values + self._operand(other), self.backend)

    def __sub__(self, other):
        return ArrayExpr(self.values - self._operand(other), self.backend)

    def __mul__(self, other):
        self.backend.products["ct" if isinstance(other, ArrayExpr) else "plain"] += 1
        return ArrayExpr(self.values * self._operand(other), self.backend)

    def rotate(self, offset):
        self.backend.rotations.append(offset)
        return ArrayExpr(np.roll(self.values, -offset), self.backend)


class ArrayBackend:
    def __init__(self, config):
        self.config = config
        self.labels = []
        self.plains = []
        self.rotations = []
        self.products = {"ct": 0, "plain": 0}

    def plain(self, values):
        assert isinstance(values, np.ndarray)
        assert values.shape == (self.config.slots,)
        self.plains.append(values.copy())
        return Plain(values)

    def encrypt(self, values):
        return [ArrayExpr(row, self) for row in values]

    def bootstrap(self, expr, label=""):
        self.labels.append(label)
        return ArrayExpr(expr.values, self)


class HookBackend(ArrayBackend):
    def __init__(self, config):
        super().__init__(config)
        self.linear_calls = []

    def compact_linear(self, inputs, weight, bias, config, label):
        self.linear_calls.append((label, weight, bias, config))
        # A prepared backend may use its own implementation, not the fallback.
        # Here independently decode/project/encode with NumPy for CPU testing.
        source = np.stack([expr.values for expr in inputs])
        width = weight.shape[1]
        dense = source.reshape(width // config.tile, config.tokens, config.tile).transpose(
            1, 0, 2).reshape(config.tokens, width)
        output = dense @ weight.T
        if bias is not None:
            output += bias
        arrays = output.reshape(config.tokens, weight.shape[0] // config.tile, config.tile).transpose(
            1, 0, 2).reshape(weight.shape[0] // config.tile, config.slots)
        return self.encrypt(arrays)


def dense_attention(q, k, v, config):
    """Independent [head,query,key] polynomial oracle, no rotation helpers."""
    queries = q.reshape(config.tokens, config.heads, config.head_dim)
    keys = k.reshape(config.tokens, config.heads, config.head_dim)
    values = v.reshape(config.tokens, config.heads, config.head_dim)
    scores = np.einsum("qhd,khd->hqk", queries, keys) / math.sqrt(config.head_dim)
    scores = scores - scores.mean(axis=-1, keepdims=True)
    exp = 1 + scores + scores**2 / 2 + scores**3 / 6 + scores**4 / 24
    denominator = exp.sum(axis=-1, keepdims=True)
    inverse = np.full_like(denominator, 1 / config.tokens)
    for _ in range(3):
        inverse = inverse * (2 - denominator * inverse)
    context = np.einsum("hqk,khd->qhd", exp, values)
    return (context * inverse[:, :, 0].T[:, :, None]).reshape(config.tokens, config.hidden)


def dense_ln(values, gamma, beta, config):
    centered = values - values.mean(axis=1, keepdims=True)
    variance = (centered * centered).mean(axis=1, keepdims=True) + config.ln_epsilon
    u = variance / config.ln_variance_center - 1
    inverse = (1 - u / 2 + 3 * u**2 / 8 - 5 * u**3 / 16) / math.sqrt(config.ln_variance_center)
    return centered * inverse * gamma + beta


def dense_model(inputs, weights, config):
    hidden = np.array(inputs, dtype=np.float64, copy=True)
    for layer in weights:
        def linear(values, name):
            return values @ layer[name + "_weight"].T + layer[name + "_bias"]
        context = dense_attention(linear(hidden, "q"), linear(hidden, "k"), linear(hidden, "v"), config)
        attended = dense_ln(linear(context, "o") + hidden,
                            layer["attention_ln_gamma"], layer["attention_ln_beta"], config)
        up = linear(attended, "ffn_up")
        gelu = up / 2 + (up**2 - up**4 / 6) / math.sqrt(2 * math.pi)
        hidden = dense_ln(linear(gelu, "ffn_down") + attended,
                          layer["ffn_ln_gamma"], layer["ffn_ln_beta"], config)
    return hidden


class CeriumPackingTests(unittest.TestCase):
    def setUp(self):
        self.config = compact.Config(tokens=4, hidden=12, heads=3, tile=4,
                                     slots=16, intermediate=20, weight_std=0.035,
                                     attention_layout="offset")
        self.rng = np.random.default_rng(390)

    def decode(self, expressions, config=None, width=None):
        config = self.config if config is None else config
        return compact.unpack_outputs(np.stack([expr.values for expr in expressions]), config, width=width)

    def encrypt(self, backend, values, width=None):
        return backend.encrypt(compact.pack_inputs(values, backend.config, width=width))

    def test_default_layout_same_slots_and_full_numpy_packing(self):
        config = compact.Config()
        self.assertEqual((config.tokens, config.hidden, config.heads, config.tile, config.slots,
                          config.intermediate, config.layers), (128, 768, 12, 256, 32768, 3072, 1))
        self.assertEqual((config.hidden_tiles, config.intermediate_tiles, config.head_dim,
                          config.heads_per_tile, config.bsgs_baby_step), (3, 12, 64, 4, 16))
        self.assertEqual(config.attention_layout, "cyclic")
        self.assertEqual(config.attention_groups, 6)
        self.assertEqual(config.stage_boots_per_layer, 92)
        self.assertEqual(replace(config, attention_layout="offset").stage_boots_per_layer, 59)
        # Full-size arrays only: no full-size Expr graph or dense weights.
        for width in (config.hidden, config.intermediate):
            values = np.arange(1, config.tokens * width + 1).reshape(config.tokens, width)
            arrays = compact.pack_inputs(values, config, width=width)
            self.assertEqual(arrays.shape, (width // config.tile, 32768))
            self.assertTrue(np.all(arrays != 0))  # Every single slot carries data.
            for tile in range(width // config.tile):
                np.testing.assert_array_equal(arrays[tile].reshape(128, 256),
                                              values[:, tile * 256:(tile + 1) * 256])
            np.testing.assert_array_equal(compact.unpack_outputs(arrays, config, width=width), values)
            arrays[0, 0] = -1
            self.assertEqual(values[0, 0], 1)  # Client pack is not an alias.

    def test_config_and_public_validation(self):
        for kwargs in ({"tokens": 64}, {"tile": 128}, {"slots": 32767}, {"hidden": 769},
                       {"intermediate": 3073}, {"heads": 16}, {"heads": 7},
                       {"tokens": True}, {"tile": 0}, {"layers": 0},
                       {"bsgs_baby_step": 0}, {"reciprocal_iterations": 4},
                       {"ln_variance_center": 0}, {"ln_epsilon": float("nan")},
                       {"attention_layout": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                compact.Config(**kwargs)
        config = self.config
        backend = ArrayBackend(config)
        inputs = self.encrypt(backend, np.zeros((config.tokens, config.hidden)))
        weights = compact.create_weights(config)
        for policy in ("auto", "fixed"):
            with self.assertRaises(ValueError):
                compact.build_model(backend, inputs, weights, config, boot_policy=policy)
        with self.assertRaises(ValueError):
            compact.build_model(backend, inputs[:1], weights, config)
        with self.assertRaises(ValueError):
            compact.build_model(backend, inputs, [], config)
        for invalid in (np.zeros((11, 12)), np.zeros((12, 13)), np.zeros(12), np.full((12, 12), np.nan)):
            with self.assertRaises(ValueError):
                compact.linear(backend, inputs, invalid, config=config)
        with self.assertRaises(ValueError):
            compact.linear(backend, inputs[:1], np.zeros((12, 12)), config=config)
        with self.assertRaises(ValueError):
            compact.linear(backend, inputs, np.zeros((12, 12)), np.zeros(11), config=config)
        with self.assertRaises(TypeError):
            compact.linear(backend, inputs, np.zeros((12, 12)), config=config, baby_step=True)
        with self.assertRaises(ValueError):
            compact.attention(backend, inputs, inputs[:1], inputs, config=config)
        with self.assertRaises(TypeError):
            compact.shift_tokens(backend, inputs, 0.5, config=config)
        with self.assertRaises(ValueError):
            compact.shift_tokens(backend, inputs[:1], 1, config=config)
        with self.assertRaises(ValueError):
            compact.pack_inputs(np.zeros((4, 11)), config, width=11)
        with self.assertRaises(ValueError):
            compact.unpack_outputs(np.zeros((2, 16)), config)
        bad = [dict(weights[0])]
        bad[0]["q_weight"] = np.zeros((12, 11))
        with self.assertRaises(ValueError):
            compact.build_model(backend, inputs, bad, config)
        bad[0].pop("q_weight")
        with self.assertRaises(ValueError):
            compact.build_model(backend, inputs, bad, config)

    def test_cyclic_shifts_use_only_whole_token_rotations(self):
        config = self.config
        backend = ArrayBackend(config)
        values = np.arange(config.tokens * config.hidden).reshape(config.tokens, config.hidden)
        inputs = self.encrypt(backend, values)
        for offset in range(-5, 6):
            with self.subTest(offset=offset):
                outputs = compact.shift_tokens(backend, inputs, offset, config=config)
                np.testing.assert_array_equal(self.decode(outputs), np.roll(values, -offset, axis=0))
        self.assertTrue(all(rotation % config.tile == 0 for rotation in backend.rotations))
        self.assertEqual(backend.plains, [])  # No gather mask or cross-CT input.

    def test_tiled_rectangular_bsgs_weight_indices_and_shared_babies(self):
        config = self.config
        for out_size, in_size in ((20, 12), (12, 20), (4, 4), (8, 16), (16, 8)):
            x = self.rng.normal(size=(config.tokens, in_size)) + np.arange(config.tokens)[:, None]
            # Unique nonsymmetric block/row/column values expose tile indexing.
            w = np.arange(out_size * in_size).reshape(out_size, in_size) / 103
            w += self.rng.normal(0, 0.1, size=w.shape)
            bias = self.rng.normal(size=out_size)
            for baby in (1, 2, 3, 16):
                with self.subTest(shape=w.shape, baby=baby):
                    backend = ArrayBackend(config)
                    inputs = self.encrypt(backend, x, width=in_size)
                    outputs = compact.linear(backend, inputs, w, bias, config=config, baby_step=baby)
                    np.testing.assert_allclose(self.decode(outputs, width=out_size), x @ w.T + bias,
                                               rtol=2e-13, atol=2e-13)
                    self.assertEqual(backend.products["ct"], 0)
                    nt_in, nt_out = in_size // config.tile, out_size // config.tile
                    diagonals = 2 * config.tile - 1
                    # No input mask: exactly one Plain/diagonal/block plus biases.
                    self.assertEqual(len(backend.plains), nt_in * nt_out * diagonals + nt_out)
                    baby = min(baby, diagonals)
                    giants = range((1 - config.tile) // baby, (config.tile - 1) // baby + 1)
                    expected_rotations = nt_in * (baby - 1) + nt_out * sum(g != 0 for g in giants)
                    self.assertEqual(len(backend.rotations), expected_rotations)

    def test_full_rectangular_hidden_ffn_hidden_pipeline(self):
        config = self.config
        backend = ArrayBackend(config)
        x = self.rng.normal(size=(config.tokens, config.hidden))
        up_w = self.rng.normal(size=(config.intermediate, config.hidden))
        down_w = self.rng.normal(size=(config.hidden, config.intermediate))
        up_b, down_b = self.rng.normal(size=config.intermediate), self.rng.normal(size=config.hidden)
        up = compact.linear(backend, self.encrypt(backend, x), up_w, up_b, config=config)
        self.assertEqual(len(up), 5)
        down = compact.linear(backend, up, down_w, down_b, config=config)
        self.assertEqual(len(down), 3)
        np.testing.assert_allclose(self.decode(down), (x @ up_w.T + up_b) @ down_w.T + down_b,
                                   rtol=2e-13, atol=2e-13)

    def test_attention_dense_oracle_and_no_redundant_dot_masks(self):
        # Two heads in each CT tests within-tile head isolation as well.
        config = replace(self.config, heads=6)
        q, k = self.rng.normal(0, 0.2, size=(2, config.tokens, config.hidden))
        v = self.rng.normal(size=(config.tokens, config.hidden))
        backend = ArrayBackend(config)
        batches = [self.encrypt(backend, a) for a in (q, k, v)]
        outputs = compact.attention(backend, *batches, config=config)
        np.testing.assert_allclose(self.decode(outputs, config), dense_attention(q, k, v, config),
                                   rtol=2e-13, atol=2e-13)
        self.assertEqual(backend.labels, [])
        # Test the dot primitive alone: precisely one head-start CP product,
        # no preliminary full-slot mask and no trailing mask after broadcast.
        backend = ArrayBackend(config)
        arithmetic = compact._Arithmetic(backend, config)
        queries, keys = self.encrypt(backend, q), self.encrypt(backend, k)
        dots = [arithmetic.head_dot(a, b) for a, b in zip(queries, keys)]
        expected = (q.reshape(4, 6, 2) * k.reshape(4, 6, 2)).sum(axis=-1) / math.sqrt(2)
        expected = np.repeat(expected, 2, axis=-1)
        np.testing.assert_allclose(self.decode(dots, config), expected, rtol=2e-13, atol=2e-13)
        self.assertEqual(backend.products, {"ct": 3, "plain": 3})
        self.assertEqual(len(backend.plains), 1)  # Shared head-start mask only.

    def test_attention_noncausal_all_keys_and_head_tile_isolation(self):
        config = replace(self.config, heads=6)
        q = self.rng.normal(0, 0.2, size=(config.tokens, config.hidden))
        k = self.rng.normal(0, 0.2, size=q.shape)
        baseline = np.zeros_like(q)
        # Each head in turn, including heads next to tile/token boundaries.
        for head in range(config.heads):
            v = baseline.copy()
            start = head * config.head_dim
            v[-1, start:start + config.head_dim] = [2, 5]
            backend = ArrayBackend(config)
            outputs = compact.attention(backend, *[self.encrypt(backend, a) for a in (q, k, v)], config=config)
            result = self.decode(outputs, config)
            expected = dense_attention(q, k, v, config)
            np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
            self.assertTrue(np.all(result[0, start:start + config.head_dim] > 0))  # Future token.
            other = np.ones(config.hidden, dtype=bool)
            other[start:start + config.head_dim] = False
            np.testing.assert_array_equal(result[:, other], np.zeros_like(result[:, other]))
        # Uniform scores prove that every cyclic key, not just the future key,
        # contributes, while query/feature tiles stay distinct.
        v = np.arange(config.tokens * config.hidden).reshape(config.tokens, config.hidden) / 10
        backend = ArrayBackend(config)
        zero = np.zeros_like(q)
        outputs = compact.attention(backend, *[self.encrypt(backend, a) for a in (zero, zero, v)], config=config)
        np.testing.assert_allclose(self.decode(outputs, config), np.tile(v.mean(axis=0), (4, 1)), atol=1e-14)

    def test_layer_norm_shares_true_full_hidden_mean_and_one_variance_ct(self):
        config = self.config
        x = self.rng.normal(size=(config.tokens, config.hidden))
        x += np.repeat([-1.0, 0.0, 1.0], config.tile)[None, :]  # Different tile means.
        x += np.arange(config.tokens)[:, None] * 2  # Different token means.
        gamma = self.rng.normal(1, 0.1, size=config.hidden)
        beta = self.rng.normal(0, 0.1, size=config.hidden)
        backend = ArrayBackend(config)
        arithmetic = compact._Arithmetic(backend, config, "stage")
        result = arithmetic.layer_norm(self.encrypt(backend, x), gamma, beta, "layer0.attention_ln")
        np.testing.assert_allclose(self.decode(result), dense_ln(x, gamma, beta, config),
                                   rtol=5e-13, atol=5e-13)
        self.assertEqual(backend.labels, ["layer0.attention_ln.variance.group0"] +
                         [f"layer0.attention_ln.output.group{i}" for i in range(3)])
        # Every LN reduction mask is the same row-start mask, never a tile's
        # separate statistic or an input/output full-slot cleanup mask.
        sparse = [p for p in backend.plains if np.any(p == 0)]
        self.assertEqual(len(sparse), 1)
        np.testing.assert_array_equal(sparse[0].reshape(4, 4),
                                      np.tile([1 / config.hidden, 0, 0, 0], (4, 1)))

    def test_weight_factory_preserves_schema_seed_and_independent_layers(self):
        config = replace(self.config, layers=2)
        weights = compact.create_weights(config, seed=20261114)
        original = model.create_weights(config, seed=20261114)
        for index in range(2):
            self.assertEqual(weights[index].keys(), original[index].keys())
            for key in weights[index]:
                np.testing.assert_array_equal(weights[index][key], original[index][key])
                self.assertEqual(weights[index][key].dtype, np.float32)
                self.assertFalse(np.shares_memory(weights[0][key], weights[1][key]))
                self.assertFalse(np.array_equal(weights[0][key], weights[1][key]))
        self.assertFalse(np.shares_memory(weights[0]["q_weight"], weights[0]["k_weight"]))

    def test_full_model_dense_polynomials_independent_layers_and_stage_counts(self):
        config = replace(self.config, layers=2, bsgs_baby_step=3, ln_variance_center=1.2)
        x = self.rng.normal(size=(config.tokens, config.hidden))
        x -= x.mean(axis=-1, keepdims=True)
        x *= math.sqrt(config.ln_variance_center) / np.sqrt((x * x).mean(axis=-1, keepdims=True))
        x += np.arange(config.tokens)[:, None] / 3  # RMSNorm must fail this oracle.
        weights = compact.create_weights(config, seed=19)
        expected = dense_model(x, weights, config)
        for policy in ("none", "stage"):
            backend = ArrayBackend(config)
            if policy == "none":
                backend.bootstrap = None  # A real no-bootstrap backend is valid.
            outputs = compact.build_model(backend, self.encrypt(backend, x), weights, config, boot_policy=policy)
            np.testing.assert_allclose(self.decode(outputs, config), expected, rtol=5e-12, atol=5e-12)
            self.assertEqual(len(outputs), 3)
            expected_count = 45 * config.layers if policy == "stage" else 0
            self.assertEqual(len(backend.labels), expected_count)
            self.assertEqual(len(set(backend.labels)), expected_count)
            if policy == "stage":
                self.assertEqual(sum(".variance." in label for label in backend.labels), 2 * config.layers)
                self.assertEqual(sum(".ffn_up." in label for label in backend.labels), 5 * config.layers)
                self.assertEqual(sum(".gelu." in label for label in backend.labels), 5 * config.layers)
                self.assertFalse(any("token" in label for label in backend.labels))

    def test_compact_linear_hook_all_layers_matches_fallback_and_dense(self):
        config = replace(self.config, layers=2)
        x = self.rng.normal(0, 0.7, size=(config.tokens, config.hidden))
        weights = compact.create_weights(config, seed=123)
        actual = []
        for backend_type in (ArrayBackend, HookBackend):
            backend = backend_type(config)
            result = compact.build_model(backend, self.encrypt(backend, x), weights, config)
            actual.append(self.decode(result, config))
            self.assertEqual(len(backend.labels), 90)
            if isinstance(backend, HookBackend):
                expected_labels = [f"layer{i}.{name}_weight" for i in range(2)
                                   for name in ("q", "k", "v", "o", "ffn_up", "ffn_down")]
                self.assertEqual([call[0] for call in backend.linear_calls], expected_labels)
                for label, weight, bias, received_config in backend.linear_calls:
                    layer, name = label.split(".")
                    index = int(layer[5:])
                    self.assertIs(weight, weights[index][name])
                    self.assertIs(bias, weights[index][name.removesuffix("_weight") + "_bias"])
                    self.assertIs(received_config, config)
        np.testing.assert_allclose(actual[0], actual[1], rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(actual[1], dense_model(x, weights, config), rtol=5e-12, atol=5e-12)
        backend = HookBackend(config)
        inputs = self.encrypt(backend, x)
        w = weights[0]["ffn_up_weight"]
        output = compact.linear(backend, inputs, w, config=config, label="custom.ffn_up_weight")
        np.testing.assert_allclose(self.decode(output, width=config.intermediate), x @ w.T,
                                   rtol=2e-13, atol=2e-13)
        self.assertEqual(backend.linear_calls[0][0], "custom.ffn_up_weight")
        self.assertEqual(backend.plains, [])  # Hook bypasses fallback entirely.
        backend.compact_linear = lambda *args, **kwargs: inputs[:1]
        with self.assertRaises(ValueError):
            compact.linear(backend, inputs, w, config=config)

    def test_shared_graph_backend_tiny_only(self):
        if __package__:
            from .backend import Graph
        else:
            from backend import Graph
        config = self.config
        graph = Graph(config.slots)
        inputs = [graph.input(f"tile{i}") for i in range(config.hidden_tiles)]
        weights = compact.create_weights(config, seed=29)
        x = self.rng.normal(0, 0.5, size=(config.tokens, config.hidden))
        physical = compact.pack_inputs(x, config)
        outputs = compact.build_model(graph, inputs, weights, config)
        values = graph.evaluate(outputs, {expr.data: row for expr, row in zip(inputs, physical)})
        result = compact.unpack_outputs(np.stack(values), config)
        np.testing.assert_allclose(result, dense_model(x, weights, config), rtol=5e-12, atol=5e-12)
        self.assertEqual(sum(node.op == "boot" for node in graph.reachable(outputs)), 45)


class CyclicAttentionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(391)
        self.config = compact.Config(tokens=4, hidden=8, heads=4, tile=8,
                                     slots=32, intermediate=24, weight_std=0.035)

    def configs(self):
        # T=16 additionally exercises nonzero giant rotations in both QK and
        # AV; T=4 alone would have gs=1 and miss giant compensation defects.
        return (self.config,
                compact.Config(tokens=16, hidden=64, heads=8, tile=32,
                               slots=512, intermediate=96, weight_std=0.025))

    @staticmethod
    def head_pairs(config):
        return [(4 * tile + a, 4 * tile + b) for tile in range(config.hidden_tiles)
                for a, b in ((0, 2), (1, 3))]

    @staticmethod
    def encrypt(backend, dense):
        return backend.encrypt(compact.pack_inputs(dense, backend.config))

    @staticmethod
    def decode(expressions, config):
        return compact.unpack_outputs(np.stack([expr.values for expr in expressions]), config)

    def test_cyclic_shape_constraints_and_pair_head_order(self):
        with self.assertRaises(ValueError):
            replace(self.config, heads=2)  # T != 2*head_dim.
        with self.assertRaises(ValueError):
            replace(self.config, tokens=8, slots=64)  # tile != 2*T.
        self.assertEqual(self.config.attention_groups, 2)
        self.assertEqual(self.config.stage_boots_per_layer, 30)
        for config in self.configs():
            values = np.arange(config.tokens * config.hidden).reshape(config.tokens, config.hidden) + 1
            backend = ArrayBackend(config)
            arithmetic = compact._Arithmetic(backend, config)
            expanded = arithmetic.expand_heads(self.encrypt(backend, values))
            self.assertEqual(len(expanded), config.heads // 2)
            for expr, pair in zip(expanded, self.head_pairs(config)):
                physical = expr.values.reshape(config.tokens, 2, config.tokens)
                expected = np.zeros_like(physical)
                for lane, head in enumerate(pair):
                    expected[:, lane, :config.head_dim] = values[:, head * config.head_dim:(head + 1) * config.head_dim]
                np.testing.assert_array_equal(physical, expected)
            np.testing.assert_array_equal(self.decode(arithmetic.merge_heads(expanded), config), values)
            np.testing.assert_array_equal(backend.plains[0].reshape(config.tokens, 2, config.tokens),
                                          np.tile(np.r_[np.ones(config.head_dim), np.zeros(config.head_dim)],
                                                  (config.tokens, 2, 1)))

    def test_qk_full_cyclic_score_indices_against_independent_numpy(self):
        for config in self.configs():
            q, k = self.rng.normal(0, 0.2, size=(2, config.tokens, config.hidden))
            backend = ArrayBackend(config)
            arithmetic = compact._Arithmetic(backend, config)
            queries, keys = [arithmetic.expand_heads(self.encrypt(backend, dense)) for dense in (q, k)]
            backend.products = {"ct": 0, "plain": 0}
            scores = [arithmetic.cyclic_qk(query, key) for query, key in zip(queries, keys)]
            self.assertEqual(len(scores), config.heads // 2)  # NOT T*hidden_tiles CTs.
            q_heads = q.reshape(config.tokens, config.heads, config.head_dim)
            k_heads = k.reshape(config.tokens, config.heads, config.head_dim)
            for expr, pair in zip(scores, self.head_pairs(config)):
                expected = np.empty((config.tokens, 2, config.tokens))
                for query in range(config.tokens):
                    cyclic_keys = (query + np.arange(config.tokens)) % config.tokens
                    for lane, head in enumerate(pair):
                        expected[query, lane] = (k_heads[cyclic_keys, head] @ q_heads[query, head]
                                                 / math.sqrt(config.head_dim))
                np.testing.assert_allclose(expr.values.reshape(config.tokens, 2, config.tokens),
                                           expected, rtol=2e-13, atol=2e-13)
            expected_products = config.tokens * config.attention_groups
            self.assertEqual(backend.products, {"ct": expected_products, "plain": expected_products})
            self.assertEqual(len(backend.plains), 2)  # split mask, scaled QK start mask.

    def test_av_bsgs_arbitrary_full_scores_and_zero_padding(self):
        for config in self.configs():
            values = self.rng.normal(size=(config.tokens, config.hidden))
            heads = values.reshape(config.tokens, config.heads, config.head_dim)
            # Arbitrary full cyclic A isolates AV from QK, mean subtraction,
            # exp, and reciprocal: no shared production helper in the oracle.
            scores = self.rng.uniform(0.5, 1.5, size=(config.attention_groups, config.tokens, 2, config.tokens))
            backend = ArrayBackend(config)
            arithmetic = compact._Arithmetic(backend, config)
            gapped = arithmetic.expand_heads(self.encrypt(backend, values))
            backend.products = {"ct": 0, "plain": 0}
            score_exprs = backend.encrypt(scores.reshape(config.attention_groups, config.slots))
            products = [arithmetic.cyclic_av(score, value) for score, value in zip(score_exprs, gapped)]
            for group, (expr, pair) in enumerate(zip(products, self.head_pairs(config))):
                expected = np.zeros((config.tokens, 2, config.tokens))
                for query in range(config.tokens):
                    cyclic_keys = (query + np.arange(config.tokens)) % config.tokens
                    for lane, head in enumerate(pair):
                        expected[query, lane, :config.head_dim] = scores[group, query, lane] @ heads[cyclic_keys, head]
                physical = expr.values.reshape(config.tokens, 2, config.tokens)
                np.testing.assert_allclose(physical, expected, rtol=2e-13, atol=2e-13)
                np.testing.assert_array_equal(physical[:, :, config.head_dim:], 0)
            # T cc products per pair, not T*log(head_dim) broadcast products.
            self.assertEqual(backend.products["ct"], config.tokens * config.attention_groups)

    def test_cyclic_attention_matches_offset_dense_and_future_tokens(self):
        for config in self.configs():
            q, k = self.rng.normal(0, 0.2, size=(2, config.tokens, config.hidden))
            v = self.rng.normal(size=(config.tokens, config.hidden))
            expected = dense_attention(q, k, v, config)
            for layout in ("cyclic", "offset"):
                current = replace(config, attention_layout=layout)
                for policy in ("none", "stage"):
                    backend = ArrayBackend(current)
                    if policy == "none":
                        backend.bootstrap = None
                    result = compact.attention(backend, *[self.encrypt(backend, a) for a in (q, k, v)],
                                               config=current, boot_policy=policy)
                    np.testing.assert_allclose(self.decode(result, current), expected, rtol=2e-13, atol=2e-13)
                    stages = 6 * current.attention_groups if layout == "cyclic" else 4 * current.hidden_tiles
                    self.assertEqual(len(backend.labels), stages if policy == "stage" else 0)
                    if policy == "stage" and layout == "cyclic":
                        self.assertEqual(sum(".qk." in label for label in backend.labels), current.heads // 2)
                        self.assertEqual(sum(".exp." in label for label in backend.labels), current.heads // 2)
        config = self.config
        zeros = np.zeros((config.tokens, config.hidden))
        for head in range(config.heads):
            values = zeros.copy()
            start = head * config.head_dim
            values[-1, start:start + config.head_dim] = [3, 7]
            backend = ArrayBackend(config)
            outputs = compact.attention(backend, *[self.encrypt(backend, a) for a in (zeros, zeros, values)], config=config)
            expected = np.tile(values[-1] / config.tokens, (config.tokens, 1))
            result = self.decode(outputs, config)
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)
            self.assertTrue(np.all(result[0, start:start + config.head_dim] > 0))

    def test_full_block_hook_fallback_offset_dense_and_stage_ct_counts(self):
        for base in self.configs():
            config = replace(base, layers=2)
            values = self.rng.normal(size=(config.tokens, config.hidden))
            values -= values.mean(axis=-1, keepdims=True)
            values /= np.sqrt((values * values).mean(axis=-1, keepdims=True))
            values += np.arange(config.tokens)[:, None] / (2 * config.tokens)
            weights = compact.create_weights(config, seed=20261114)
            expected = dense_model(values, weights, config)
            for layout, backend_type, policy in (("cyclic", ArrayBackend, "stage"),
                                                  ("cyclic", HookBackend, "none"),
                                                  ("offset", ArrayBackend, "none")):
                current = replace(config, attention_layout=layout)
                backend = backend_type(current)
                if policy == "none":
                    backend.bootstrap = None
                result = compact.build_model(backend, self.encrypt(backend, values), weights, current, boot_policy=policy)
                np.testing.assert_allclose(self.decode(result, current), expected, rtol=5e-12, atol=5e-12)
                if isinstance(backend, HookBackend):
                    self.assertEqual(len(backend.linear_calls), 6 * current.layers)
                if policy == "stage":
                    h, p, f = current.hidden_tiles, current.attention_groups, current.intermediate_tiles
                    stage_sizes = {"q": p, "k": p, "v": p, "attention.qk": p,
                                   "attention.exp": p, "attention.denominator": p,
                                   "attention.reciprocal": p, "attention.weighted_value": p,
                                   "attention.context": p, "attention_residual": h,
                                   "attention_ln.variance": 1, "attention_ln.output": h,
                                   "ffn_up": f, "gelu": f, "ffn_residual": h,
                                   "ffn_ln.variance": 1, "ffn_ln.output": h}
                    expected_labels = {f"layer{layer}.{stage}.group{group}"
                                       for layer in range(current.layers)
                                       for stage, size in stage_sizes.items() for group in range(size)}
                    self.assertEqual(set(backend.labels), expected_labels)
                    self.assertEqual(len(backend.labels), current.stage_boots_per_layer * current.layers)

    def test_tiny_cyclic_shared_graph_backend(self):
        if __package__:
            from .backend import Graph
        else:
            from backend import Graph
        config = self.config
        graph = Graph(config.slots)
        inputs = [graph.input(f"tile{i}") for i in range(config.hidden_tiles)]
        weights = compact.create_weights(config, seed=29)
        dense = self.rng.normal(0, 0.7, size=(config.tokens, config.hidden))
        physical = compact.pack_inputs(dense, config)
        outputs = compact.build_model(graph, inputs, weights, config)
        evaluated = graph.evaluate(outputs, {expr.data: row for expr, row in zip(inputs, physical)})
        result = compact.unpack_outputs(np.stack(evaluated), config)
        np.testing.assert_allclose(result, dense_model(dense, weights, config), rtol=5e-12, atol=5e-12)
        boots = [node for node in graph.reachable(outputs) if node.op == "boot"]
        self.assertEqual(len(boots), 30)


if __name__ == "__main__":
    unittest.main()
