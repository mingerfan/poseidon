"""Multi-token SIMD BERT encoder, independent of Hecate and execution policy.

Default layout: 128 tokens / 8 rows = 16 ciphertexts. Each 4096-slot row
contains one token: its first 768 slots hold hidden features and its first
3072 slots hold FFN features. Unused slots may be dirty. No operation creates
one ciphertext per token or head; attention uses 128 cyclic key offsets with
all eight queries and all twelve heads evaluated together.

Backend API is ``plain(np.ndarray)``, ``bootstrap(expr, label='')``, and
expressions supporting +, -, *, rotate(k). Positive rotations are LEFT:
rotate(k)[i] = x[(i+k) % slots]. All plaintexts have exactly ``slots`` entries.
Matrices are independent dense [out,in] arrays using model.py's weight schema.

This is post-LN, fully bidirectional BERT, without embeddings or a classifier.
GELU is degree four; softmax mean-centers scores, uses degree-four exp and
3/4 reciprocal Newton steps; LN subtracts the true mean and uses degree-three
Taylor rsqrt. These are workload polynomials, not pretrained-BERT accuracy
claims. Small weights and variance near ln_variance_center are recommended.

``stage`` refreshes 15 ciphertexts per *group*, per layer: Q/K/V; attention
exp denominator and weighted-V numerator; reciprocal; context; attention
residual; attention-LN variance/output; FFN up; GELU; FFN residual; FFN-LN
variance/output. In particular, it does NOT bootstrap each key offset. The
longest attention interval includes Q*K, quartic exp and exp*V (four dependent
ciphertext products), plus plaintext masks/scales. Backend scale/level and
residual alignment remain the runner's responsibility. ``none`` emits no boot.
"""

from dataclasses import dataclass
import math
import operator

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _array(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite real array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


@dataclass(frozen=True)
class PackedConfig:
    tokens: int = 128
    hidden: int = 768
    intermediate: int = 3072
    heads: int = 12
    slots: int = 32768
    row_stride: int = 4096
    layers: int = 1
    bsgs_baby_step: int | None = None
    reciprocal_iterations: int = 3
    ln_epsilon: float = 1e-5
    ln_variance_center: float = 1.0
    weight_std: float = 0.02

    def __post_init__(self):
        for name in ("tokens", "hidden", "intermediate", "heads", "slots",
                     "row_stride", "layers", "reciprocal_iterations"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("slots", "row_stride"):
            value = getattr(self, name)
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")
        if self.row_stride > self.slots:
            raise ValueError("row_stride must fit slots")
        if self.tokens % self.rows:
            raise ValueError("tokens must be an integer multiple of rows=slots//row_stride")
        if max(self.hidden, self.intermediate) > self.row_stride:
            raise ValueError("hidden and intermediate must fit a row")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.head_dim & (self.head_dim - 1):
            raise ValueError("head_dim=hidden//heads must be a power of two")
        if self.reciprocal_iterations not in (3, 4):
            raise ValueError("reciprocal_iterations must be 3 or 4")
        if self.bsgs_baby_step is not None:
            object.__setattr__(self, "bsgs_baby_step",
                               _integer(self.bsgs_baby_step, "bsgs_baby_step"))
        for name in ("ln_epsilon", "ln_variance_center", "weight_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def rows(self):
        return self.slots // self.row_stride

    @property
    def groups(self):
        return self.tokens // self.rows

    @property
    def head_dim(self):
        return self.hidden // self.heads


Config = PackedConfig


def create_weights(config, seed=0):
    """Use exactly the token-prefix model's independent dense weight factory."""
    try:
        from .model import create_weights as factory
    except ImportError:  # Support `python .../test_packed_model.py` as well.
        from model import create_weights as factory
    return factory(config, seed=seed)


def pack_inputs(values, config, *, tail_value=0.0):
    """Client-only packing of [tokens,hidden] into [groups,slots] NumPy arrays."""
    values = _array(values, "inputs", (config.tokens, config.hidden))
    tail_value = float(tail_value)
    if not math.isfinite(tail_value):
        raise ValueError("tail_value must be finite")
    result = np.full((config.groups, config.slots), tail_value, dtype=np.float64)
    result.reshape(config.groups, config.rows, config.row_stride)[:, :, :config.hidden] = (
        values.reshape(config.groups, config.rows, config.hidden))
    return result


def unpack_outputs(values, config, *, width=None):
    """Client-only decoding of already decrypted [groups,slots] arrays."""
    width = config.hidden if width is None else _integer(width, "width")
    if width > config.row_stride:
        raise ValueError("width must fit a row")
    values = _array(values, "outputs", (config.groups, config.slots))
    return values.reshape(config.groups, config.rows, config.row_stride)[:, :, :width].reshape(
        config.tokens, width).copy()


class _Arithmetic:
    def __init__(self, backend, config, boot_policy="none"):
        if boot_policy not in ("stage", "none"):
            raise ValueError("boot_policy must be 'stage' or 'none'")
        if not callable(getattr(backend, "plain", None)):
            raise TypeError("backend must provide plain(array)")
        if boot_policy == "stage" and not callable(getattr(backend, "bootstrap", None)):
            raise TypeError("stage policy requires backend.bootstrap(expr, label=...)")
        self.backend = backend
        self.config = config
        self.boot_policy = boot_policy
        self.constants = {}

    def row_plain(self, values):
        """Repeat one row's public prefix in EVERY physical row."""
        row = np.zeros(self.config.row_stride, dtype=np.float64)
        row[:len(values)] = values
        return self.backend.plain(np.tile(row, self.config.rows))

    def constant(self, width, value=1.0):
        key = ("prefix", width, float(value))
        if key not in self.constants:
            self.constants[key] = self.row_plain(np.full(width, value))
        return self.constants[key]

    def mask(self, expr, width):
        return expr * self.constant(width)

    def stage(self, expr, label, width=None):
        if self.boot_policy == "stage":
            expr = self.backend.bootstrap(expr, label=label)
            # A bootstrap is allowed to leave unused slots dirty as well.
            expr = self.mask(expr, self.config.hidden if width is None else width)
        return expr

    def row_total(self, expr, width, factor=1.0):
        """Reduce into row starts, then broadcast; never reduce a dirty tail."""
        total = self.mask(expr, width)
        shift = 1
        while shift < width:
            total = total + total.rotate(shift)
            shift *= 2
        total = total * self.constant(1)
        shift = 1
        while shift < width:
            total = total + total.rotate(-shift)
            shift *= 2
        return total * self.constant(width, factor)

    def linear(self, inputs, weight, bias=None, baby_step=None):
        """Rectangular BSGS, shared diagonals/baby rotations across token groups.

        d runs from -(out-1) through in-1, with d=g*B+b and 0<=b<B.
        The plaintext is pre-rotated by -g*B BEFORE the giant rotation:
        rot(g*B, rot(-g*B,D_d) * rot(b,x)) = D_d * rot(d,x).
        D_d has nonzeros only where both the output and input column fit the
        SAME row. Thus the global ciphertext rotations cannot leak other rows
        or dirty tails, even for rectangular and non-power-of-two dimensions.
        """
        out_size, in_size = weight.shape
        count = out_size + in_size - 1
        baby = min(baby_step or self.config.bsgs_baby_step or math.isqrt(count), count)
        clean = [self.mask(expr, in_size) for expr in inputs]
        babies = [[expr] + [expr.rotate(b) for b in range(1, baby)] for expr in clean]
        outputs = [None] * len(inputs)
        row_indices = np.arange(out_size)
        first, last = 1 - out_size, in_size - 1
        for giant in range(first // baby, last // baby + 1):
            giant_shift = giant * baby
            subtotals = [None] * len(inputs)
            for little in range(baby):
                diagonal_shift = giant_shift + little
                if not first <= diagonal_shift <= last:
                    continue
                columns = row_indices + diagonal_shift
                valid = (columns >= 0) & (columns < in_size)
                row = np.zeros(self.config.row_stride, dtype=np.float64)
                row[row_indices[valid]] = weight[row_indices[valid], columns[valid]]
                # np.roll(+g) is plaintext rotate(-g) under the left-rotate API.
                diagonal = np.roll(np.tile(row, self.config.rows), giant_shift)
                plain = self.backend.plain(diagonal)
                for group, rotations in enumerate(babies):
                    term = rotations[little] * plain
                    subtotals[group] = term if subtotals[group] is None else subtotals[group] + term
            for group, subtotal in enumerate(subtotals):
                if giant_shift:
                    subtotal = subtotal.rotate(giant_shift)
                outputs[group] = subtotal if outputs[group] is None else outputs[group] + subtotal
        if bias is not None:
            offset = self.row_plain(bias)
            outputs = [expr + offset for expr in outputs]
        return outputs

    def _row_selection(self, start, stop, width):
        key = ("rows", start, stop, width)
        if key not in self.constants:
            values = np.zeros((self.config.rows, self.config.row_stride), dtype=np.float64)
            values[start:stop, :width] = 1
            self.constants[key] = self.backend.plain(values.reshape(-1))
        return self.constants[key]

    def shift_clean(self, inputs, offset, width):
        """Row q receives token (q+offset)%tokens, with at most two sources."""
        offset %= self.config.tokens
        group_shift, row_shift = divmod(offset, self.config.rows)
        if row_shift == 0:
            return [inputs[(group + group_shift) % self.config.groups]
                    for group in range(self.config.groups)]
        cut = self.config.rows - row_shift
        low = self._row_selection(0, cut, width)
        high = self._row_selection(cut, self.config.rows, width)
        rotation = row_shift * self.config.row_stride
        return [(inputs[(group + group_shift) % self.config.groups].rotate(rotation) * low
                 + inputs[(group + group_shift + 1) % self.config.groups].rotate(rotation) * high)
                for group in range(self.config.groups)]

    def head_dot(self, query, key):
        config = self.config
        total = self.mask(query * key, config.hidden)
        shift = 1
        while shift < config.head_dim:
            total = total + total.rotate(shift)
            shift *= 2
        cache_key = ("head_starts",)
        if cache_key not in self.constants:
            row = np.zeros(config.row_stride, dtype=np.float64)
            row[:config.hidden:config.head_dim] = 1 / math.sqrt(config.head_dim)
            self.constants[cache_key] = self.backend.plain(np.tile(row, config.rows))
        total = total * self.constants[cache_key]
        shift = 1
        while shift < config.head_dim:
            total = total + total.rotate(-shift)
            shift *= 2
        return self.mask(total, config.hidden)

    def exponential(self, expr):
        width = self.config.hidden
        square = expr * expr
        cube = square * expr
        fourth = square * square
        return (expr + self.constant(width) + square * self.constant(width, 0.5)
                + cube * self.constant(width, 1 / 6) + fourth * self.constant(width, 1 / 24))

    def reciprocal(self, denominator):
        config = self.config
        # First Newton step from y0=1/tokens, folded to save a ct*ct product.
        inverse = (denominator * self.constant(config.hidden, -1 / config.tokens)
                   + self.constant(config.hidden, 2)) * self.constant(config.hidden, 1 / config.tokens)
        for _ in range(config.reciprocal_iterations - 1):
            correction = (denominator * inverse) * self.constant(config.hidden, -1) + self.constant(config.hidden, 2)
            inverse = inverse * correction
        return inverse

    def attention(self, queries, keys, values, label):
        config = self.config
        width = config.hidden
        queries, keys, values = ([self.mask(expr, width) for expr in batch]
                                 for batch in (queries, keys, values))
        # mean_j <Q_i,K_j> = <Q_i,mean_j K_j>. Reduce K first so
        # centering does not keep 128*groups score ciphertexts live until exp.
        # Row-stride rotations preserve feature lanes and sum all physical rows.
        key_total = keys[0]
        for key in keys[1:]:
            key_total = key_total + key
        shift = 1
        while shift < config.rows:
            key_total = key_total + key_total.rotate(shift * config.row_stride)
            shift *= 2
        mean_key = key_total * self.constant(width, 1 / config.tokens)
        means = [self.head_dot(query, mean_key) for query in queries]
        denominators, numerators = [None] * config.groups, [None] * config.groups
        for offset in range(config.tokens):
            shifted_keys = self.shift_clean(keys, offset, width)
            shifted = self.shift_clean(values, offset, width)
            for group in range(config.groups):
                score = self.head_dot(queries[group], shifted_keys[group])
                exp = self.exponential(score - means[group])
                weighted = exp * shifted[group]
                denominators[group] = exp if denominators[group] is None else denominators[group] + exp
                numerators[group] = weighted if numerators[group] is None else numerators[group] + weighted
        outputs = []
        for group in range(config.groups):
            suffix = f".group{group}"
            denominator = self.stage(denominators[group], label + ".denominator" + suffix)
            numerator = self.stage(numerators[group], label + ".weighted_value" + suffix)
            inverse = self.stage(self.reciprocal(denominator), label + ".reciprocal" + suffix)
            # One normalization AFTER accumulating all exp*V branches.
            outputs.append(self.stage(numerator * inverse, label + ".context" + suffix))
        return outputs

    def layer_norm(self, inputs, gamma, beta, label):
        config = self.config
        width, center = config.hidden, config.ln_variance_center
        scale = self.row_plain(np.asarray(gamma, dtype=np.float64) / math.sqrt(center))
        offset = self.row_plain(beta)
        outputs = []
        for group, expr in enumerate(inputs):
            clean = self.mask(expr, width)
            centered = clean - self.row_total(clean, width, 1 / width)
            variance = self.row_total(centered * centered, width, 1 / width) + self.constant(width, config.ln_epsilon)
            variance = self.stage(variance, f"{label}.variance.group{group}")
            u = variance * self.constant(width, 1 / center) - self.constant(width)
            square = u * u
            cube = square * u
            inverse = (u * self.constant(width, -0.5) + self.constant(width)
                       + square * self.constant(width, 3 / 8) - cube * self.constant(width, 5 / 16))
            result = centered * inverse * scale + offset
            outputs.append(self.stage(result, f"{label}.output.group{group}"))
        return outputs

    def gelu(self, expr):
        width = self.config.intermediate
        square = expr * expr
        fourth = square * square
        coefficient = 1 / math.sqrt(2 * math.pi)
        return (expr * self.constant(width, 0.5) + square * self.constant(width, coefficient)
                - fourth * self.constant(width, coefficient / 6))


def linear(backend, expr, weight, bias=None, *, config, baby_step=None):
    """Single rowpacked ciphertext rectangular [out,in] BSGS primitive."""
    weight = _array(weight, "weight")
    if weight.ndim != 2 or min(weight.shape) < 1 or max(weight.shape) > config.row_stride:
        raise ValueError("weight must be nonempty [out,in] fitting a row")
    if bias is not None:
        bias = _array(bias, "bias", (weight.shape[0],))
    if baby_step is not None:
        baby_step = _integer(baby_step, "baby_step")
    return _Arithmetic(backend, config).linear([expr], weight, bias, baby_step)[0]


def shift_tokens(backend, inputs, offset, *, config, width=None):
    """Public cyclic gather, including cross-ciphertext and wraparound shifts."""
    offset = _integer(offset, "offset", None)
    width = config.hidden if width is None else _integer(width, "width")
    if width > config.row_stride:
        raise ValueError("width must fit a row")
    inputs = list(inputs)
    if len(inputs) != config.groups:
        raise ValueError(f"expected {config.groups} input ciphertexts")
    arithmetic = _Arithmetic(backend, config)
    return arithmetic.shift_clean([arithmetic.mask(expr, width) for expr in inputs], offset, width)


def attention(backend, queries, keys, values, *, config, boot_policy="none"):
    """Full bidirectional attention on projected rowpacked Q/K/V (no linear)."""
    batches = [list(batch) for batch in (queries, keys, values)]
    if any(len(batch) != config.groups for batch in batches):
        raise ValueError(f"Q/K/V must each contain {config.groups} ciphertexts")
    return _Arithmetic(backend, config, boot_policy).attention(*batches, "attention")


def _weight_shapes(config):
    h, f = config.hidden, config.intermediate
    shapes = {}
    for name in ("q", "k", "v", "o"):
        shapes[name + "_weight"] = (h, h)
        shapes[name + "_bias"] = (h,)
    shapes.update(ffn_up_weight=(f, h), ffn_up_bias=(f,),
                  ffn_down_weight=(h, f), ffn_down_bias=(h,))
    for name in ("attention_ln", "ffn_ln"):
        shapes[name + "_gamma"] = (h,)
        shapes[name + "_beta"] = (h,)
    return shapes


def build_model(backend, inputs, weights, config, boot_policy="stage"):
    """Build configurable BERT blocks; return ``tokens//rows`` ciphertexts.

    The weight schema and mathematical polynomials match model.py; only the
    SIMD layout and bootstrap boundaries differ. Tail slots of both inputs
    and bootstrap results may be arbitrary. Weights/packing are public data.
    """
    inputs, weights = list(inputs), list(weights)
    if len(inputs) != config.groups:
        raise ValueError(f"expected {config.groups} input ciphertexts")
    if len(weights) != config.layers:
        raise ValueError(f"expected {config.layers} layer weight dictionaries")
    checked = []
    for index, layer in enumerate(weights):
        converted = {}
        for name, shape in _weight_shapes(config).items():
            if name not in layer:
                raise ValueError(f"layer {index} is missing {name}")
            converted[name] = _array(layer[name], f"layer {index}.{name}", shape)
        checked.append(converted)
    arithmetic = _Arithmetic(backend, config, boot_policy)
    hidden = [arithmetic.mask(expr, config.hidden) for expr in inputs]
    for index, layer in enumerate(checked):
        label = f"layer{index}"

        def project(batch, name):
            return arithmetic.linear(batch, layer[name + "_weight"], layer[name + "_bias"])

        projected = {}
        for name in ("q", "k", "v"):
            projected[name] = [arithmetic.stage(expr, f"{label}.{name}.group{group}")
                               for group, expr in enumerate(project(hidden, name))]
        contexts = arithmetic.attention(projected["q"], projected["k"], projected["v"], label + ".attention")
        residuals = [arithmetic.stage(expr + residual, f"{label}.attention_residual.group{group}")
                     for group, (expr, residual) in enumerate(zip(project(contexts, "o"), hidden))]
        attended = arithmetic.layer_norm(residuals, layer["attention_ln_gamma"], layer["attention_ln_beta"],
                                         label + ".attention_ln")
        up = [arithmetic.stage(expr, f"{label}.ffn_up.group{group}", config.intermediate)
              for group, expr in enumerate(project(attended, "ffn_up"))]
        activated = [arithmetic.stage(arithmetic.gelu(expr), f"{label}.gelu.group{group}", config.intermediate)
                     for group, expr in enumerate(up)]
        residuals = [arithmetic.stage(expr + residual, f"{label}.ffn_residual.group{group}")
                     for group, (expr, residual) in enumerate(zip(project(activated, "ffn_down"), attended))]
        hidden = arithmetic.layer_norm(residuals, layer["ffn_ln_gamma"], layer["ffn_ln_beta"], label + ".ffn_ln")
    return [arithmetic.mask(expr, config.hidden) for expr in hidden]
