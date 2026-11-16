"""Backend-independent, prefix-packed BERT-base encoder arithmetic.

Inputs are already embedded vectors: one token per ciphertext, with ``hidden``
meaningful prefix slots. Outputs have the same layout. This is the encoder stack
only (no embedding lookup, pooler, classifier, dropout, RoPE, GQA or SwiGLU).
Every layer has dense Q/K/V/O and FFN weights, bidirectional attention, and two
*post-residual* LayerNorms with actual mean subtraction.

Backend contract
----------------
``backend.plain(values)`` accepts a real, one-dimensional NumPy array of exactly
``config.slots`` elements. Expressions support ``+``, ``-``, ``*`` with expressions
or Plain operands, and ``rotate(k)``. Positive rotation means left rotation:
``rotate(k)[i] == original[(i + k) % slots]``. No scalar overloads are required.
``backend.bootstrap(expr, label=...)`` is an explicit callback returning an
expression. ``boot_policy='none'`` never calls it; ``'stage'`` uses the named
boundaries below, not scalar-level automatic refreshing. A backend must manage
its own scale/level alignment, including residual branches retained over boots.

Approximation contract
----------------------
These deliberately low-degree approximations are workload definitions, NOT
claims of pretrained-BERT accuracy or runtime encrypted range checks:

* GELU: x/2 + x^2/sqrt(2*pi) - x^4/(6*sqrt(2*pi)), degree 4 about zero
  (a useful local calibration interval is [-1, 1]).
* Softmax: subtract the encrypted per-head score mean, use degree-4 Taylor exp,
  then three (or four) reciprocal Newton steps from 1/tokens. All keys are used;
  there is no causal mask. Convergence requires 0 < sum(exp4)/tokens < 2;
  centered scores near [-0.5, 0.5] are an intended calibration regime.
* LayerNorm: subtract the encrypted feature mean, then use degree-3 Taylor
  rsqrt around ``ln_variance_center`` on variance + epsilon. The expansion
  variable is u = (variance + epsilon)/center - 1; |u| < 1 is a local regime.

The stage schedule, per layer and token, is Q/K/V; packed attention scores;
packed exp; packed reciprocal; attention context; attention residual;
attention-LN variance; attention-LN output; FFN up; GELU; FFN residual;
FFN-LN variance; FFN-LN output. Thus there are 15*tokens boots per layer.
Softmax heads/keys share a packed ciphertext: no dot-product scalar is booted.
The largest stage interval has four dependent ciphertext products (three-step
reciprocal), or six for four-step reciprocal. Plain-mask multiplications add
backend-dependent rescale depth and must not be ignored in CKKS budgeting.

Weights are a list of layer dictionaries. Matrices use [output, input] order;
keys are q/k/v/o_weight and _bias, ffn_up/down_weight and _bias, and
attention_ln/ffn_ln_gamma and _beta. ``create_weights`` makes independent dense
float32 arrays, not tied, diagonal, low-rank or pretrained weights.
"""

from dataclasses import dataclass
import math
import operator

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _power_of_two_ceiling(value):
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class BertConfig:
    """BERT-base defaults; smaller dimensions/slot counts support CPU tests.

    ``hidden == heads * head_dim`` is required. Attention rows are packed at
    power-of-two token strides, so heads*next_power_of_two(tokens) must fit.
    Linear output chunks default to next_power_of_two(input width); they are
    internal tiles, not extra ciphertext outputs. ``linear_chunk_size`` can
    reduce that tile size; baby rotations remain shared across all tiles.
    """

    tokens: int = 2
    layers: int = 12
    hidden: int = 768
    intermediate: int = 3072
    heads: int = 12
    head_dim: int = 64
    slots: int = 32768
    bsgs_baby_step: int | None = None
    linear_chunk_size: int | None = None
    reciprocal_iterations: int = 3
    ln_epsilon: float = 1e-5
    ln_variance_center: float = 1.0
    weight_std: float = 0.02

    def __post_init__(self):
        for name in ("tokens", "layers", "hidden", "intermediate", "heads",
                     "head_dim", "slots", "reciprocal_iterations"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.slots & (self.slots - 1):
            raise ValueError("slots must be a power of two")
        if self.hidden != self.heads * self.head_dim:
            raise ValueError("hidden must equal heads * head_dim")
        if max(self.hidden, self.intermediate) > self.slots:
            raise ValueError("hidden and intermediate must fit one ciphertext")
        if self.heads * _power_of_two_ceiling(self.tokens) > self.slots:
            raise ValueError("packed head/token score rows do not fit slots")
        if self.reciprocal_iterations not in (3, 4):
            raise ValueError("reciprocal_iterations must be 3 or 4")
        for name in ("bsgs_baby_step", "linear_chunk_size"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, name))
        for name in ("ln_epsilon", "ln_variance_center", "weight_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


# A short spelling is convenient for runners constructing configurations.
Config = BertConfig


def _weight_shapes(config):
    h, f = config.hidden, config.intermediate
    shapes = {}
    for name in ("q", "k", "v", "o"):
        shapes[f"{name}_weight"] = (h, h)
        shapes[f"{name}_bias"] = (h,)
    shapes.update(ffn_up_weight=(f, h), ffn_up_bias=(f,),
                  ffn_down_weight=(h, f), ffn_down_bias=(h,))
    for name in ("attention_ln", "ffn_ln"):
        shapes[f"{name}_gamma"] = (h,)
        shapes[f"{name}_beta"] = (h,)
    return shapes


def create_weights(config, seed=0):
    """Return deterministic, independent dense float32 weights for every layer.

    Default BERT-base weights occupy roughly 340 MB; this function neither
    downloads data nor writes files. LayerNorm gamma is sampled around one;
    every other array is sampled around zero with ``config.weight_std``.
    """
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(config.layers):
        layer = {}
        for name, shape in _weight_shapes(config).items():
            values = rng.standard_normal(shape, dtype=np.float32)
            values *= np.float32(config.weight_std)
            if name.endswith("_gamma"):
                values += np.float32(1)
            layer[name] = values
        layers.append(layer)
    return layers


def _real_array(values, name, shape=None):
    result = np.asarray(values)
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real array")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


class _Arithmetic:
    """Slot arithmetic with cached public masks, but no frontend dependency."""

    def __init__(self, backend, slots, boot_policy="none"):
        if boot_policy not in ("stage", "none"):
            raise ValueError("boot_policy must be 'stage' or 'none'")
        if not callable(getattr(backend, "plain", None)):
            raise TypeError("backend must provide plain(array)")
        if boot_policy == "stage" and not callable(getattr(backend, "bootstrap", None)):
            raise TypeError("stage policy requires backend.bootstrap(expr, label=...)")
        self.backend = backend
        self.slots = slots
        self.boot_policy = boot_policy
        self._constants = {}

    def constant(self, size, value=1.0):
        key = (size, float(value))
        if key not in self._constants:
            values = np.zeros(self.slots, dtype=np.float64)
            values[:size] = value
            self._constants[key] = self.backend.plain(values)
        return self._constants[key]

    def group_constant(self, groups, width, stride, value=1.0):
        key = (groups, width, stride, float(value))
        if key not in self._constants:
            values = np.zeros(self.slots, dtype=np.float64)
            for head in range(groups):
                values[head * stride:head * stride + width] = value
            self._constants[key] = self.backend.plain(values)
        return self._constants[key]

    def mask(self, expr, size):
        return expr * self.constant(size)

    def extract(self, expr, start, size):
        shifted = expr.rotate(start) if start else expr
        return self.mask(shifted, size)

    def sum_slice(self, expr, start, size):
        """Reduce only a masked slice into slot zero; ignore all dirty tails."""
        total = self.extract(expr, start, size)
        shift = 1
        while shift < size:
            total = total + total.rotate(shift)
            shift *= 2
        return self.mask(total, 1)

    def broadcast(self, scalar, size):
        result = self.mask(scalar, 1)
        shift = 1
        while shift < size:
            result = result + result.rotate(-shift)
            shift *= 2
        return self.mask(result, size)

    def group_total(self, expr, groups, width, factor=1.0):
        """Sum each padded row and broadcast within its meaningful prefix.

        Row stride is next_power_of_two(width). Masks before reduction, at row
        starts, and after broadcast prevent both tail and adjacent-head leaks.
        """
        stride = _power_of_two_ceiling(width)
        total = expr * self.group_constant(groups, width, stride)
        shift = 1
        while shift < stride:
            total = total + total.rotate(shift)
            shift *= 2
        total = total * self.group_constant(groups, 1, stride)
        shift = 1
        while shift < stride:
            total = total + total.rotate(-shift)
            shift *= 2
        return total * self.group_constant(groups, width, stride, factor)

    def stage(self, expr, label):
        if self.boot_policy == "stage":
            return self.backend.bootstrap(expr, label=label)
        return expr

    def linear(self, expr, weight, bias=None, baby_step=None, chunk_size=None):
        """Rectangular cyclic-diagonal BSGS; all chunks share baby rotations.

        Pad the input width to p=next_power_of_two(input). One duplicated block
        supplies modulo-p reads, without assuming p divides the *true* input
        width. For diagonal k=g+b, pre-rotate its plaintext by -g, multiply the
        shared rotate(b) input, then rotate each giant subtotal by g. Output
        tiles have <=p rows; concatenate them into ONE output ciphertext.
        Plaintexts are streamed one diagonal at a time (no dense slot tensor).
        """
        out_size, in_size = weight.shape
        padded = _power_of_two_ceiling(in_size)
        baby = min(baby_step or math.isqrt(padded), padded)
        chunk = min(chunk_size or padded, padded)
        clean = self.mask(expr, in_size)
        # At most two blocks are read: row + diagonal < 2*p. If p==slots,
        # ordinary physical cyclic rotation already provides the wraparound.
        source = clean if padded == self.slots else clean + clean.rotate(-padded)
        babies = [source]
        babies.extend(source.rotate(shift) for shift in range(1, baby))
        result = None
        for start in range(0, out_size, chunk):
            count = min(chunk, out_size - start)
            rows = np.arange(count)
            tile = None
            for giant in range(0, padded, baby):
                subtotal = None
                # rotate(-giant) moves diagonal row r to r+giant.
                positions = (rows + giant) % self.slots
                for little in range(min(baby, padded - giant)):
                    columns = (rows + giant + little) % padded
                    valid = columns < in_size
                    diagonal = np.zeros(self.slots, dtype=np.float64)
                    diagonal[positions[valid]] = weight[start + rows[valid], columns[valid]]
                    term = babies[little] * self.backend.plain(diagonal)
                    subtotal = term if subtotal is None else subtotal + term
                if giant:
                    subtotal = subtotal.rotate(giant)
                tile = subtotal if tile is None else tile + subtotal
            if start:
                tile = tile.rotate(-start)
            result = tile if result is None else result + tile
        if bias is not None:
            values = np.zeros(self.slots, dtype=np.float64)
            values[:out_size] = bias
            result = result + self.backend.plain(values)
        return result


def linear(backend, expr, weight, bias=None, *, slots=32768,
           baby_step=None, chunk_size=None):
    """Public rectangular BSGS primitive, useful for isolated backend checks.

    ``weight`` is [out,in], both dimensions <= slots, with arbitrary (including
    non-power-of-two) dimensions. The input tail may be dirty; the output tail
    is zero. ``baby_step`` rotations are computed once, not once per output
    chunk. With p padded input slots and c output chunks, rotation count is
    <= 1+(baby-1)+c*(ceil(p/baby)-1)+(c-1).
    """
    slots = _integer(slots, "slots")
    if slots & (slots - 1):
        raise ValueError("slots must be a power of two")
    weight = _real_array(weight, "weight")
    if weight.ndim != 2 or min(weight.shape) < 1 or max(weight.shape) > slots:
        raise ValueError("weight must be nonempty [out,in] fitting slots")
    if bias is not None:
        bias = _real_array(bias, "bias", (weight.shape[0],))
    if baby_step is not None:
        baby_step = _integer(baby_step, "baby_step")
    if chunk_size is not None:
        chunk_size = _integer(chunk_size, "chunk_size")
    return _Arithmetic(backend, slots).linear(expr, weight, bias, baby_step, chunk_size)


def _gelu(arithmetic, expr, size):
    square = expr * expr
    fourth = square * square
    coefficient = 1 / math.sqrt(2 * math.pi)
    return (expr * arithmetic.constant(size, 0.5)
            + square * arithmetic.constant(size, coefficient)
            - fourth * arithmetic.constant(size, coefficient / 6))


def _softmax(arithmetic, scores, config, label):
    heads, tokens = config.heads, config.tokens
    stride = _power_of_two_ceiling(tokens)
    constant = lambda value: arithmetic.group_constant(heads, tokens, stride, value)
    scores = scores * constant(1)
    centered = scores - arithmetic.group_total(scores, heads, tokens, 1 / tokens)
    square = centered * centered
    cube = square * centered
    fourth = square * square
    exponential = (centered + constant(1) + square * constant(0.5)
                   + cube * constant(1 / 6) + fourth * constant(1 / 24))
    exponential = arithmetic.stage(exponential * constant(1), label + ".exp")
    denominator = arithmetic.group_total(exponential, heads, tokens)
    # y0=1/tokens; this is the first Newton step without a ct*ct product.
    inverse = (denominator * constant(-1 / tokens) + constant(2)) * constant(1 / tokens)
    for _ in range(config.reciprocal_iterations - 1):
        correction = (denominator * inverse) * constant(-1) + constant(2)
        inverse = inverse * correction
    inverse = arithmetic.stage(inverse, label + ".reciprocal")
    return exponential * inverse * constant(1)


def _layer_norm(arithmetic, expr, gamma, beta, config, label):
    size = config.hidden
    clean = arithmetic.mask(expr, size)
    mean = arithmetic.group_total(clean, 1, size, 1 / size)
    centered = clean - mean
    variance = arithmetic.group_total(centered * centered, 1, size, 1 / size)
    variance = variance + arithmetic.constant(size, config.ln_epsilon)
    variance = arithmetic.stage(variance, label + ".variance")
    center = config.ln_variance_center
    u = variance * arithmetic.constant(size, 1 / center) - arithmetic.constant(size)
    square = u * u
    cube = square * u
    inverse = (u * arithmetic.constant(size, -0.5) + arithmetic.constant(size)
               + square * arithmetic.constant(size, 3 / 8)
               - cube * arithmetic.constant(size, 5 / 16))
    # Merge the public rsqrt center factor with learned gamma.
    scale = np.zeros(config.slots, dtype=np.float64)
    scale[:size] = np.asarray(gamma, dtype=np.float64) / math.sqrt(center)
    offset = np.zeros(config.slots, dtype=np.float64)
    offset[:size] = beta
    result = centered * inverse * arithmetic.backend.plain(scale) + arithmetic.backend.plain(offset)
    return arithmetic.stage(result, label + ".output")


def _attention(arithmetic, queries, keys, values, config, layer_label):
    outputs = []
    stride = _power_of_two_ceiling(config.tokens)
    scale = arithmetic.constant(1, 1 / math.sqrt(config.head_dim))
    for token, query in enumerate(queries):
        scores = None
        # Pack ALL keys for ALL heads into a single ciphertext for this query.
        for key_index, key in enumerate(keys):
            products = query * key
            for head in range(config.heads):
                score = arithmetic.sum_slice(products, head * config.head_dim, config.head_dim) * scale
                destination = head * stride + key_index
                if destination:
                    score = score.rotate(-destination)
                scores = score if scores is None else scores + score
        label = f"{layer_label}.attention.token{token}"
        scores = arithmetic.stage(scores, label + ".scores")
        probabilities = _softmax(arithmetic, scores, config, label + ".softmax")
        context = None
        for key_index, value in enumerate(values):
            head_weights = None
            for head in range(config.heads):
                scalar = arithmetic.extract(probabilities, head * stride + key_index, 1)
                repeated = arithmetic.broadcast(scalar, config.head_dim)
                if head:
                    repeated = repeated.rotate(-head * config.head_dim)
                head_weights = repeated if head_weights is None else head_weights + repeated
            term = head_weights * value
            context = term if context is None else context + term
        outputs.append(arithmetic.stage(context, label + ".context"))
    return outputs


def build_model(backend, inputs, weights, config, boot_policy="stage"):
    """Build the encoder and return one hidden-prefix ciphertext per token.

    ``inputs`` has exactly ``config.tokens`` ciphertexts; their unused slots
    may be arbitrary. ``weights`` follows the ``create_weights`` dictionary
    schema and has exactly ``config.layers`` independent layer entries. Public
    array shapes/finiteness are checked; encrypted approximation domains are
    deliberately not checked. No Hecate, CUDA or other model package is used.
    """
    inputs = list(inputs)
    weights = list(weights)
    if len(inputs) != config.tokens:
        raise ValueError(f"expected {config.tokens} input ciphertexts")
    if len(weights) != config.layers:
        raise ValueError(f"expected {config.layers} layer weight dictionaries")
    checked = []
    for index, layer in enumerate(weights):
        converted = {}
        for name, shape in _weight_shapes(config).items():
            if name not in layer:
                raise ValueError(f"layer {index} is missing {name}")
            converted[name] = _real_array(layer[name], f"layer {index}.{name}", shape)
        checked.append(converted)
    arithmetic = _Arithmetic(backend, config.slots, boot_policy)
    hidden = [arithmetic.mask(expr, config.hidden) for expr in inputs]
    for index, layer in enumerate(checked):
        label = f"layer{index}"

        def project(expr, name):
            return arithmetic.linear(expr, layer[name + "_weight"], layer[name + "_bias"],
                                     config.bsgs_baby_step, config.linear_chunk_size)

        projected = {}
        for name in ("q", "k", "v"):
            projected[name] = [arithmetic.stage(project(expr, name), f"{label}.{name}.token{token}")
                               for token, expr in enumerate(hidden)]
        contexts = _attention(arithmetic, projected["q"], projected["k"], projected["v"], config, label)
        attended = []
        for token, (expr, context) in enumerate(zip(hidden, contexts)):
            residual = arithmetic.stage(project(context, "o") + expr,
                                        f"{label}.attention_residual.token{token}")
            attended.append(_layer_norm(arithmetic, residual,
                                        layer["attention_ln_gamma"], layer["attention_ln_beta"],
                                        config, f"{label}.attention_ln.token{token}"))
        hidden = []
        for token, expr in enumerate(attended):
            up = arithmetic.stage(project(expr, "ffn_up"), f"{label}.ffn_up.token{token}")
            activated = arithmetic.stage(_gelu(arithmetic, up, config.intermediate),
                                         f"{label}.gelu.token{token}")
            residual = arithmetic.stage(project(activated, "ffn_down") + expr,
                                        f"{label}.ffn_residual.token{token}")
            hidden.append(_layer_norm(arithmetic, residual, layer["ffn_ln_gamma"], layer["ffn_ln_beta"],
                                      config, f"{label}.ffn_ln.token{token}"))
    # Final masks also make the interface robust to nonzero bootstrap tails.
    return [arithmetic.mask(expr, config.hidden) for expr in hidden]
