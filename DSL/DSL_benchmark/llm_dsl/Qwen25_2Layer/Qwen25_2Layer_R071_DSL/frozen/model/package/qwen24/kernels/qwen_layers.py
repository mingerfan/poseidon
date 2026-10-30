"""Qwen layer reference DSL: prefix-packed vectors lowered to Hecate ops.

One token/vector per ciphertext. This is a correctness-oriented lowering,
not an optimized implementation of the full 0.5B model. Nonlinear approximation
profiles must be supplied explicitly; no implicit approximation or Boot.
"""

from dataclasses import asdict, dataclass
import math
import operator

import numpy as np

from . import ops
from . import qwen_nonlinear as nonlinear
from .expr import Expr, Plain

ChebyshevApproximation = nonlinear.ChebyshevApproximation
InverseSqrtApproximation = nonlinear.InverseSqrtApproximation
NormalizeSquareSoftmax = nonlinear.NormalizeSquareSoftmax


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    value = operator.index(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class PackedVector:
    """First size slots are meaningful; slots is the actual context capacity.

    The caller binds this metadata to a real ciphertext. Tail slots need not
    be zero: reductions, extraction, and concatenation mask them explicitly.
    """

    expr: Expr
    size: int
    slots: int

    def __post_init__(self):
        ops._ciphertext(self.expr, "expr")
        size = _integer(self.size, "size")
        slots = _integer(self.slots, "slots")
        if slots & (slots - 1) or size > slots:
            raise ValueError("slots must be a power of two and size must fit slots")


@dataclass(frozen=True)
class PolynomialApproximation:
    """Public approximation contract; encrypted inputs cannot be range-checked.

    domain describes where the caller has calibrated this polynomial. It is
    not a proof of accuracy or a runtime bound check. Coefficients use ascending
    powers and must be validated against the named function separately.
    """

    function: str
    coefficients: tuple
    domain: tuple

    def __post_init__(self):
        if self.function not in {"silu", "rsqrt", "exp", "reciprocal"}:
            raise ValueError(f"unsupported approximation function: {self.function!r}")
        coefficients = ops._real_array(self.coefficients, "coefficients")
        domain = ops._real_array(self.domain, "domain")
        if coefficients.ndim != 1 or len(coefficients) < 2 or coefficients[-1] == 0:
            raise ValueError("a nonconstant polynomial with nonzero leading coefficient is required")
        if domain.shape != (2,) or domain[0] >= domain[1]:
            raise ValueError("domain must be a finite increasing [lower, upper] pair")
        if self.function in {"rsqrt", "reciprocal"} and domain[0] <= 0:
            raise ValueError("rsqrt and softmax reciprocal domains must be positive")
        object.__setattr__(self, "coefficients", tuple(coefficients))
        object.__setattr__(self, "domain", tuple(domain))


def _matching(x, y):
    if (x.size, x.slots) != (y.size, y.slots):
        raise ValueError("vectors must have identical prefix sizes and slot capacities")


def add(x, y):
    _matching(x, y)
    return PackedVector(ops.add(x.expr, y.expr), x.size, x.slots)


def mul(x, y):
    _matching(x, y)
    return PackedVector(ops.mul(x.expr, y.expr), x.size, x.slots)


def _plain(x, values, operation):
    values = ops._real_array(values, "plaintext")
    if values.shape != (x.size,):
        raise ValueError(f"plaintext must have shape {(x.size,)}")
    return PackedVector(operation(x.expr, Plain(values)), x.size, x.slots)


def _scale(x, factor):
    return _plain(x, np.full(x.size, factor), ops.mul)


def extract(x, start, size):
    """Copy a public slice into the prefix; masks all unused slots."""
    start = _integer(start, "start", 0)
    size = _integer(size, "size")
    if start + size > x.size:
        raise ValueError("slice exceeds vector size")
    shifted = x.expr.rotate(start) if start else x.expr
    return PackedVector(ops.mul(shifted, Plain(np.ones(size))), size, x.slots)


def concatenate(vectors):
    vectors = tuple(vectors)
    if not vectors:
        raise ValueError("cannot concatenate an empty sequence")
    slots = vectors[0].slots
    size = sum(x.size for x in vectors)
    if size > slots or any(x.slots != slots for x in vectors):
        raise ValueError("concatenation must fit one shared slot capacity")
    return _concatenate_clean(tuple(extract(x, 0, x.size) for x in vectors))


def _concatenate_clean(vectors):
    """Internal only: inputs must already have zero tails from a mask."""
    slots = vectors[0].slots
    size = sum(x.size for x in vectors)
    offset = 0
    for x in vectors:
        term = x.expr
        if offset:
            term = term.rotate(-offset)
        result = term if offset == 0 else ops.add(result, term)
        offset += x.size
    return PackedVector(result, size, slots)


def _sum(x, factor=1.0):
    result = extract(x, 0, x.size).expr
    shift = 1
    while shift < x.size:
        result = ops.add(result, result.rotate(shift))
        shift *= 2
    return PackedVector(ops.mul(result, Plain([factor])), 1, x.slots)


def _total_repeated(x):
    """Sum the meaningful prefix into EVERY physical slot, with one mask.

    Full-capacity cyclic reduction avoids scalar extraction and rebroadcast.
    The extra rotations trade runtime for fewer rescale levels.
    """
    result = extract(x, 0, x.size).expr
    shift = 1
    while shift < x.slots:
        result = ops.add(result, result.rotate(shift))
        shift *= 2
    return result


def _broadcast(x, size):
    size = _integer(size, "size")
    if x.size != 1 or size > x.slots:
        raise ValueError("broadcast requires a scalar and a size within slot capacity")
    result = extract(x, 0, 1).expr
    shift = 1
    while shift < size:
        result = ops.add(result, result.rotate(-shift))
        shift *= 2
    return PackedVector(ops.mul(result, Plain(np.ones(size))), size, x.slots)


def linear(x, weight, bias=None):
    """W @ x + b, W=[out,in]. Supports any positive dimensions fitting slots.

    Rectangular diagonals lower to rotate/multiply/add. Uses in+out-1 terms;
    deliberately simple and not an optimized large-model matrix multiply.
    """
    weight = ops._real_array(weight, "weight")
    if (weight.ndim != 2 or weight.shape[1] != x.size or
            not 0 < weight.shape[0] <= x.slots):
        raise ValueError("weight must have shape [out<=slots, x.size] with out>0")
    out_size = weight.shape[0]
    if bias is not None:
        bias = ops._real_array(bias, "bias")
        if bias.shape != (out_size,):
            raise ValueError("bias must have shape [out]")
    rows = np.arange(out_size)
    for shift in range(1 - out_size, x.size):
        columns = rows + shift
        valid = (columns >= 0) & (columns < x.size)
        diagonal = np.zeros(out_size)
        diagonal[valid] = weight[rows[valid], columns[valid]]
        source = x.expr.rotate(shift) if shift else x.expr
        term = ops.mul(source, Plain(diagonal))
        result = term if shift == 1 - out_size else ops.add(result, term)
    output = PackedVector(result, out_size, x.slots)
    return output if bias is None else _plain(output, bias, ops.add)


def lm_head(x, weight, *, chunk_size):
    """Large-vocabulary linear projection returned as ordered ciphertext chunks.

    Concatenate decrypted prefixes in order to recover logits. No argmax,
    sampling, or full-vocabulary softmax is performed.
    """
    chunk_size = _integer(chunk_size, "chunk_size")
    weight = np.asarray(weight)
    if weight.ndim != 2 or weight.shape[1] != x.size or weight.shape[0] == 0:
        raise ValueError("lm_head weight must have shape [vocabulary, hidden]")
    if chunk_size > x.slots:
        raise ValueError("chunk_size exceeds slot capacity")
    return tuple(linear(x, weight[start:start + chunk_size])
                 for start in range(0, weight.shape[0], chunk_size))


def client_embedding(token_ids, weight):
    """Plaintext CLIENT operation before encryption, not encrypted-ID lookup."""
    token_ids = np.asarray(token_ids)
    weight = np.asarray(weight)
    if token_ids.ndim != 1 or token_ids.dtype.kind not in "iu" or token_ids.size == 0:
        raise ValueError("token_ids must be a nonempty 1D integer array on the client")
    if weight.ndim != 2 or weight.shape[1] == 0:
        raise ValueError("embedding weight must have shape [vocabulary, hidden]")
    if np.any(token_ids < 0) or np.any(token_ids >= weight.shape[0]):
        raise ValueError("token id outside embedding vocabulary")
    return ops._real_array(weight[token_ids], "selected embedding rows")


def approximate(x, approximation, *, function):
    if approximation.function != function:
        raise ValueError(f"expected {function} approximation, got {approximation.function}")
    constant = lambda value: Plain(np.full(x.slots, value))
    if isinstance(approximation, ChebyshevApproximation):
        result = nonlinear.chebyshev(extract(x, 0, x.size).expr, approximation, constant)
    elif isinstance(approximation, InverseSqrtApproximation):
        result = nonlinear.inverse_sqrt(extract(x, 0, x.size).expr, approximation, constant)
    elif isinstance(approximation, PolynomialApproximation):
        result = ops.polynomial(x.expr, approximation.coefficients, slots=x.size)
    else:
        raise TypeError("unsupported nonlinear approximation profile")
    return PackedVector(result, x.size, x.slots)


def approximation_from_dict(data):
    """Legacy power profiles omit method; new methods must be explicit."""
    fields = dict(data)
    method = fields.pop("method", "power")
    classes = {"power": PolynomialApproximation, "chebyshev": ChebyshevApproximation,
               "newton_rsqrt": InverseSqrtApproximation, "normalize_square": NormalizeSquareSoftmax}
    if method not in classes:
        raise ValueError(f"unknown nonlinear method: {method}")
    return classes[method](**fields)


def approximation_to_dict(profile):
    methods = {PolynomialApproximation: "power", ChebyshevApproximation: "chebyshev",
               InverseSqrtApproximation: "newton_rsqrt", NormalizeSquareSoftmax: "normalize_square"}
    if type(profile) not in methods:
        raise TypeError("unsupported nonlinear approximation profile")
    return {"method": methods[type(profile)], **asdict(profile)}


def rms_norm(x, weight, *, rsqrt, eps=1e-6):
    """x * approximate_rsqrt(mean(x*x) + eps) * weight, over one token."""
    weight = ops._real_array(weight, "RMSNorm weight")
    if weight.shape != (x.size,) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm requires weight [hidden] and finite eps>0")
    constant = lambda value: Plain(np.full(x.slots, value))
    denominator = _total_repeated(mul(x, x)) * constant(1.0 / x.size) + constant(eps)
    if isinstance(rsqrt, InverseSqrtApproximation):
        inverse = nonlinear.inverse_sqrt(denominator, rsqrt, constant)
    else:
        inverse = approximate(PackedVector(denominator, x.slots, x.slots), rsqrt,
                              function="rsqrt").expr
    return _plain(PackedVector(x.expr * inverse, x.size, x.slots), weight, ops.mul)


def rope(x, position, *, head_dim=64, theta=1_000_000.0):
    """Qwen split-half RoPE; heads are contiguous, position/theta are public."""
    position = _integer(position, "position", 0)
    head_dim = _integer(head_dim, "head_dim")
    if head_dim % 2 or x.size % head_dim or not math.isfinite(theta) or theta <= 0:
        raise ValueError("RoPE requires even head_dim dividing size and finite theta>0")
    half = head_dim // 2
    angle = position / (theta ** (np.arange(0, head_dim, 2) / head_dim))
    angles = np.tile(np.concatenate((angle, angle)), x.size // head_dim)
    first = np.arange(x.size) % head_dim < half
    left = ops.mul(x.expr.rotate(half), Plain(-first.astype(np.float64)))
    right = ops.mul(x.expr.rotate(-half), Plain((~first).astype(np.float64)))
    rotated = PackedVector(ops.add(left, right), x.size, x.slots)
    return add(_plain(x, np.cos(angles), ops.mul), _plain(rotated, np.sin(angles), ops.mul))


def softmax(x, *, exp=None, reciprocal=None, public_shift=0.0, normalization=None):
    """Approximate softmax with one explicitly selected arithmetic method.

    The legacy method uses two polynomials and a PUBLIC constant shift.
    normalization instead selects mean centering and normalize-and-square.
    No encrypted max, comparison, or -infinity mask. Domains are caller
    contracts, not runtime checks or guarantees about CKKS noise.
    """
    if normalization is not None:
        if not isinstance(normalization, NormalizeSquareSoftmax):
            raise TypeError("normalization must be NormalizeSquareSoftmax")
        if exp is not None or reciprocal is not None or public_shift != 0:
            raise ValueError("normalize-square cannot be combined with exp/reciprocal/public_shift")
        if x.size > normalization.max_length:
            raise ValueError("attention row exceeds calibrated max_length")
        if x.size == 1:
            return _plain(_scale(x, 0), [1.0], ops.add)
        constant = lambda value: Plain(np.full(x.slots, value))
        # Mask before the polynomial: unused input slots have no domain bound.
        centered = x.expr + _total_repeated(x) * constant(-1.0 / x.size)
        centered = extract(PackedVector(centered, x.size, x.slots), 0, x.size).expr
        result = nonlinear.normalize_square(
            centered, normalization, x.size, constant,
            lambda expr: _total_repeated(PackedVector(expr, x.size, x.slots)),
            lambda expr: expr,
            scalar_constant=constant,
        )
        return extract(PackedVector(result, x.size, x.slots), 0, x.size)
    if exp is None or reciprocal is None:
        raise ValueError("softmax requires exp/reciprocal or a normalize-square profile")
    if not math.isfinite(public_shift):
        raise ValueError("public_shift must be finite")
    shifted = _plain(x, np.full(x.size, -public_shift), ops.add) if public_shift else x
    numerator = approximate(shifted, exp, function="exp")
    inverse = approximate(_sum(numerator), reciprocal, function="reciprocal")
    return mul(numerator, _broadcast(inverse, x.size))


def _positions(values, count, name):
    values = tuple(_integer(value, name, 0) for value in values)
    if len(values) != count or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must match token count and be strictly increasing")
    return values


def gqa_attention(queries, keys, values, *, query_positions, key_positions,
                  exp=None, reciprocal=None, query_heads=14, kv_heads=2, head_dim=64,
                  public_shift=0.0, normalization=None):
    """Causal GQA core, after Q/K RoPE; returns one packed vector per query.

    Each key/value ciphertext holds all KV heads for one token. Each query
    holds all Q heads. Already rotated cached keys may be supplied explicitly;
    cache allocation/update is outside this layer. Positions are public and
    static for a trace. Future keys are excluded before softmax, never -inf.
    """
    query_heads = _integer(query_heads, "query_heads")
    kv_heads = _integer(kv_heads, "kv_heads")
    head_dim = _integer(head_dim, "head_dim")
    queries, keys, values = tuple(queries), tuple(keys), tuple(values)
    if not queries or not keys or len(keys) != len(values) or query_heads % kv_heads:
        raise ValueError("GQA requires nonempty Q/K/V, matching K/V counts and Q heads divisible by KV heads")
    slots = queries[0].slots
    if any(x.slots != slots or x.size != query_heads * head_dim for x in queries):
        raise ValueError("query shape/capacity mismatch")
    if any(x.slots != slots or x.size != kv_heads * head_dim for x in keys + values):
        raise ValueError("key/value shape/capacity mismatch")
    if len(keys) > slots:
        raise ValueError("attention score vector exceeds slot capacity")
    query_positions = _positions(query_positions, len(queries), "query_positions")
    key_positions = _positions(key_positions, len(keys), "key_positions")
    if normalization is None and (exp is None or reciprocal is None or
                                  exp.function != "exp" or reciprocal.function != "reciprocal"):
        raise ValueError("GQA requires exp and reciprocal approximations")
    if normalization is not None and (not isinstance(normalization, NormalizeSquareSoftmax) or
                                      exp is not None or reciprocal is not None or public_shift != 0):
        raise ValueError("GQA requires one unambiguous softmax method")
    if normalization is not None and any(sum(k <= q for k in key_positions) > normalization.max_length
                                          for q in query_positions):
        raise ValueError("attention row exceeds calibrated max_length")
    outputs = []
    for query, position in zip(queries, query_positions):
        visible = [i for i, key_position in enumerate(key_positions) if key_position <= position]
        if not visible:
            raise ValueError("each query requires at least one visible key")
        head_outputs = []
        for head in range(query_heads):
            kv_head = head // (query_heads // kv_heads)
            q = extract(query, head * head_dim, head_dim)
            selected_keys = [extract(keys[i], kv_head * head_dim, head_dim) for i in visible]
            selected_values = [extract(values[i], kv_head * head_dim, head_dim) for i in visible]
            if len(visible) == 1:
                # Softmax of a single score is exactly one; avoids unnecessary
                # approximation at the first causal token / single-key decode.
                head_outputs.append(selected_values[0])
                continue
            scores = _concatenate_clean([_sum(mul(q, k), 1.0 / math.sqrt(head_dim))
                                         for k in selected_keys])
            probabilities = softmax(scores, exp=exp, reciprocal=reciprocal,
                                    public_shift=public_shift, normalization=normalization)
            for i, value in enumerate(selected_values):
                probability = _broadcast(extract(probabilities, i, 1), head_dim)
                term = mul(probability, value)
                result = term if i == 0 else add(result, term)
            head_outputs.append(result)
        outputs.append(_concatenate_clean(head_outputs))
    return tuple(outputs)


def attention(hidden, q_weight, k_weight, v_weight, o_weight, *, positions,
              q_bias=None, k_bias=None, v_bias=None, cached_positions=(),
              cached_keys=(), cached_values=(), query_heads=14, kv_heads=2,
              head_dim=64, theta=1_000_000.0, exp=None, reciprocal=None,
              public_shift=0.0, normalization=None):
    """Full causal attention: Q/K/V projections, RoPE, GQA, and O projection.

    Returns outputs plus immutable accumulated K/V. Cached ciphertexts are
    consumed directly; no decrypt/re-encrypt or mutation of the old cache.
    RMSNorm and residual connections belong to the surrounding decoder block.
    """
    hidden = tuple(hidden)
    positions = _positions(positions, len(hidden), "positions")
    cached_keys, cached_values = tuple(cached_keys), tuple(cached_values)
    cached_positions = _positions(cached_positions, len(cached_keys), "cached_positions")
    if not hidden or len(cached_values) != len(cached_keys):
        raise ValueError("attention requires hidden inputs and matching cached K/V")
    if cached_positions and cached_positions[-1] >= positions[0]:
        raise ValueError("new attention positions must follow the cache")
    queries, keys, values = [], [], []
    for x, position in zip(hidden, positions):
        queries.append(rope(linear(x, q_weight, q_bias), position, head_dim=head_dim, theta=theta))
        keys.append(rope(linear(x, k_weight, k_bias), position, head_dim=head_dim, theta=theta))
        values.append(linear(x, v_weight, v_bias))
    keys, values = cached_keys + tuple(keys), cached_values + tuple(values)
    attended = gqa_attention(queries, keys, values, query_positions=positions,
                             key_positions=cached_positions + positions,
                             query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
                             exp=exp, reciprocal=reciprocal, public_shift=public_shift,
                             normalization=normalization)
    return tuple(linear(x, o_weight) for x in attended), keys, values


def swiglu(x, gate_weight, up_weight, down_weight, *, silu):
    """W_down @ (polynomial_silu(W_gate @ x) * (W_up @ x)); no biases."""
    gate_weight = ops._real_array(gate_weight, "gate_weight")
    up_weight = ops._real_array(up_weight, "up_weight")
    down_weight = ops._real_array(down_weight, "down_weight")
    if (gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape or
            gate_weight.shape[1] != x.size or gate_weight.shape[0] < 1 or
            gate_weight.shape[0] > x.slots or
            down_weight.shape != (x.size, gate_weight.shape[0])):
        raise ValueError("SwiGLU requires gate/up [ffn,hidden], down [hidden,ffn], ffn<=slots")
    if silu.function != "silu":
        raise ValueError("SwiGLU requires a SiLU approximation")
    gate = approximate(linear(x, gate_weight), silu, function="silu")
    up = linear(x, up_weight)
    return linear(mul(gate, up), down_weight)
