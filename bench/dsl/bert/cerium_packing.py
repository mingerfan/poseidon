"""Compact feature-tiled BERT arithmetic, independent of HE execution policy.

BERT-base uses three hidden ciphertexts, each a [128 tokens, 256 features]
array, and twelve FFN ciphertexts. All 32768 slots are semantic (CKKS N=65536);
there are no input-prefix/tail masks. Weights, seed and polynomial mathematics
are unchanged from model.py: dense [out,in] matrices, bidirectional exp4
attention with encrypted mean subtraction and three reciprocal Newton steps,
GELU4, and true full-hidden LayerNorm with rsqrt3.

Backend contract: plain(full_slot_numpy_array), Expr + / - / *, and left
rotate(k)[i] = x[(i+k) % slots]. Plain objects may be opaque; scalar arithmetic
is not required. Optionally compact_linear(inputs, weight, bias, config,
label) returns output ciphertexts using a prepared/fused implementation. EVERY
linear dispatches through that hook when present; otherwise rectangular tiled
BSGS uses all 2*tile-1 diagonals and shared baby rotations per input tile.

Default cyclic attention follows Cerium's QK/AV BSGS: split each compact CT
into two gapped head pairs (head order 0,2,1,3,...), storing ALL cyclic key
scores in six CTs rather than 384 repeated-score CTs. It requires tile=2*tokens
and tokens=2*head_dim. attention_layout='offset' retains the original direct
cyclic-offset implementation and supports the more general compact layout.

stage callbacks preserve dotted layer/stage/group labels without prescribing
physical levels. Per-layer callbacks are 92 for default cyclic BERT-base, or
59 with offset attention, including only one variance CT per LayerNorm. Cyclic
Q/K/V and each attention stage have six CTs; QK and exp are shared frontiers,
not per-offset refreshes. The backend owns scale/level alignment and native
Boot scheduling. none requires no bootstrap method.
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
class CeriumConfig:
    tokens: int = 128
    hidden: int = 768
    heads: int = 12
    tile: int = 256
    slots: int = 32768
    intermediate: int = 3072
    layers: int = 1
    bsgs_baby_step: int = 16
    reciprocal_iterations: int = 3
    ln_epsilon: float = 1e-5
    ln_variance_center: float = 1.0
    weight_std: float = 0.02
    attention_layout: str = "cyclic"

    def __post_init__(self):
        for name in ("tokens", "hidden", "heads", "tile", "slots",
                     "intermediate", "layers", "bsgs_baby_step",
                     "reciprocal_iterations"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.slots & (self.slots - 1):
            raise ValueError("slots must be a power of two")
        if self.tokens * self.tile != self.slots:
            raise ValueError("tokens*tile must equal slots; no unused slots")
        if self.hidden % self.tile or self.intermediate % self.tile:
            raise ValueError("hidden and intermediate must be divisible by tile")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if self.tile % self.head_dim:
            raise ValueError("head_dim must divide tile; heads cannot cross tiles")
        if self.attention_layout not in ("cyclic", "offset"):
            raise ValueError("attention_layout must be 'cyclic' or 'offset'")
        if self.attention_layout == "cyclic":
            if self.tile != 2 * self.tokens or self.tokens != 2 * self.head_dim:
                raise ValueError("cyclic attention requires tile=2*tokens and tokens=2*head_dim")
        if self.reciprocal_iterations != 3:
            raise ValueError("this workload uses exactly three reciprocal Newton steps")
        for name in ("ln_epsilon", "ln_variance_center", "weight_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def head_dim(self):
        return self.hidden // self.heads

    @property
    def hidden_tiles(self):
        return self.hidden // self.tile

    @property
    def intermediate_tiles(self):
        return self.intermediate // self.tile

    @property
    def heads_per_tile(self):
        return self.tile // self.head_dim

    @property
    def attention_groups(self):
        return self.heads // 2 if self.attention_layout == "cyclic" else self.hidden_tiles

    @property
    def stage_boots_per_layer(self):
        if self.attention_layout == "cyclic":
            return (9 * self.attention_groups + 4 * self.hidden_tiles
                    + 2 * self.intermediate_tiles + 2)
        return 11 * self.hidden_tiles + 2 * self.intermediate_tiles + 2


Config = CeriumConfig


def create_weights(config, seed=0):
    """Reuse exactly model.py's independent float32 dense weight schema/RNG."""
    try:
        from .model import create_weights as factory
    except ImportError:
        from model import create_weights as factory
    return factory(config, seed=seed)


def _width(config, width):
    width = config.hidden if width is None else _integer(width, "width")
    if width % config.tile:
        raise ValueError("width must be divisible by tile")
    return width


def pack_inputs(values, config, *, width=None):
    """Client-only [tokens,width] -> [width//tile,slots], with no padding."""
    width = _width(config, width)
    values = _array(values, "inputs", (config.tokens, width))
    return values.reshape(config.tokens, width // config.tile, config.tile).transpose(
        1, 0, 2).reshape(width // config.tile, config.slots).astype(np.float64, copy=True)


def unpack_outputs(values, config, *, width=None):
    """Client-only decoding of decrypted [width//tile,slots] NumPy arrays."""
    width = _width(config, width)
    values = _array(values, "outputs", (width // config.tile, config.slots))
    return values.reshape(width // config.tile, config.tokens, config.tile).transpose(
        1, 0, 2).reshape(config.tokens, width).copy()


class _Arithmetic:
    def __init__(self, backend, config, boot_policy="none"):
        if boot_policy not in ("stage", "none"):
            raise ValueError("boot_policy must be 'stage' or 'none'")
        if not callable(getattr(backend, "plain", None)):
            raise TypeError("backend must provide plain(array)")
        if boot_policy == "stage" and not callable(getattr(backend, "bootstrap", None)):
            raise TypeError("stage policy requires backend.bootstrap(expr, label=...)")
        self.backend, self.config, self.boot_policy = backend, config, boot_policy
        self.constants = {}

    def row_plain(self, values):
        """Repeat a complete feature tile in every physical token row."""
        return self.backend.plain(np.tile(np.asarray(values, dtype=np.float64), self.config.tokens))

    def constant(self, value=1.0):
        key = ("constant", float(value))
        if key not in self.constants:
            self.constants[key] = self.backend.plain(np.full(self.config.slots, value, dtype=np.float64))
        return self.constants[key]

    def stage(self, expr, label):
        if self.boot_policy == "stage":
            return self.backend.bootstrap(expr, label=label)
        return expr

    def linear(self, inputs, weight, bias, label, baby_step=None):
        hook = getattr(self.backend, "compact_linear", None)
        if callable(hook):
            outputs = list(hook(inputs, weight, bias, config=self.config, label=label))
            if len(outputs) != weight.shape[0] // self.config.tile:
                raise ValueError("compact_linear returned the wrong number of output tiles")
            return outputs
        return self.linear_fallback(inputs, weight, bias, baby_step)

    def linear_fallback(self, inputs, weight, bias, baby_step=None):
        """Rectangular [out,in] made of full tile x tile BSGS blocks.

        d=-(tile-1)..tile-1, d=g*B+b, 0<=b<B. For each block D_d,
        rot(g*B, rot(-g*B,D_d)*rot(b,x)) = D_d*rot(d,x).
        Invalid diagonal lanes are zero, so global rotations never import a
        different token's features. The SOURCE is fully semantic: no input
        mask. All output tiles share each input tile's baby rotations, and
        input-block subtotals are combined before each giant rotation.
        """
        config = self.config
        tile = config.tile
        out_tiles, in_tiles = weight.shape[0] // tile, weight.shape[1] // tile
        baby = min(baby_step or config.bsgs_baby_step, 2 * tile - 1)
        babies = [[expr] + [expr.rotate(b) for b in range(1, baby)] for expr in inputs]
        lanes = np.arange(tile)
        outputs = []
        first, last = 1 - tile, tile - 1
        for out_tile in range(out_tiles):
            output = None
            for giant in range(first // baby, last // baby + 1):
                giant_shift = giant * baby
                subtotal = None
                for in_tile in range(in_tiles):
                    for little in range(baby):
                        diagonal_shift = giant_shift + little
                        if not first <= diagonal_shift <= last:
                            continue
                        columns = lanes + diagonal_shift
                        valid = (columns >= 0) & (columns < tile)
                        row = np.zeros(tile, dtype=np.float64)
                        row[valid] = weight[out_tile * tile + lanes[valid],
                                            in_tile * tile + columns[valid]]
                        # np.roll(+g) is plaintext rotate(-g) for LEFT rotation.
                        diagonal = self.backend.plain(np.roll(np.tile(row, config.tokens), giant_shift))
                        term = babies[in_tile][little] * diagonal
                        subtotal = term if subtotal is None else subtotal + term
                if giant_shift:
                    subtotal = subtotal.rotate(giant_shift)
                output = subtotal if output is None else output + subtotal
            if bias is not None:
                output = output + self.row_plain(bias[out_tile * tile:(out_tile + 1) * tile])
            outputs.append(output)
        return outputs

    def row_total(self, expr, factor):
        """Reduce a full feature tile at row starts, then broadcast to lanes."""
        shift = 1
        while shift < self.config.tile:
            expr = expr + expr.rotate(shift)
            shift *= 2
        key = ("row_starts", float(factor))
        if key not in self.constants:
            row = np.zeros(self.config.tile, dtype=np.float64)
            row[0] = factor
            self.constants[key] = self.row_plain(row)
        expr = expr * self.constants[key]
        shift = 1
        while shift < self.config.tile:
            expr = expr + expr.rotate(-shift)
            shift *= 2
        return expr

    def head_dot(self, query, key):
        total = query * key
        shift = 1
        while shift < self.config.head_dim:
            total = total + total.rotate(shift)
            shift *= 2
        cache_key = ("head_starts",)
        if cache_key not in self.constants:
            row = np.zeros(self.config.tile, dtype=np.float64)
            row[::self.config.head_dim] = 1 / math.sqrt(self.config.head_dim)
            self.constants[cache_key] = self.row_plain(row)
        # The ONLY dot-product mask; no redundant full-slot input/output mask.
        total = total * self.constants[cache_key]
        shift = 1
        while shift < self.config.head_dim:
            total = total + total.rotate(-shift)
            shift *= 2
        return total

    def exponential(self, expr):
        square = expr * expr
        cube = square * expr
        fourth = square * square
        return (expr + self.constant() + square * self.constant(0.5)
                + cube * self.constant(1 / 6) + fourth * self.constant(1 / 24))

    def reciprocal(self, denominator, *, zero_head_gaps=False):
        # Default syntax is preserved for byte-identical historical lowering.
        if zero_head_gaps:
            config = self.config
            if (config.attention_layout != 'cyclic' or config.tile != 2 * config.tokens or
                    config.tokens != 2 * config.head_dim):
                raise ValueError('masked reciprocal requires the cyclic two-head gap layout')
            # Folded FIRST Newton step with public vector coefficients, not an
            # extra encrypted mask. Subsequent updates still number iterations-1.
            mask = np.zeros(config.tile, dtype=np.float64)
            mask[:config.head_dim] = 1
            mask[config.tokens:config.tokens + config.head_dim] = 1
            inverse = (denominator * self.row_plain(mask * (-1 / config.tokens ** 2))
                       + self.row_plain(mask * (2 / config.tokens)))
        else:
            # First Newton step from 1/tokens, without an encrypted seed.
            inverse = (denominator * self.constant(-1 / self.config.tokens)
                       + self.constant(2)) * self.constant(1 / self.config.tokens)
        for _ in range(self.config.reciprocal_iterations - 1):
            correction = denominator * inverse * self.constant(-1) + self.constant(2)
            inverse = inverse * correction
        return inverse

    def attention(self, queries, keys, values, label, *, prepared=False):
        if self.config.attention_layout == "cyclic":
            if not prepared:
                queries, keys, values = [self.expand_heads(batch) for batch in (queries, keys, values)]
            return self.attention_cyclic(queries, keys, values, label)
        return self.attention_offset(queries, keys, values, label)

    def attention_offset(self, queries, keys, values, label):
        config = self.config
        means = []
        for query, key in zip(queries, keys):
            key_total = key
            shift = 1
            while shift < config.tokens:
                key_total = key_total + key_total.rotate(shift * config.tile)
                shift *= 2
            mean_key = key_total * self.constant(1 / config.tokens)
            # mean_j <Q_i,K_j> == <Q_i,mean_j K_j>: no score frontier.
            means.append(self.head_dot(query, mean_key))
        denominators, numerators = [None] * len(queries), [None] * len(queries)
        for offset in range(config.tokens):
            rotation = offset * config.tile
            for group, (query, key, value) in enumerate(zip(queries, keys, values)):
                shifted_key = key.rotate(rotation) if rotation else key
                shifted_value = value.rotate(rotation) if rotation else value
                score = self.head_dot(query, shifted_key) - means[group]
                exp = self.exponential(score)
                weighted = exp * shifted_value
                denominators[group] = exp if denominators[group] is None else denominators[group] + exp
                numerators[group] = weighted if numerators[group] is None else numerators[group] + weighted
        outputs = []
        for group in range(len(queries)):
            suffix = f".group{group}"
            denominator = self.stage(denominators[group], label + ".denominator" + suffix)
            numerator = self.stage(numerators[group], label + ".weighted_value" + suffix)
            inverse = self.stage(self.reciprocal(denominator), label + ".reciprocal" + suffix)
            outputs.append(self.stage(numerator * inverse, label + ".context" + suffix))
        return outputs

    def expand_heads(self, inputs):
        """[h0,h1,h2,h3] -> [h0,0,h2,0], [h1,0,h3,0] per CT.

        Each output has two T-lane head groups with only the first head_dim
        lanes meaningful. This is Cerium's head order (0,2,1,3,4,6,5,7,...).
        """
        cache_key = ("head_pair_mask",)
        if cache_key not in self.constants:
            row = np.zeros(self.config.tile, dtype=np.float64)
            row[:self.config.head_dim] = 1
            row[self.config.tokens:self.config.tokens + self.config.head_dim] = 1
            self.constants[cache_key] = self.row_plain(row)
        mask = self.constants[cache_key]
        outputs = []
        for expr in inputs:
            outputs.extend((expr * mask, expr.rotate(self.config.head_dim) * mask))
        return outputs

    def merge_heads(self, inputs):
        """Invert expand_heads after AV; the padded head lanes must be zero."""
        return [inputs[i] + inputs[i + 1].rotate(-self.config.head_dim)
                for i in range(0, len(inputs), 2)]

    def group_row_sum(self, expr, factor=1.0):
        """Sum each contiguous T-lane score row and broadcast within that row."""
        width = self.config.tokens
        shift = 1
        while shift < width:
            expr = expr + expr.rotate(shift)
            shift *= 2
        key = ("score_row_starts", float(factor))
        if key not in self.constants:
            mask = np.zeros(self.config.slots, dtype=np.float64)
            mask[::width] = factor
            self.constants[key] = self.backend.plain(mask)
        expr = expr * self.constants[key]
        shift = 1
        while shift < width:
            expr = expr + expr.rotate(-shift)
            shift *= 2
        return expr

    def cyclic_qk(self, query, key, *, label=None):
        """Cerium QK BSGS, with no HE-specific rescale/level instructions.

        Output slot tile*t + T*k + d is
        dot(Q_head[t], K_head[(t+d)%T])/sqrt(head_dim), k in {0,1}.
        The full score frontier is just heads/2 CTs. See the reference
        bert_cerium.py:matmul_ct128x128_ct128x128_transpose_function.
        """
        hook = getattr(self.backend, 'compact_qk', None)
        if callable(hook):
            prepared = hook(query, key, config=self.config, label=label)
            if prepared is not None:
                return prepared
        config = self.config
        baby = min(8, config.tokens)
        giants = config.tokens // baby
        keys = [key if b == 0 else key.rotate(config.tile * b) for b in range(baby)]
        mask_key = ("cyclic_qk_starts",)
        if mask_key not in self.constants:
            mask = np.zeros(config.slots, dtype=np.float64)
            mask[::config.tokens] = 1 / math.sqrt(config.head_dim)
            self.constants[mask_key] = self.backend.plain(mask)
        output = None
        for giant in range(giants):
            rotation = -config.tile * baby * giant
            shifted_query = query if not rotation else query.rotate(rotation)
            subtotal = None
            for little, shifted_key in enumerate(keys):
                product = shifted_query * shifted_key
                shift = 1
                while shift < config.head_dim:
                    product = product + product.rotate(shift)
                    shift *= 2
                product = product * self.constants[mask_key]
                if little:
                    product = product.rotate(-little)
                subtotal = product if subtotal is None else subtotal + product
            rotation = (config.tile * baby - baby) * giant
            if rotation:
                subtotal = subtotal.rotate(rotation)
            output = subtotal if output is None else output + subtotal
        return output

    def _cyclic_av_mask(self, kind, shift):
        key = ("cyclic_av", kind, shift)
        if key not in self.constants:
            width = self.config.tokens
            row = np.zeros(width, dtype=np.float64)
            if kind == "diagonal":
                row[shift] = 1
            elif kind == "left":
                row[:width - shift] = 1
            else:
                row[width - shift:] = 1
            self.constants[key] = self.backend.plain(np.tile(row, self.config.slots // width))
        return self.constants[key]

    def cyclic_av(self, scores, value, *, label=None):
        """Cerium BSGS weighted-V, before normalization; no head broadcasts.

        scores is a full cyclic score/exp CT, value is a gapped two-head CT.
        First form diag[t,k,j] = V[(t+j)%T,k,j] using BSGS and shift::T
        masks. Then cyclic-within-T rotations (maskL/maskR) give
        sum_d scores[t,k,d]*V[(t+d)%T,k,j]. Padded value lanes remain zero.
        This translates bert_cerium.py:matmul_ct128x128_ct128x128_function.
        """
        hook = getattr(self.backend, 'compact_av', None)
        if callable(hook):
            prepared = hook(scores, value, config=self.config, label=label)
            if prepared is not None:
                return prepared
        config = self.config
        baby = min(8, config.tokens)
        giants = config.tokens // baby
        values = [value if b == 0 else value.rotate(config.tile * b) for b in range(baby)]
        diagonal = None
        for giant in range(giants):
            subtotal = None
            for little, shifted_value in enumerate(values):
                shift = baby * giant + little
                # Giant rotations are multiples of tile=2*T, so these
                # T-periodic masks need no separate giant compensation.
                term = shifted_value * self._cyclic_av_mask("diagonal", shift)
                subtotal = term if subtotal is None else subtotal + term
            rotation = config.tile * baby * giant
            if rotation:
                subtotal = subtotal.rotate(rotation)
            diagonal = subtotal if diagonal is None else diagonal + subtotal
        diagonals = [diagonal if b == 0 else diagonal.rotate(config.tile * b) for b in range(baby)]
        output = None
        for giant in range(giants):
            rotation = (baby - config.tile * baby) * giant
            base = scores if not rotation else scores.rotate(rotation)
            subtotal = None
            for little, shifted_diagonal in enumerate(diagonals):
                shift = baby * giant + little
                if not shift:
                    cyclic = base  # maskL=1, maskR=0: do not add redundant CPs.
                else:
                    left = base if not little else base.rotate(little)
                    right = base.rotate(little - config.tokens)
                    cyclic = (left * self._cyclic_av_mask("left", shift)
                              + right * self._cyclic_av_mask("right", shift))
                term = cyclic * shifted_diagonal
                subtotal = term if subtotal is None else subtotal + term
            rotation = config.tile * baby * giant
            if rotation:
                subtotal = subtotal.rotate(rotation)
            output = subtotal if output is None else output + subtotal
        return output

    def attention_cyclic(self, queries, keys, values, label):
        outputs = []
        for group, (query, key, value) in enumerate(zip(queries, keys, values)):
            suffix = f".group{group}"
            scores = self.stage(self.cyclic_qk(query, key, label=label + ".qk" + suffix), label + ".qk" + suffix)
            means = self.group_row_sum(scores, 1 / self.config.tokens)
            exp = self.stage(self.exponential(scores - means), label + ".exp" + suffix)
            denominator = self.stage(self.group_row_sum(exp), label + ".denominator" + suffix)
            inverse = self.stage(self.reciprocal(denominator,
                zero_head_gaps=bool(getattr(self.backend, 'av_pack_inputs', False))),
                label + ".reciprocal" + suffix)
            numerator = self.stage(self.cyclic_av(exp, value, label=label + ".weighted_value" + suffix),
                                   label + ".weighted_value" + suffix)
            outputs.append(self.stage(numerator * inverse, label + ".context" + suffix))
        return self.merge_heads(outputs)

    def layer_norm(self, inputs, gamma, beta, label):
        config = self.config
        total = inputs[0]
        for expr in inputs[1:]:
            total = total + expr
        mean = self.row_total(total, 1 / config.hidden)
        centered = [expr - mean for expr in inputs]
        square_total = centered[0] * centered[0]
        for expr in centered[1:]:
            square_total = square_total + expr * expr
        variance = self.row_total(square_total, 1 / config.hidden) + self.constant(config.ln_epsilon)
        # One statistic CT holds all tokens and is shared by ALL feature tiles.
        variance = self.stage(variance, label + ".variance.group0")
        u = variance * self.constant(1 / config.ln_variance_center) - self.constant()
        square = u * u
        cube = square * u
        inverse = (u * self.constant(-0.5) + self.constant()
                   + square * self.constant(3 / 8) - cube * self.constant(5 / 16))
        outputs = []
        for group, expr in enumerate(centered):
            start = group * config.tile
            scale = self.row_plain(np.asarray(gamma[start:start + config.tile], dtype=np.float64)
                                   / math.sqrt(config.ln_variance_center))
            offset = self.row_plain(beta[start:start + config.tile])
            outputs.append(self.stage(expr * inverse * scale + offset, f"{label}.output.group{group}"))
        return outputs

    def gelu(self, expr):
        square = expr * expr
        fourth = square * square
        coefficient = 1 / math.sqrt(2 * math.pi)
        return (expr * self.constant(0.5) + square * self.constant(coefficient)
                - fourth * self.constant(coefficient / 6))


def _check_linear(inputs, weight, bias, config):
    inputs = list(inputs)
    weight = _array(weight, "weight")
    if (weight.ndim != 2 or min(weight.shape) < 1
            or any(size % config.tile for size in weight.shape)):
        raise ValueError("weight must be nonempty [out,in], both dimensions divisible by tile")
    if len(inputs) != weight.shape[1] // config.tile:
        raise ValueError("input tile count must match weight input width")
    if bias is not None:
        bias = _array(bias, "bias", (weight.shape[0],))
    return inputs, weight, bias


def linear(backend, inputs, weight, bias=None, *, config, baby_step=None, label="linear.weight"):
    """Public full-width rectangular linear; dispatch hook or standard Expr BSGS."""
    inputs, weight, bias = _check_linear(inputs, weight, bias, config)
    if baby_step is not None:
        baby_step = _integer(baby_step, "baby_step")
    return _Arithmetic(backend, config).linear(inputs, weight, bias, label, baby_step)


def shift_tokens(backend, inputs, offset, *, config, width=None):
    """Each feature CT independently gathers token (q+offset)%tokens."""
    offset = _integer(offset, "offset", None) % config.tokens
    width = _width(config, width)
    inputs = list(inputs)
    if len(inputs) != width // config.tile:
        raise ValueError("input tile count must match width")
    return [expr.rotate(offset * config.tile) for expr in inputs] if offset else inputs


def attention(backend, queries, keys, values, *, config, boot_policy="none"):
    """Bidirectional projected-Q/K/V attention, all tokens in each feature CT."""
    batches = [list(batch) for batch in (queries, keys, values)]
    if any(len(batch) != config.hidden_tiles for batch in batches):
        raise ValueError(f"Q/K/V must each contain {config.hidden_tiles} ciphertexts")
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
    """Build post-LN BERT; return hidden//tile ciphertexts (default three).

    Hook labels are layer{n}.{q,k,v,o,ffn_up,ffn_down}_weight. Stage labels
    retain .group{tile_index} for compatibility with existing backend selectors.
    This function neither chooses physical levels nor constructs HE parameters.
    """
    inputs, weights = list(inputs), list(weights)
    if len(inputs) != config.hidden_tiles:
        raise ValueError(f"expected {config.hidden_tiles} input ciphertexts")
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
    hidden = inputs
    for index, layer in enumerate(checked):
        label = f"layer{index}"

        def project(batch, name):
            return arithmetic.linear(batch, layer[name + "_weight"], layer[name + "_bias"],
                                     f"{label}.{name}_weight")

        projected = {}
        for name in ("q", "k", "v"):
            batch = project(hidden, name)
            if config.attention_layout == "cyclic":
                batch = arithmetic.expand_heads(batch)
            projected[name] = [arithmetic.stage(expr, f"{label}.{name}.group{group}")
                               for group, expr in enumerate(batch)]
        contexts = arithmetic.attention(projected["q"], projected["k"], projected["v"],
                                        label + ".attention", prepared=True)
        residuals = [arithmetic.stage(expr + residual, f"{label}.attention_residual.group{group}")
                     for group, (expr, residual) in enumerate(zip(project(contexts, "o"), hidden))]
        attended = arithmetic.layer_norm(residuals, layer["attention_ln_gamma"], layer["attention_ln_beta"],
                                        label + ".attention_ln")
        up = [arithmetic.stage(expr, f"{label}.ffn_up.group{group}")
              for group, expr in enumerate(project(attended, "ffn_up"))]
        activated = [arithmetic.stage(arithmetic.gelu(expr), f"{label}.gelu.group{group}")
                     for group, expr in enumerate(up)]
        residuals = [arithmetic.stage(expr + residual, f"{label}.ffn_residual.group{group}")
                     for group, (expr, residual) in enumerate(zip(project(activated, "ffn_down"), attended))]
        hidden = arithmetic.layer_norm(residuals, layer["ffn_ln_gamma"], layer["ffn_ln_beta"], label + ".ffn_ln")
    return hidden
