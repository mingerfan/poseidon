"""Trusted arithmetic interfaces for fixed-prefix encrypted greedy generation.

The selector produces an approximate one-hot vector.  Its independent
arithmetic check and its agreement with discrete greedy selection are separate
questions.  A declared comparison gap is a deployment assumption, recorded by
the plaintext diagnostic and never inferred from encrypted values here.

These helpers use existing backend operations and leave the original model,
weights and DSL unchanged.  They perform no decryption or encryption calls.
"""

from dataclasses import dataclass, field
import math


def _numpy():
    import numpy as np
    return np


def validate_coefficients(coefficients):
    np = _numpy()
    stages = tuple(np.asarray(stage, dtype=np.float64) for stage in coefficients)
    if not stages:
        raise ValueError('An explicit composite half-sign polynomial is required')
    for stage in stages:
        if (stage.ndim != 1 or len(stage) < 2 or not np.isfinite(stage).all()
                or np.any(stage[::2] != 0) or not np.any(stage[1::2] != 0)):
            raise ValueError('Half-sign stages require finite nonconstant odd Chebyshev coefficients')
    return stages


def composite_half_sign(degrees, normalized_gap):
    """Reuse the existing odd composite fit; its last stage already halves sign."""
    from .approx import composite_coeffs
    if not math.isfinite(normalized_gap) or not 0 < normalized_gap < 1:
        raise ValueError('Composite selection requires a declared positive normalized comparison gap')
    return validate_coefficients(composite_coeffs(tuple(degrees), normalized_gap))


def _contract(bound, normalized_gap):
    if (not math.isfinite(bound) or bound <= 0 or not math.isfinite(normalized_gap)
            or not 0 < normalized_gap < 1):
        raise ValueError('Selection requires a finite difference bound and normalized gap in (0,1)')


def _check_values(backend, values, count):
    if len(values) != count or any(getattr(value, 'backend', None) is not backend for value in values):
        raise ValueError('Packed generation values must belong to the same backend and layout')


class _PatternMasks:
    """A small per-selection inventory, independent of vocabulary offsets."""
    def __init__(self, backend):
        self.backend, self.capacity, self.cache = backend, backend.slots//2, {}

    def _plain(self, key, build):
        if key not in self.cache:
            np = _numpy()
            vector = np.zeros(self.backend.slots, dtype=np.float64)
            build(vector)
            self.cache[key] = self.backend.const(vector)
        return self.cache[key]

    def prefix(self, size):
        def build(vector):
            vector[:size] = 1.0
        return self._plain(('prefix', size), build)

    def parity(self, size, parity):
        def build(vector):
            vector[parity:size:2] = 1.0
        return self._plain(('parity', size, parity), build)

    def bit(self, bit, present):
        def build(vector):
            np = _numpy()
            indices = np.arange(self.capacity)
            vector[:self.capacity] = ((indices & bit) != 0) == present
        return self._plain(('bit', bit, present), build)

    def one(self, position):
        def build(vector):
            vector[position] = 1.0
        return self._plain(('one', position), build)


def _compress_parity(value, size, parity, masks):
    """Gather even/odd positions using log(capacity) two-way bit moves."""
    value = value * masks.parity(size, parity)
    if parity:
        value = value.rotate(1)
    shift = 1
    while shift < masks.capacity//2:
        value = value*masks.bit(2*shift, False) + (value*masks.bit(2*shift, True)).rotate(shift)
        shift *= 2
    return value


def _split_pairs(backend, values, layout, masks):
    from .layout import layout_blocks
    halves = [[], []]
    for parity in (0, 1):
        compressed = [_compress_parity(value, stop-start, parity, masks)
                      for value, (start, stop, _) in zip(values, layout_blocks(layout))]
        for index in range(0, len(compressed), 2):
            value = compressed[index]
            if index+1 < len(compressed):
                value = value + compressed[index+1].rotate(-masks.capacity//2)
            halves[parity].append(value)
    return halves


def _expand_half(value, upper, masks):
    if upper:
        value = value.rotate(masks.capacity//2)
    value = value * masks.prefix(masks.capacity//2)
    shift = masks.capacity//4
    while shift:
        value = value*masks.bit(shift, False) + (value*masks.bit(shift, True)).rotate(-shift)
        shift //= 2
    return value


def _merge_pairs(left, right, child_layout, masks):
    from .layout import layout_blocks
    result = []
    for index, _ in enumerate(layout_blocks(child_layout)):
        source, upper = index//2, bool(index % 2)
        even = _expand_half(left[source], upper, masks)
        odd = _expand_half(right[source], upper, masks).rotate(-1)
        result.append(even+odd)
    return result


def select_maximum(backend, logits, vocabulary, *, bound, normalized_gap, coefficients):
    """Packed tournament plus reverse path products; never inspect slot values."""
    from .layout import layout_blocks, multi_flat_layout
    from .poly import cheb_eval
    _contract(bound, normalized_gap)
    stages = validate_coefficients(coefficients)
    if type(vocabulary) is not int or vocabulary < 2 or backend.slots < 4:
        raise ValueError('Encrypted selection requires at least two vocabulary entries')
    masks = _PatternMasks(backend)
    original = multi_flat_layout(backend.slots, vocabulary)
    _check_values(backend, logits, len(layout_blocks(original)))
    current, layout, width = list(logits), original, vocabulary
    tree = []
    while width > 1:
        parents = (width + 1) // 2
        target = multi_flat_layout(backend.slots, parents)
        left, right = _split_pairs(backend, current, layout, masks)
        real_pairs = width // 2
        if width % 2:
            carry = masks.one((parents-1) % masks.capacity)
            right[-1] = right[-1] + left[-1]*carry
        choices, maximum = [], []
        for index, (a, b, (start, stop, _)) in enumerate(zip(left, right, layout_blocks(target))):
            difference = a - b
            sign = difference * (1.0 / bound)
            for coefficients_one_stage in stages:
                sign = cheb_eval(backend, sign, coefficients_one_stage)
            active = masks.prefix(min(stop, real_pairs)-start)
            choice = (sign + 0.5) * active
            if width % 2 and index == len(left)-1:
                choice = choice + carry
            choices.append(choice)
            maximum.append(b + difference * choice)
        tree.append((width, layout, target, choices))
        current, layout, width = maximum, target, parents
    importance = [current[0] * 0.0 + masks.prefix(1)]
    for children, child_layout, parent_layout, choices in reversed(tree):
        left = [weight * choose for weight, choose in zip(importance, choices)]
        right = [weight * (1.0 - choose) for weight, choose in zip(importance, choices)]
        importance = _merge_pairs(left, right, child_layout, masks)
    return dict(one_hot=importance, layout=original, maximum=current[0],
                metadata=dict(vocabulary=vocabulary, tournament_rounds=len(tree),
                    declared_difference_bound=bound, declared_normalized_gap=normalized_gap,
                    degrees=[len(stage)-1 for stage in stages],
                    general_gather_calls=0, pattern_mask_count=len(masks.cache),
                    pattern_mask_logical_bytes=len(masks.cache)*backend.slots*8,
                    reordering='log-capacity parity compression/expansion',
                    output_semantics='approximate one-hot; greedy agreement needs independent quality observation',
                    intermediate_decryptions=0))


def embedding_from_selection(backend, selection, original_embedding):
    """Clear original embedding weights multiply the encrypted selection vector."""
    from ..backends.emit import EmitBackend
    if isinstance(backend, EmitBackend):
        raise ValueError('Embedding Emit requires a separate constant-storage implementation')
    np = _numpy()
    from .transformer import linear, values
    embedding = np.asarray(original_embedding)
    vocabulary = selection['metadata']['vocabulary']
    if embedding.dtype != np.float64 or embedding.ndim != 2 or embedding.shape[0] != vocabulary or embedding.shape[1] < 1:
        raise ValueError('Embedding selection requires the original float64 [vocabulary,hidden] matrix')
    for start in range(0, vocabulary, 128):
        if not np.isfinite(embedding[start:start+128]).all():
            raise ValueError('Original embedding weights must be finite')
    weight_view = embedding.T
    from ..backends.count import CountBackend
    from ..backends.streaming_mock import StreamingMockBackend
    from ..backends.streaming_emit import StreamingEmitBackend
    if isinstance(backend, (CountBackend, StreamingMockBackend, StreamingEmitBackend)):
        from .streaming_linear import streaming_linear
        result, layout = streaming_linear(backend, selection['one_hot'], selection['layout'], weight_view)
    else:
        result, layout = linear(backend, selection['one_hot'], selection['layout'], weight_view)
    return list(values(result, layout)), layout


@dataclass
class EncryptedPrefix:
    backend: object
    token_blocks: list
    prompt_length: int
    hidden_size: int
    sequence_length: int = 32
    selections: list = field(default_factory=list)

    def __post_init__(self):
        from .layout import TokenLayout, layout_blocks
        if (type(self.prompt_length) is not int or not 1 <= self.prompt_length <= self.sequence_length-2
                or len(self.token_blocks) != self.sequence_length):
            raise ValueError('Two-token generation requires an original prefix and two padding positions')
        count = len(layout_blocks(TokenLayout(self.backend.slots, 1, self.hidden_size).token))
        for token in self.token_blocks:
            _check_values(self.backend, token, count)
        self.token_blocks = [list(token) for token in self.token_blocks]

    @property
    def length(self):
        return self.prompt_length + len(self.selections)

    def public_masks(self):
        np = _numpy()
        attention = np.zeros((1, self.sequence_length), dtype=np.float64)
        attention[0, :self.length] = 1.0
        selected = np.zeros_like(attention)
        selected[0, self.length-1] = 1.0
        return dict(attention_mask=attention, sel=selected)

    def append(self, embedding, selection):
        if len(self.selections) >= 2 or self.length >= self.sequence_length:
            raise ValueError('The fixed two-token generation state is complete')
        _check_values(self.backend, embedding, len(self.token_blocks[self.length]))
        self.token_blocks[self.length] = list(embedding)
        self.selections.append(selection)


def generate_two(state, frozen_forward, original_embedding, *, bound, normalized_gap, coefficients):
    """Exactly two whole-prefix forwards; keys/context remain owned by the caller.

    ``frozen_forward(state)`` returns packed last-token logits on the same
    backend.  This callback will later bind the full, unchanged 24-layer model.
    All generated states retain ciphertext handles, including the next token's
    embedding.  Client-side decoding is a separate final operation.
    """
    if state.selections:
        raise ValueError('A fresh fixed-prefix state is required')
    vocabulary = original_embedding.shape[0]
    for _ in range(2):
        logits = frozen_forward(state)
        selection = select_maximum(state.backend, logits, vocabulary, bound=bound,
            normalized_gap=normalized_gap, coefficients=coefficients)
        embedding, _ = embedding_from_selection(state.backend, selection, original_embedding)
        state.append(embedding, selection)
    return dict(state=state, token_selections=[item['one_hot'] for item in state.selections],
                metadata=dict(model_forward_calls=2, embedding_selections=2,
                    full_prefix_recomputations=2, kv_cache_used=False,
                    intermediate_decryptions=0, intermediate_reencryptions=0,
                    generated_tokens=2, fixed_server_step_count=2,
                    original_model_binding_verified=False, real_ckks_executions=0))


def eligible_prompts(tokenizer, texts, sample_ids, *, sequence_length=32, new_tokens=2, limit=3):
    if len(texts) != len(sample_ids) or new_tokens != 2 or sequence_length != 32 or type(limit) is not int or limit < 1:
        raise ValueError('The fixed Qwen interface uses original texts, sequence 32 and two generated tokens')
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Original real prompt texts are required')
    encoded = tokenizer(list(texts), padding=False, truncation=False)
    lengths = [len(tokens) for tokens in encoded['input_ids']]
    if len(lengths) != len(texts):
        raise ValueError('Tokenizer returned a different prompt count')
    selected = [index for index, length in enumerate(lengths) if 1 <= length <= sequence_length-new_tokens][:limit]
    if len(selected) != limit:
        raise ValueError('Original prompts do not supply enough prefixes with two padding positions')
    return dict(indices=selected, sample_ids=[sample_ids[index] for index in selected],
                token_lengths=[lengths[index] for index in selected], all_token_lengths=lengths,
                padding_positions_reserved=2, truncation=False, sample_texts_modified=False)


def numpy_selection(logits, *, bound, normalized_gap, coefficients):
    """Independent array implementation for mathematical and quality diagnostics."""
    np = _numpy()
    _contract(bound, normalized_gap)
    stages = validate_coefficients(coefficients)
    current = np.asarray(logits)
    if current.dtype != np.float64 or current.ndim != 1 or len(current) < 2 or not np.isfinite(current).all():
        raise ValueError('Selection reference requires finite float64 logits')
    tree, pairs = [], []
    while len(current) > 1:
        width = len(current)
        left = current[::2]
        right = current[np.minimum(np.arange(0, width, 2)+1, width-1)]
        difference = left-right
        for value in difference[:width//2]:
            pairs.append(float(value))
        sign = difference * (1.0 / bound)
        for stage in stages:
            sign = np.polynomial.chebyshev.chebval(sign, stage)
        choice = sign + 0.5
        if width % 2:
            choice[-1] = 1.0
        tree.append((width, choice))
        current = right + difference*choice
    importance = np.ones(1, dtype=np.float64)
    for width, choice in reversed(tree):
        left, right = importance*choice, importance*(1.0-choice)
        importance = np.empty(width, dtype=np.float64)
        importance[::2] = left
        importance[1::2] = right[:width//2]
    distances = np.abs(np.asarray(pairs)) / bound
    return importance, dict(comparisons=len(pairs), min_normalized_comparison_gap=float(distances.min()),
        max_normalized_comparison_difference=float(distances.max()),
        comparison_gap_contract_passed=bool(np.all(distances >= normalized_gap) and np.all(distances <= 1.0)))


def selection_quality(logits, observed_one_hot, *, bound, normalized_gap, coefficients):
    np = _numpy()
    expected, contract = numpy_selection(logits, bound=bound, normalized_gap=normalized_gap, coefficients=coefficients)
    observed = np.asarray(observed_one_hot)
    if observed.dtype != np.float64 or observed.shape != expected.shape or not np.isfinite(observed).all():
        raise ValueError('Selection diagnostics require finite matching float64 vectors')
    difference = np.abs(observed-expected)
    tolerance = 1e-12 + 1e-12*np.abs(expected)
    token = int(np.argmax(logits))
    one_hot = np.zeros_like(expected)
    one_hot[token] = 1.0
    return dict(contract, arithmetic_passed=bool(np.all(difference <= tolerance)),
        arithmetic_max_absolute_error=float(difference.max()),
        arithmetic_criterion='1e-12 + 1e-12*abs(reference)',
        original_greedy_token=token, approximate_token=int(np.argmax(observed)),
        token_agreement=bool(np.argmax(observed) == token),
        approximate_one_hot_max_error=float(np.max(np.abs(observed-one_hot))),
        approximate_one_hot_sum=float(observed.sum()),
        exact_discrete_one_hot_claimed=False, original_qwen_execution_verified=False)


def decode_final_tokens(decrypted_one_hots, *, eos_token_id):
    """Client-only postprocessing after the complete fixed server computation."""
    np = _numpy()
    values = np.asarray(decrypted_one_hots)
    if (values.dtype != np.float64 or values.ndim != 2 or len(values) != 2
            or not np.isfinite(values).all() or type(eos_token_id) is not int
            or not 0 <= eos_token_id < values.shape[1]):
        raise ValueError('Final decoding requires the two decrypted finite selection vectors')
    tokens = []
    for value in values:
        token = int(np.argmax(value))
        tokens.append(token)
        if token == eos_token_id:
            break
    return tokens
