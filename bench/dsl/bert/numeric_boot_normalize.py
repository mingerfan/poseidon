"""Opt-in public-token normalization of attention denominator Boot hints.

For the stage labels emitted by cerium_packing._Arithmetic.attention and
attention_cyclic, replace Boot(x, label) by T * Boot(x / T, SAME label).
The recognizer requires the complete ``.attention.denominator.groupN`` suffix,
where N is the canonical nonnegative decimal group index; substring matches,
other attention stages and LayerNorm hints are not eligible.

Integration order (all passes return fresh Expr graphs)::

    graph, outputs, normalization = normalize_denominator_bootstraps(
        graph, outputs, tokens=config.tokens)
    # Optional equivalent_rewrite.rewrite_compact_graph, then
    # polynomial_rewrite.reassociate_polynomials, then fresh physical lowering.

Calling this function is the opt-in; existing generation paths are unaffected.
The factors are public full-physical-slot vectors, not observations or fitted
constants. This is an ideal-Boot real-arithmetic identity, not a complete Boot
or BERT accuracy fix or a ciphertext-bitwise equivalence claim. The genuine CP
products can consume levels and change noise/scheduling: regenerate the physical
plan, never wrap an existing plan or edit scale/level metadata.
"""
from __future__ import annotations

from copy import deepcopy
import math
import operator
import re

import numpy as np

try:
    from .cerium_lowering import CompactGraph
except ImportError:
    from cerium_lowering import CompactGraph


_DENOMINATOR_STAGE = re.compile(r'.+\.attention\.denominator\.group(?:0|[1-9][0-9]*)')


def _token_factors(tokens):
    message = 'tokens must be a finite positive power-of-two integer'
    if isinstance(tokens, (bool, np.bool_)):
        raise ValueError(message)
    try:
        tokens = operator.index(tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if tokens < 1 or tokens & (tokens - 1):
        raise ValueError(message)
    try:
        factor = float(tokens)
    except OverflowError as exc:
        raise ValueError(message) from exc
    inverse = 1.0 / factor
    if not math.isfinite(factor) or not math.isfinite(inverse) or inverse <= 0:
        raise ValueError(message)
    return tokens, inverse, factor


def normalize_denominator_bootstraps(graph, outputs, *, tokens):
    """Return (independent CompactGraph, output list, JSON-compatible stats).

    Only reachable ``boot`` nodes with an exact attention-denominator stage
    label are normalized. Each original hint keeps its label and identity
    (distinct hints are not merged), and all its consumers share the restored
    output. Other uses of its input remain unscaled. All original operations,
    coefficients, dense matrix IDs/bytes and prepared packing specifications
    are cloned without arithmetic rewrites or mutation of the source graph.

    ``tokens`` must implement integer indexing, be positive and power-of-two,
    and have finite, nonzero float64 factors T and 1/T. Powers of two avoid
    introducing rounded public coefficients. This does not guarantee that an
    arbitrary denominator is within native Boot's supported message range.
    Run this pass BEFORE algebra/polynomial passes and physical lowering; the
    latter must account for real multiplication/rescale costs on both sides.
    """
    tokens, inverse, factor = _token_factors(tokens)
    outputs = list(outputs)
    if any(getattr(node, 'backend', None) is not graph for node in outputs):
        raise ValueError('outputs must belong to the source graph')
    reachable = graph.reachable(outputs)
    target = CompactGraph(graph.slots,
        prepared_qk=getattr(graph, 'use_prepared_qk', False),
        prepared_av=getattr(graph, 'use_prepared_av', False),
        qk_pack_inputs=getattr(graph, 'qk_pack_inputs', False),
        av_pack_inputs=getattr(graph, 'av_pack_inputs', False))
    target.qk_specs = deepcopy(getattr(graph, 'qk_specs', {}))
    target.av_specs = deepcopy(getattr(graph, 'av_specs', {}))
    target.matrices = deepcopy(getattr(graph, 'matrices', {}))
    mapped, labels = {}, []

    def scalar(value):
        # Short plain payloads have zero tails. Both factors MUST occupy every
        # physical slot, including head gaps and unused/padded token rows.
        return target.plain(np.full(target.slots, value, dtype=np.float64))

    for node in reachable:
        args = tuple(mapped[arg.index] for arg in node.args)
        if node.op == 'input':
            result = target.input(deepcopy(node.data))
        elif node.op == 'plain':
            result = target.plain(graph.constants[node.data])
        elif (node.op == 'boot' and isinstance(node.data, str) and
                _DENOMINATOR_STAGE.fullmatch(node.data)):
            divided = target.node('mul', (args[0], scalar(inverse)))
            refreshed = target.bootstrap(divided, label=node.data)
            result = target.node('mul', (refreshed, scalar(factor)))
            labels.append(node.data)
        else:
            # Clone verbatim, rather than using binary(): even its ordinary
            # constant/mask folding is outside this pass's narrow remit.
            result = target.node(node.op, args, deepcopy(node.data))
        mapped[node.index] = result

    rewritten = [mapped[node.index] for node in outputs]
    stats = dict(enabled=True, tokens=tokens, normalized_labels=labels,
        rewrite_counts=({'denominator_bootstraps_normalized': len(labels)} if labels else {}),
        source_reachable_nodes=len(reachable),
        result_reachable_nodes=len(target.reachable(rewritten)),
        metadata_only_rewrites=False, approximation_changed=False,
        ciphertext_bitwise_equivalent=False, requires_physical_relowering=True,
        note='ideal Boot identity; real CP products may consume levels and change CKKS noise; not a complete accuracy fix')
    return target, rewritten, stats
