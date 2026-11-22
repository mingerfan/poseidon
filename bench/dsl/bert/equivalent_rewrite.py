"""Conservative real-arithmetic rewrites for the compact BERT Expr graph.

No HE metadata is edited. The rewritten graph must be physically lowered again.
Only FULL-slot scalar +/-1 and the first reciprocal Newton step are recognized;
prefix/head/row masks and arbitrary BSGS coefficient folding are not rewritten.
"""
from __future__ import annotations

from collections import Counter
import operator
import numpy as np

try:
    from .cerium_lowering import CompactGraph
except ImportError:
    from cerium_lowering import CompactGraph


def _full_scalar(graph, expr):
    if expr.op != 'plain':
        return None
    values = graph.constants[expr.data]
    # A short payload has a ZERO tail, not implicit repetition. In particular,
    # a prefix of ones or minus-ones is NEVER a full-slot scalar.
    if len(values) != graph.slots or not np.all(values == values[0]):
        return None
    return float(values[0])


def rewrite_compact_graph(graph, outputs, *, tokens, scalar_units=True,
                          reciprocal_seed=True):
    """Return (independent rewritten graph, outputs, rewrite statistics).

    The reciprocal seed rewrite is limited to this exact BERT form:
      ((d * (-1/T)) + 2) * (1/T) -> d * (-1/T**2) + 2/T.
    All three Newton iterations, their ciphertext products and all nonlinear
    terms remain. T must be a power of two, so public seed coefficients are
    exactly representable. Float addition association can still change rounding.
    """
    if isinstance(tokens, (bool, np.bool_)):
        raise ValueError('tokens must be a positive power of two')
    tokens = operator.index(tokens)
    if tokens < 1 or tokens & (tokens - 1):
        raise ValueError('tokens must be a positive power of two')
    outputs = list(outputs)
    if any(expr.backend is not graph for expr in outputs):
        raise ValueError('outputs must belong to the source graph')
    target = CompactGraph(graph.slots, prepared_qk=getattr(graph, 'use_prepared_qk', False),
                          prepared_av=getattr(graph, 'use_prepared_av', False),
                          qk_pack_inputs=getattr(graph, 'qk_pack_inputs', False),
                          av_pack_inputs=getattr(graph, 'av_pack_inputs', False))
    target.qk_specs = {label: dict(spec) for label, spec in getattr(graph, 'qk_specs', {}).items()}
    target.av_specs = {label: dict(spec) for label, spec in getattr(graph, 'av_specs', {}).items()}
    for label, matrix in getattr(graph, 'matrices', {}).items():
        target.matrices[label] = {key: value.copy() if isinstance(value, np.ndarray) else value
                                  for key, value in matrix.items()}
    mapped = {}
    counts = Counter()
    inv_tokens = 1.0 / tokens

    def scalar(value):
        return target.plain(np.full(target.slots, value, dtype=np.float64))

    for node in graph.reachable(outputs):
        args = [mapped[arg.index] for arg in node.args]
        if node.op == 'input':
            result = target.input(node.data)
        elif node.op == 'plain':
            result = target.plain(graph.constants[node.data])
        elif node.op in ('compact_linear', 'compact_qk', 'compact_av', 'compact_av_partial'):
            result = target.node(node.op, tuple(args), node.data)
        elif node.op == 'boot':
            result = target.bootstrap(args[0], label=node.data)
        elif node.op == 'rotate':
            result = args[0].rotate(node.data)
        elif node.op == 'neg':
            result = -args[0]
        elif node.op in ('add', 'sub', 'mul'):
            matched = False
            if reciprocal_seed and tokens > 1 and node.op == 'mul' and _full_scalar(graph, node.args[1]) == inv_tokens:
                addition = node.args[0]
                if addition.op == 'add' and _full_scalar(graph, addition.args[1]) == 2.0:
                    product = addition.args[0]
                    if (product.op == 'mul' and
                            _full_scalar(graph, product.args[1]) == -inv_tokens):
                        source = mapped[product.args[0].index]
                        result = source * scalar(-(inv_tokens * inv_tokens)) + scalar(2 * inv_tokens)
                        counts['reciprocal_first_steps_folded'] += 1
                        matched = True
            if not matched and scalar_units and node.op == 'mul':
                for i in (0, 1):
                    value = _full_scalar(graph, node.args[i])
                    if value == 1.0:
                        result = args[1 - i]
                        counts['full_slot_mul_one_removed'] += 1
                        matched = True
                        break
                    if value == -1.0:
                        result = -args[1 - i]
                        counts['full_slot_mul_minus_one_to_negate'] += 1
                        matched = True
                        break
            if not matched:
                result = target.binary(node.op, args[0], args[1])
        else:
            raise ValueError('unsupported compact graph operation: ' + node.op)
        mapped[node.index] = result
    rewritten = [mapped[expr.index] for expr in outputs]
    stats = dict(enabled=True, scalar_units=bool(scalar_units), reciprocal_seed=bool(reciprocal_seed),
        rewrite_counts=dict(counts), source_reachable_nodes=len(graph.reachable(outputs)),
        result_reachable_nodes=len(target.reachable(rewritten)),
        metadata_only_rewrites=False, approximation_changed=False,
        ciphertext_bitwise_equivalent=False,
        note='same real-polynomial math; floating reassociation and CKKS rescale/Boot noise can differ')
    return target, rewritten, stats
