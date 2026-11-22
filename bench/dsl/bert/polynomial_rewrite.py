"""Opt-in reassociation of exact existing exp4/GELU4/rsqrt3 polynomials.

No coefficient approximation, degree reduction, HE metadata change, or mask
folding. Recognizers require the original power DAG and full-slot coefficients.
Exclusive internal uses avoid retaining old powers beside new Horner products.
"""
from collections import Counter
import math
import numpy as np

try:
    from .cerium_lowering import CompactGraph
    from .equivalent_rewrite import _full_scalar
except ImportError:
    from cerium_lowering import CompactGraph
    from equivalent_rewrite import _full_scalar


def _scaled(graph, node, coefficient):
    if node.op != 'mul':
        return None
    for side in (0, 1):
        if _full_scalar(graph, node.args[side]) == coefficient:
            return node.args[1 - side]
    return None


def _square(node, source):
    return node.op == 'mul' and all(arg.index == source.index for arg in node.args)


def _product(node, left, right):
    return node.op == 'mul' and sorted(arg.index for arg in node.args) == sorted((left.index, right.index))


def reassociate_polynomials(graph, outputs, *, exponential=True, gelu=True, rsqrt=True):
    """Clone graph, retain all public arrays, and rewrite only exact patterns."""
    outputs = list(outputs)
    if any(node.backend is not graph for node in outputs):
        raise ValueError('outputs must belong to the source graph')
    reachable = graph.reachable(outputs)
    references = Counter(arg.index for node in reachable for arg in node.args)
    references.update(node.index for node in outputs)
    target = CompactGraph(graph.slots, prepared_qk=getattr(graph, 'use_prepared_qk', False),
                          prepared_av=getattr(graph, 'use_prepared_av', False),
                          qk_pack_inputs=getattr(graph, 'qk_pack_inputs', False),
                          av_pack_inputs=getattr(graph, 'av_pack_inputs', False))
    target.qk_specs = {name: dict(spec) for name, spec in getattr(graph, 'qk_specs', {}).items()}
    target.av_specs = {name: dict(spec) for name, spec in getattr(graph, 'av_specs', {}).items()}
    target.matrices = {name: {key: value.copy() if isinstance(value, np.ndarray) else value
                              for key, value in matrix.items()}
                       for name, matrix in getattr(graph, 'matrices', {}).items()}
    mapped, counts = {}, Counter()

    def scalar(value):
        return target.plain(np.full(graph.slots, value, dtype=np.float64))

    def match_exp(node):
        if node.op != 'add' or node.args[0].op != 'add':
            return None
        fourth = _scaled(graph, node.args[1], 1 / 24)
        cube = _scaled(graph, node.args[0].args[1], 1 / 6)
        lower = node.args[0].args[0]
        if fourth is None or cube is None or lower.op != 'add' or lower.args[0].op != 'add':
            return None
        square = _scaled(graph, lower.args[1], .5)
        linear = lower.args[0]
        if square is None or _full_scalar(graph, linear.args[1]) != 1:
            return None
        x = linear.args[0]
        if not (_square(square, x) and _product(cube, square, x) and _square(fourth, square)):
            return None
        if references[node.args[1].index] != 1 or references[node.args[0].args[1].index] != 1:
            return None
        return x, square, cube, fourth

    def match_gelu(node):
        if node.op != 'sub' or node.args[0].op != 'add':
            return None
        coefficient = 1 / math.sqrt(2 * math.pi)
        half, quadratic = node.args[0].args
        x, square = _scaled(graph, half, .5), _scaled(graph, quadratic, coefficient)
        fourth = _scaled(graph, node.args[1], coefficient / 6)
        if x is None or square is None or fourth is None:
            return None
        if not _square(square, x) or not _square(fourth, square):
            return None
        if references[fourth.index] != 1 or references[node.args[1].index] != 1:
            return None
        return x, square, coefficient

    def match_rsqrt(node):
        if node.op != 'sub' or node.args[0].op != 'add':
            return None
        lower, quadratic = node.args[0].args
        cube = _scaled(graph, node.args[1], 5 / 16)
        square = _scaled(graph, quadratic, 3 / 8)
        if cube is None or square is None or lower.op != 'add':
            return None
        x = _scaled(graph, lower.args[0], -.5)
        if x is None or _full_scalar(graph, lower.args[1]) != 1:
            return None
        if not (_square(square, x) and _product(cube, square, x)):
            return None
        if (references[node.args[0].index] != 1 or references[square.index] != 2 or
                references[cube.index] != 1 or references[quadratic.index] != 1 or
                references[node.args[1].index] != 1):
            return None
        return x

    for node in reachable:
        args = [mapped[arg.index] for arg in node.args]
        exp_match = match_exp(node) if exponential else None
        gelu_match = match_gelu(node) if gelu else None
        rsqrt_match = match_rsqrt(node) if rsqrt else None
        if exp_match is not None:
            x, square, cube, fourth = (mapped[n.index] for n in exp_match)
            result = (x + scalar(1) + square * scalar(.5)) + (
                cube * scalar(1 / 6) + fourth * scalar(1 / 24))
            counts['exp4_balanced'] += 1
        elif gelu_match is not None:
            x, square, coefficient = gelu_match
            x, square = mapped[x.index], mapped[square.index]
            result = x * scalar(.5) + square * (square * scalar(-coefficient / 6) + scalar(coefficient))
            counts['gelu4_factored'] += 1
        elif rsqrt_match is not None:
            x = mapped[rsqrt_match.index]
            result = ((x * scalar(-5 / 16) + scalar(3 / 8)) * x + scalar(-.5)) * x + scalar(1)
            counts['rsqrt3_horner'] += 1
        elif node.op == 'plain':
            result = target.plain(graph.constants[node.data])
        elif node.op == 'input':
            result = target.input(node.data)
        elif node.op == 'boot':
            result = target.bootstrap(args[0], label=node.data)
        elif node.op == 'rotate':
            result = args[0].rotate(node.data)
        elif node.op == 'neg':
            result = -args[0]
        elif node.op in ('compact_linear', 'compact_qk', 'compact_av', 'compact_av_partial'):
            result = target.node(node.op, tuple(args), node.data)
        elif node.op in ('add', 'sub', 'mul'):
            result = target.binary(node.op, args[0], args[1])
        else:
            raise ValueError('unsupported polynomial rewrite operation: ' + node.op)
        mapped[node.index] = result
    result = [mapped[node.index] for node in outputs]
    return target, result, dict(enabled=True, exponential=bool(exponential), gelu=bool(gelu), rsqrt=bool(rsqrt),
        rewrite_counts=dict(counts), source_reachable_nodes=len(reachable),
        result_reachable_nodes=len(target.reachable(result)), approximation_changed=False,
        metadata_only_rewrites=False, ciphertext_bitwise_equivalent=False,
        note='same polynomial coefficients and degrees; floating association and CKKS noise ordering differ')
