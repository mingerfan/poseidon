"""Count core nodes against the user-corrected R3 kernel coverage set."""

from collections import Counter

from .core_ops import CORE_OPS, KERNEL_OPS


def _pool_reason(node):
    if node.op == 'maxpool_indices':
        return '暂不支持 maxpool return_indices used by a consumer'
    if node.op != 'maxpool':
        return None
    a = node.attrs
    if a.get('ceil_mode', False):
        return '暂不支持 maxpool ceil_mode=True'
    if tuple(a.get('dilation', (1, 1))) != (1, 1):
        return '暂不支持 maxpool dilation != 1'
    k, s, p = tuple(a['kernel']), tuple(a['stride']), tuple(a['padding'])
    if k[0] != k[1] or k[0] not in (2, 3) or s[0] != s[1] or s[0] not in (2, k[0]) or p[0] != p[1] or p[0] not in (0, 1):
        return '暂不支持 this maxpool kernel/stride/padding'
    return None


def _supported(node):
    return _pool_reason(node) is None and node.op in KERNEL_OPS and not (node.op == 'act' and node.attrs['fn'] in
                                         ('rsqrt', 'sqrt', 'reciprocal', 'erf', 'pow'))


def report(graph) -> dict:
    counts = dict(Counter(node.op for node in graph.nodes))
    supported = sum(_supported(node) for node in graph.nodes)
    total = len(graph.nodes)
    mapped = sum(node.op in CORE_OPS for node in graph.nodes)
    original = getattr(graph, 'intake_stats', {})
    return dict(total=total, supported=supported, needs_composition=total-supported,
                by_op=counts, needs_composition_ops=sorted({node.op for node in graph.nodes if not _supported(node)}),
                original_nodes=original.get('original_nodes', total), folded_nodes=original.get('folded_nodes'), mapped=mapped,
                mapped_ratio=mapped/total if total else 1.0,
                lowering_ratio=supported/total if total else 1.0,
                unmapped={op: count for op, count in counts.items() if op not in CORE_OPS},
                original_unmapped=original.get('original_unmapped', {}), ratio_denominator='normalized_core_nodes')


def check(graph, *, slots=16384):
    """Report kernel capability at the check boundary, after semantic import.

    A graph has one statement per node; line numbers refer to that statement
    listing (not to an invented DSL program or to the original Python source).
    """
    diags = [dict(op_id=node.id, lineno=index+1, source=node.src,
                  reason=_pool_reason(node) or f'{node.op}: no lowering implementation', severity='error')
             for index, node in enumerate(graph.nodes)
             if not _supported(node)]
    for name in graph.inputs:
        shape = graph.tensors[name].shape
        if len(shape) == 4 and shape[0] == 1 and shape[2]*shape[3] > slots:
            index, node = next(((i, node) for i, node in enumerate(graph.nodes) if name in node.inputs),
                               (0, None))
            diags.append(dict(op_id=node.id if node is not None else name, lineno=index+1,
                              source=node.src if node is not None else name,
                              reason='spatial tiling not supported: k²·H·W > n', severity='error'))
    return dict(ok=not diags, diags=diags, line_kind='PT-Graph statement')
