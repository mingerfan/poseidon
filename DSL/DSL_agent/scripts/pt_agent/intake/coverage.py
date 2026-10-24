"""Count core nodes against the user-corrected R3 kernel coverage set."""

from collections import Counter

from .core_ops import CORE_OPS, KERNEL_OPS


def _supported(node):
    return node.op in KERNEL_OPS and not (node.op == 'act' and node.attrs['fn'] in
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


def check(graph):
    """Report kernel capability at the check boundary, after semantic import.

    A graph has one statement per node; line numbers refer to that statement
    listing (not to an invented DSL program or to the original Python source).
    """
    diags = [dict(op_id=node.id, lineno=index+1, source=node.src,
                  reason=f'{node.op}: no lowering implementation', severity='error')
             for index, node in enumerate(graph.nodes)
             if not _supported(node)]
    return dict(ok=not diags, diags=diags, line_kind='PT-Graph statement')
