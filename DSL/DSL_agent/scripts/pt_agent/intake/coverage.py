"""Count core nodes against the user-corrected R3 kernel coverage set."""

from collections import Counter

from .core_ops import KERNEL_OPS


def report(graph) -> dict:
    counts = dict(Counter(node.op for node in graph.nodes))
    supported = sum(node.op in KERNEL_OPS for node in graph.nodes)
    return dict(total=len(graph.nodes), supported=supported, needs_composition=len(graph.nodes)-supported,
                by_op=counts, needs_composition_ops=sorted(set(counts)-KERNEL_OPS))
