"""Conservative ideal-message packing facts, not exact-zero HE/noise claims."""
import numpy as np


def ideal_head_gaps_are_zero(expr, config):
    """Prove the two-head gap precondition from public Expr arithmetic.

    A raw ciphertext input has UNKNOWN layout. Public zero masks can establish
    the fact; slotwise operations/identity Boot hints preserve it. This proof
    concerns the real arithmetic DSL only: CKKS encoding/boot/keyswitch noise in
    a nominally zero lane is generally nonzero. It never decrypts or changes IR.
    """
    graph = expr.backend
    if (graph.slots != config.slots or config.tile != 2 * config.tokens or
            config.tokens != 2 * config.head_dim):
        raise ValueError('gap proof requires the cyclic two-head layout')
    memo = {}

    def visit(node):
        if node.backend is not graph:
            raise ValueError('packing proof encountered a foreign graph')
        if node.index in memo:
            return memo[node.index]
        if node.op == 'plain':
            values = graph.constants[node.data]
            # Short payload tails are ZERO. The two head groups have the same
            # T-periodic live/gap pattern, even across physical token rows.
            gaps = np.arange(len(values)) % config.tokens >= config.head_dim
            known = bool(np.all(values[gaps] == 0))
        elif node.op in ('boot', 'neg'):
            known = visit(node.args[0])
        elif node.op == 'rotate':
            known = node.data % config.tokens == 0 and visit(node.args[0])
        elif node.op == 'mul':
            known = visit(node.args[0]) or visit(node.args[1])
        elif node.op in ('add', 'sub'):
            known = visit(node.args[0]) and visit(node.args[1])
        elif node.op == 'compact_av':
            known = visit(node.args[1])  # weighted V mixes tokens, not features
        elif node.op == 'compact_av_partial':
            known = False  # defined cross-head/row partial sums, NOT zero gaps
        else:
            known = False
        memo[node.index] = known
        return known

    return visit(expr)


def validate_partial_av_consumers(graph, outputs):
    """Reject every reachable uncleared partial-AV fanout, without changing IR.

    This is an ideal-message gap-leak guard, NOT a full semantic/noise proof.
    Only identity Boot hints may precede a real multiplication by a separately
    proven gap-zero operand. Dead uses do not participate in export liveness.
    """
    from types import SimpleNamespace

    outputs = list(outputs)
    if any(node.backend is not graph for node in outputs):
        raise ValueError('outputs must belong to the source graph')
    nodes = graph.reachable(outputs)
    partials = [node for node in nodes if node.op == 'compact_av_partial']
    if not partials:
        return
    consumers = {}
    for node in nodes:
        for arg in node.args:
            consumers.setdefault(arg.index, []).append(node)
    finals = {node.index for node in outputs}
    for partial in partials:
        spec = graph.av_specs[partial.data]
        config = SimpleNamespace(slots=graph.slots, **{key: spec[key]
            for key in ('tokens', 'tile', 'head_dim')})
        if not ideal_head_gaps_are_zero(partial.args[1], config):
            raise ValueError('partial AV requires public proof of ideal zero value head gaps')
        dependent = {partial.index}
        for node in nodes:
            if any(arg.index in dependent for arg in node.args):
                dependent.add(node.index)

        todo, seen = [partial], set()
        while todo:
            source = todo.pop()
            if source.index in seen:
                continue
            seen.add(source.index)
            if source.index in finals:
                raise ValueError('partial AV reaches a final output before gap clearing')
            for consumer in consumers.get(source.index, ()):
                if consumer.op == 'boot':
                    todo.append(consumer)
                    continue
                if consumer.op == 'mul':
                    other = consumer.args[1] if consumer.args[0] is source else consumer.args[0]
                    if (other.index not in dependent and
                            ideal_head_gaps_are_zero(other, config)):
                        continue  # All downstream consumers now see ideal zero gaps.
                raise ValueError('partial AV consumer ' + consumer.op +
                                 ' is unsafe before independently proven gap clearing')
