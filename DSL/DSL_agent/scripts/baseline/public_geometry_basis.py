"""Finite public permutation bases; no model/candidate/reference execution."""
import math
from benchmark_graph import require


def permutation_layers(permutation):
    """Beneš adjacent-pair routing, with immutable source-index semantics.

    The result maps output i to input permutation[i]. A layer lists both ends
    of each conditional XOR-distance swap. Identity layers remain until merge.
    """
    p = list(permutation)
    n = len(p)
    require(n > 0 and n & (n - 1) == 0 and sorted(p) == list(range(n)),
            'Public basis permutation must be a power-of-two bijection')

    def route(values, wires):
        count = len(values)
        if count == 1:
            return []
        distance = wires[1] - wires[0]
        if count == 2:
            return [(distance, set(wires) if values == [1, 0] else set())]
        inverse = [0] * count
        for dst, src in enumerate(values):
            inverse[src] = dst
        colors = [-1] * count
        for seed in range(count):
            if colors[seed] >= 0:
                continue
            colors[seed] = 0
            pending = [seed]
            while pending:
                src = pending.pop()
                for neighbor in (src ^ 1, values[inverse[src] ^ 1]):
                    color = 1 - colors[src]
                    if colors[neighbor] < 0:
                        colors[neighbor] = color
                        pending.append(neighbor)
                    else:
                        require(colors[neighbor] == color, 'Public basis alternating-cycle invariant')
        first = set()
        last = set()
        upper = []
        lower = []
        for pair in range(count // 2):
            if colors[2 * pair] == 1:
                first.update(wires[2 * pair:2 * pair + 2])
            even_src, odd_src = values[2 * pair:2 * pair + 2]
            if colors[even_src] == 1:
                last.update(wires[2 * pair:2 * pair + 2])
                even_src, odd_src = odd_src, even_src
            upper.append(even_src // 2)
            lower.append(odd_src // 2)
        top = route(upper, wires[::2])
        bottom = route(lower, wires[1::2])
        require(len(top) == len(bottom), 'Public basis recursive stage count')
        middle = []
        for (step, a), (other, b) in zip(top, bottom):
            require(step == other, 'Public basis recursive distance')
            middle.append((step, a | b))
        return [(distance, first), *middle, (distance, last)]

    return [(step, sorted(swap)) for step, swap in route(p, list(range(n))) if swap]


def complete_partial(pairs, period):
    """Translate source/destination offsets before completing a partial bijection."""
    require(type(period) is int and period > 0 and period & (period - 1) == 0,
            'Public basis period')
    require(pairs and all(type(d) is int and type(s) is int and
                         0 <= d < period and 0 <= s < period for d, s in pairs),
            'Public basis partial map bounds')
    require(len({d for d, _ in pairs}) == len(pairs) and
            len({s for _, s in pairs}) == len(pairs), 'Public basis partial map is injective')
    output_shift = min(d for d, _ in pairs)
    input_shift = min(s for _, s in pairs)
    selected = {d - output_shift: s - input_shift for d, s in pairs}
    used_sources = set(selected.values())
    remaining = iter(i for i in range(period) if i not in used_sources)
    permutation = [selected[i] if i in selected else next(remaining) for i in range(period)]
    return permutation, sorted(selected), input_shift, output_shift
