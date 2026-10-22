"""The specified depth-dp boundaries and full-level bootstrap simulation."""

from dataclasses import dataclass

from .backends.dag import DagBackend


@dataclass
class Placement:
    cuts: list[int]
    barriers: int
    boots: int
    latency_us: float
    boot_nodes: list[int]
    node_latency: dict[int, float]
    max_depth: int


class PlacementError(Exception):
    pass


def depth_dp(dag: DagBackend, *, last_boot_level: int, boot_in_level: int, boot_out_level: int,
             latency_table: dict) -> Placement:
    if 'boot' in dag.kind:
        raise PlacementError('depth-dp model requires a program without explicit refresh')
    n = len(dag.kind)
    ret = n
    users = [[] for _ in range(n)]
    for nid, args in enumerate(dag.args):
        for arg in args:
            users[arg].append(nid)
    for value in dag._outputs or []:
        users[value.nid].append(ret)
    depth = []
    for kind, args in zip(dag.kind, dag.args):
        depth.append(max((depth[arg] for arg in args), default=0)+(kind in ('mul_cc', 'mul_cp')))
    maxd = max(depth, default=0)
    depth.append(maxd+1)
    for nid in range(n-1, -1, -1):
        if dag.kind[nid] in ('rot', 'neg') and users[nid]:
            depth[nid] = min(depth[user] for user in users[nid])
    crossing = [0]*(maxd+2)
    for nid, uses in enumerate(users):
        if uses:
            start = 0 if dag.kind[nid] == 'input' else depth[nid]
            end = max(start, max(depth[user] for user in uses))
            crossing[start] += 1
            crossing[end] -= 1
    for index in range(1, maxd+1):
        crossing[index] += crossing[index-1]
    budget = last_boot_level
    cost = [float('inf')]*(maxd+1)
    cost[0] = 0
    prev = [-1]*(maxd+1)
    for end in range(1, maxd+1):
        if end <= budget:
            cost[end] = 0
        for start in range(end-1, max(0, end-(budget-1))-1, -1):
            candidate = cost[start]+crossing[start]
            if candidate < cost[end]:
                cost[end], prev[end] = candidate, start
    cuts = []
    end = maxd
    while prev[end] >= 0:
        end = prev[end]
        cuts.append(end)
    cuts.reverse()
    levels = [0]*n
    remaining = [len(uses) for uses in users]
    live = {nid: None for nid, kind in enumerate(dag.kind) if kind == 'input' and users[nid]}
    nxt = barriers = 0
    boot_nodes, node_latency = [], {}
    for nid in sorted(range(n+1), key=depth.__getitem__):
        if nid != ret and dag.kind[nid] == 'input':
            continue
        if nxt < len(cuts) and depth[nid] > cuts[nxt]:
            barriers += 1
            end = cuts[nxt+1] if nxt+1 < len(cuts) else maxd
            safe = budget-(end-cuts[nxt])
            for value in live:
                if levels[value] > budget:
                    raise PlacementError(f'live value exceeds the bootstrap budget at node {value}')
                if levels[value] > safe:
                    levels[value] = 0
                    boot_nodes.append(value)
                    node_latency[value] = node_latency.get(value, 0)+latency_table['bootstrap_single'][boot_in_level]
            nxt += 1
        if nid == ret:
            continue
        args, kind = dag.args[nid], dag.kind[nid]
        level = max((levels[arg] for arg in args), default=0)
        table_level = boot_out_level-level
        if kind in ('mul_cc', 'mul_cp'):
            if level >= budget:
                raise PlacementError(f'multiplication exceeds the bootstrap budget at node {nid}')
            key = 'mul_double' if kind == 'mul_cc' else 'mul_single'
            latency = latency_table[key][table_level]+latency_table['rescale_single'][table_level]
            levels[nid] = level+1
        else:
            key = ('rotate_single' if kind == 'rot' else 'negate_single' if kind == 'neg'
                   else 'add_double' if len(args) == 2 else 'add_single')
            latency = latency_table[key][table_level]
            levels[nid] = level
        node_latency[nid] = node_latency.get(nid, 0)+latency
        for arg in args:
            remaining[arg] -= 1
            if remaining[arg] == 0:
                live.pop(arg, None)
        if remaining[nid] > 0:
            live[nid] = None
    return Placement(cuts, barriers, len(boot_nodes), sum(node_latency.values()), boot_nodes, node_latency, maxd)
