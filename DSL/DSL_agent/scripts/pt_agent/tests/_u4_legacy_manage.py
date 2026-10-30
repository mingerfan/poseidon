"""Pre-U4 §7.9 manage(), retained only as an independent test oracle."""

from collections import Counter, deque
from dataclasses import replace
import math

from pt_agent.manage.greedy import infer, schedule, placement_view
from pt_agent.manage.ir import Function, Node, Type
from pt_agent.placement import depth_dp


def legacy_manage(recorded, profile, *, name='golden', stage_callback=None):
    """Transforms/GreedyBootstrapPlacement.cpp:33–271, Common.cpp:44–140."""
    profile = dict(profile)
    waterline, factor = profile['waterline'], profile['rescalingFactor']
    last = profile['bootstrapLevelUpperBound']-profile['bootstrapLevelLowerBound']
    if not 0 < waterline <= factor or last < 2:
        raise ValueError('greedy bootstrap requires 0 < waterline <= factor and two usable levels')
    inputs = [Type(True, waterline, 0) for _ in recorded._inputs]
    outputs = [v.nid for v in recorded._outputs]
    nodes = [replace(n, args=list(n.args), attrs=dict(n.attrs)) for n in recorded.nodes]
    # RemoveBootstrap.cpp:26–29; aliases do not create managed operations.
    aliases, kept = {}, []
    for n in nodes:
        n.args = [aliases.get(a, a) for a in n.args]
        if n.kind == 'bootstrap':
            aliases[n.id] = n.args[0]
        else:
            kept.append(n)
    outputs = [aliases.get(a, a) for a in outputs]
    nodes = kept
    ordered, depths, users, maximum = schedule(nodes, inputs, outputs)
    table = {key.removeprefix('earth.'): val for key, val in profile['latencyTable'].items()}
    placement = depth_dp(placement_view(nodes, inputs, outputs), last_boot_level=last,
                         boot_in_level=profile['bootstrapLevelLowerBound'],
                         boot_out_level=profile['bootstrapLevelUpperBound'], latency_table=table)
    cuts = placement.cuts
    types = {-i-1: t for i, t in enumerate(inputs)}
    by_id = {n.id: n for n in nodes}
    result, current = [], {}
    uses = Counter(a for n in nodes for a in n.args)
    uses.update(outputs)
    live = {-i-1: None for i in range(len(inputs)) if uses[-i-1]}
    next_id = max(by_id, default=-1)+1

    def insert(kind, args, owner, attrs=None):
        # GreedyBootstrapPlacement.cpp:174–193: builder insertion points.
        nonlocal next_id
        attrs = attrs or {}
        n = Node(next_id, kind, list(args), infer(kind, [types[a] for a in args], attrs, profile),
                 attrs, op_id=owner.op_id, lineno=owner.lineno)
        next_id += 1
        result.append(n)
        by_id[n.id], types[n.id] = n, n.type
        return n.id

    def normalize(value, owner):
        # GreedyBootstrapPlacement.cpp:174–180.
        scale = types[value].scale
        if scale == waterline:
            return value
        return insert('rescale', [insert('upscale', [value], owner,
                      {'upFactor': factor+waterline-scale})], owner)

    def refresh(logical, owner):
        # GreedyBootstrapPlacement.cpp:185–193.
        value = current.get(logical, logical)
        if types[value].scale != factor:
            value = insert('upscale', [value], owner, {'upFactor': factor-types[value].scale})
        current[logical] = normalize(insert('bootstrap', [value], owner, {'targetLevel': 0}), owner)

    boundaries, nxt = [], 0
    return_owner = Node(-1, 'return', outputs, Type(True, waterline, 0),
                        op_id=nodes[-1].op_id if nodes else None, lineno=nodes[-1].lineno if nodes else 0)
    for n in [*ordered, return_owner]:
        logical = list(n.args)
        depth = depths[n.id] if n is not return_owner else maximum+1
        if nxt < len(cuts) and depth > cuts[nxt]:
            # GreedyBootstrapPlacement.cpp:198–215: original ordered live set.
            end = cuts[nxt+1] if nxt+1 < len(cuts) else maximum
            safe = last-(end-cuts[nxt])
            snapshot = []
            for a in live:
                value = current.get(a, a)
                t = types[value]
                snapshot.append((value, t.scale, t.level, n.op_id, n.lineno))
                if t.level > last or t.scale != waterline:
                    raise ValueError('live ciphertext exceeds the greedy bootstrap budget')
                if t.level > safe:
                    refresh(a, n)
            boundaries.append(snapshot)
            nxt += 1
        n.args = [current.get(a, a) for a in logical]
        if n.kind in ('add', 'mul'):
            # EarthOps.td:358–375 and456–468: processOperandsEVA.
            a, b = n.args
            if n.kind == 'mul' and any(types[v].cipher and types[v].level >= last for v in n.args):
                raise ValueError('multiplication exceeds the bootstrap placement budget')
            if not (types[a].cipher and types[b].cipher):
                plain, cipher = (a, b) if not types[a].cipher else (b, a)
                t = types[cipher]
                by_id[plain].type = types[plain] = Type(False, waterline if n.kind == 'mul' else t.scale, t.level)
            else:
                if n.kind == 'add' and types[a].scale != types[b].scale:
                    index = 0 if types[a].scale < types[b].scale else 1
                    lo, hi = n.args[index], n.args[1-index]
                    n.args[index] = insert('upscale', [lo], n, {'upFactor': types[hi].scale-types[lo].scale})
                a, b = n.args
                if types[a].level != types[b].level:
                    index = 0 if types[a].level < types[b].level else 1
                    lo, hi = n.args[index], n.args[1-index]
                    n.args[index] = insert('modswitch', [lo], n, {'downFactor': types[hi].level-types[lo].level})
        if n is return_owner:
            outputs = n.args
            break
        if n.kind != 'constant':
            n.type = infer(n.kind, [types[a] for a in n.args], n.attrs, profile)
        types[n.id] = n.type
        result.append(n)
        if n.kind == 'mul':
            current[n.id] = normalize(n.id, n)
        for a in logical:
            uses[a] -= 1
            if uses[a] == 0:
                live.pop(a, None)
                current.pop(a, None)
        if uses[n.id] and n.type.cipher:
            live[n.id] = None

    def refine(value, owner, output_val=0):
        # Common.cpp:8–41. C++ integer division truncates toward zero.
        t = types[value]
        down = math.trunc((profile['bootstrapLevelUpperBound']*factor-
                           (t.level*factor+t.scale+output_val))/factor)
        return insert('modswitch', [value], owner, {'downFactor': down})

    # Common.cpp:44–109: insert return modswitches, then bootstrap input
    # modswitches at their own insertion points, before EarlyModswitch.
    outputs = [refine(a, return_owner, profile['output_val']) for a in outputs]
    for n in list(result):
        if n.kind == 'bootstrap':
            tail = len(result)
            n.args[0] = refine(n.args[0], n)
            created = result.pop()
            result.insert(result.index(n), created)
    function = Function(name, inputs, result, outputs, [types[a] for a in outputs],
                        dict(arg_scale=[t.scale for t in inputs],
                             init_level=profile['bootstrapLevelUpperBound'],
                             res_scale=[types[a].scale for a in outputs]), profile, list(cuts), boundaries)
    if stage_callback is not None:
        stage_callback('a', function)
    legacy_early_modswitch(function)
    if stage_callback is not None:
        stage_callback('b', function)
    return function



def compare_functions(left, right):
    """Compare every Function and Node field, including payloads and ownership."""
    from dataclasses import fields
    import numpy as np
    for field in fields(Function):
        if field.name != 'nodes' and getattr(left,field.name) != getattr(right,field.name):
            return dict(equal=False,field=field.name,
                expected=repr(getattr(left,field.name))[:300],actual=repr(getattr(right,field.name))[:300])
    if len(left.nodes)!=len(right.nodes):
        return dict(equal=False,field='nodes.length',expected=len(left.nodes),actual=len(right.nodes))
    for index,(a,b) in enumerate(zip(left.nodes,right.nodes)):
        for field in fields(Node):
            x,y=getattr(a,field.name),getattr(b,field.name)
            same = (x is None and y is None or x is not None and y is not None and np.array_equal(x,y)) if field.name=='data' else x==y
            if not same:
                return dict(equal=False,node=index,field=field.name,expected=repr(x)[:300],actual=repr(y)[:300])
    return dict(equal=True,nodes=len(left.nodes),boots=sum(n.kind=='bootstrap' for n in left.nodes),
                cuts=len(left.cuts),outputs=len(left.outputs))


def legacy_early_modswitch(function):
    """Transforms/EarlyModswitch.cpp:45–99, including new reverse-iterator nodes."""
    nodes = {n.id: n for n in function.nodes}
    types = function.types()
    users = {a: [] for a in types}
    for n in function.nodes:
        for i, a in enumerate(n.args):
            users[a].append((n.id, i))
    for i, a in enumerate(function.outputs):
        users[a].append((None, i))
    before = {n.id: [] for n in function.nodes}
    todo = deque(reversed(function.nodes))
    next_id = max(nodes, default=-1)+1
    while todo:
        n = todo.popleft()
        if n.kind == 'bootstrap' or not users[n.id]:
            continue
        down = min((nodes[u].attrs['downFactor'] if u is not None and nodes[u].kind == 'modswitch'
                    else 0) for u, _ in users[n.id])
        if not down:
            continue
        if n.kind == 'modswitch':
            n.attrs['downFactor'] += down
        else:
            new = []
            for i, a in enumerate(n.args):
                m = Node(next_id, 'modswitch', [a], replace(types[a], level=types[a].level+down),
                         {'downFactor': down}, op_id=n.op_id, lineno=n.lineno)
                next_id += 1
                nodes[m.id], types[m.id], before[m.id] = m, m.type, []
                users[a].remove((n.id, i))
                users[a].append((m.id, 0))
                users[m.id] = [(n.id, i)]
                n.args[i] = m.id
                new.append(m)
            before[n.id].extend(new)
            todo.extendleft(new)
        n.type = types[n.id] = replace(n.type, level=n.type.level+down)
        for u in {u for u, _ in users[n.id]}:
            nodes[u].attrs['downFactor'] -= down

    def expanded(n):
        for m in before[n.id]:
            yield from expanded(m)
        yield n
    function.nodes = [m for n in function.nodes for m in expanded(n)]
