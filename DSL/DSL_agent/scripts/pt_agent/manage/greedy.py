"""Port of GreedyBootstrapPlacement(depth-dp), Common and EarlyModswitch.

Source references below are relative to DaCapo's lib/Dialect/Earth (or include/).
No canonicalization or CSE is performed here; MLIR runs those passes after emission.
"""

from collections import Counter, deque
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np

from ..backends.base import Backend, _Cipher, _Plain
from ..placement import depth_dp
from .ir import Function, Node, Type


class RecordedExpr(_Cipher):
    def __init__(self, backend, nid, *, is_input=False):
        super().__init__(backend, is_input=is_input)
        self.nid = nid


class RecordingBackend(Backend):
    """tools/frontend.cpp:75–119,165–249: record trusted arithmetic, not Python source."""
    def __init__(self, slots, *, locations=None, retain_constants=True, owner_resolver=None):
        super().__init__(slots)
        self.nodes = []
        self.locations = locations or {}
        self._keep_const_data = retain_constants
        self.owner_resolver = owner_resolver

    def _make_input(self, index):
        return RecordedExpr(self, -index-1, is_input=True)

    def _node(self, kind, args=(), *, data=None, attrs=None):
        op = self.locations.get(self.op_context)
        if op is None and self.owner_resolver is not None:
            op = self.owner_resolver()
        node = Node(len(self.nodes), kind, list(args), Type(kind != 'constant', 0, 0),
                    attrs or {}, data, getattr(op, 'id', self.op_context), getattr(op, 'lineno', 0))
        self.nodes.append(node)
        return node.id

    def _value(self, value):
        # frontend.cpp:131 and PrivatizeConstant.cpp:24–32: each constant use
        # has a private encoding; arrays themselves remain shared and immutable.
        if isinstance(value, _Plain):
            return self._node('constant', data=value.data)
        return value.nid

    def _apply(self, kind, left, right=None, *, step=None):
        a = self._value(left)
        b = self._value(right) if right is not None else None
        if kind == 'sub':
            # frontend.cpp:209–225: public subtraction constructs a negated
            # constant; ciphertext subtraction constructs negate followed by add.
            if isinstance(right, _Plain):
                if self.nodes[b].data is not None:
                    self.nodes[b].data = -self.nodes[b].data
            else:
                b = self._node('negate', [b])
            kind = 'add'
        aliases = dict(mul_cc='mul', mul_cp='mul', neg='negate', rot='rotate', boot='bootstrap')
        nid = self._node(aliases.get(kind, kind), [a] if b is None else [a, b],
                         attrs={'offset': [step]} if kind == 'rot' else {})
        return RecordedExpr(self, nid)


def infer(kind, args, attrs, profile):
    """IR/EarthDialect.cpp:182–335; IR/EarthOps.td:165–233 (unary traits)."""
    factor = profile['rescalingFactor']
    a = args[0] if args else None
    if kind == 'constant':
        # ConstantOp::isCompatibleReturnTypes accepts any Plain encoding;
        # processOperandsEVA retunes its scale/level (EarthOps.td:114–157).
        return None
    if kind in ('negate', 'rotate'):
        return a
    if kind in ('add', 'sub'):
        b = args[1]
        if (a.scale, a.level) != (b.scale, b.level):
            raise ValueError('add/sub operands require equal scale and level')
        return Type(True, a.scale, a.level)
    if kind == 'mul':
        b = args[1]
        if a.level != b.level:
            raise ValueError('mul operands require equal level')
        # EarthDialect.cpp:325–328: this is the explicit source capacity rule.
        if a.level*factor+a.scale > profile['bootstrapLevelUpperBound']*factor:
            raise ValueError('mul lhs exceeds the source capacity rule')
        return Type(True, a.scale+b.scale, a.level)
    if kind == 'rescale':
        return replace(a, scale=a.scale-factor, level=a.level+1)
    if kind == 'modswitch':
        down = attrs['downFactor']
        if down < 0:
            raise ValueError('negative downFactor')
        return replace(a, level=a.level+down)
    if kind == 'upscale':
        up = attrs['upFactor']
        if up < 0:
            raise ValueError('negative upFactor')
        return replace(a, scale=a.scale+up)
    if kind == 'bootstrap':
        if a.level > profile['bootstrapLevelUpperBound']-profile['bootstrapLevelLowerBound']:
            raise ValueError('bootstrap input exceeds lastBootLevel')
        return replace(a, level=attrs.get('targetLevel', 0))
    raise ValueError('unsupported Earth operation '+kind)


def schedule(nodes, inputs, outputs):
    """Transforms/GreedyBootstrapPlacement.cpp:48–79,101–129,161–167.

    Depth postponement and stable order are not the dynamic program. The sole
    cut-selection implementation remains placement.depth_dp.
    """
    users = {-i-1: [] for i in range(len(inputs))}
    depths = {-i-1: 0 for i in range(len(inputs))}
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        users.setdefault(n.id, [])
        for a in n.args:
            users[a].append(n.id)
        depths[n.id] = max((depths[a] for a in n.args), default=0)+(n.kind == 'mul')
    maximum = max(depths.values(), default=0)
    depths[None] = maximum+1
    for a in outputs:
        users[a].append(None)
    for n in reversed(nodes):
        if n.kind in ('rotate', 'negate') and users[n.id]:
            depths[n.id] = min(depths[u] for u in users[n.id])
    return sorted(nodes, key=lambda n: depths[n.id]), depths, users, maximum


def placement_view(nodes, inputs, outputs):
    """Transforms/GreedyBootstrapPlacement.cpp:115–160, using placement.depth_dp."""
    # The existing cost DAG excludes plaintext nodes. Preserve its enumeration,
    # including the input prefix, when passing the recorded computation to it.
    ids = {-i-1: i for i in range(len(inputs))}
    kinds, args = ['input']*len(inputs), [() for _ in inputs]
    for n in nodes:
        if n.kind == 'constant':
            continue
        ids[n.id] = len(kinds)
        cipher_args = tuple(ids[a] for a in n.args if a in ids)
        kind = ('mul_cc' if len(cipher_args) == 2 else 'mul_cp') if n.kind == 'mul' else {
            'rotate': 'rot', 'negate': 'neg'}.get(n.kind, n.kind)
        kinds.append(kind)
        args.append(cipher_args)
    return SimpleNamespace(kind=kinds, args=args, _outputs=[SimpleNamespace(nid=ids[a]) for a in outputs])


def manage(recorded, profile, *, name='golden', stage_callback=None):
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
    early_modswitch(function)
    if stage_callback is not None:
        stage_callback('b', function)
    return function


def early_modswitch(function):
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
