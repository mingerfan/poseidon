"""S1 §6 A/B/C checks in Earth consumed levels; no SMU grouping."""

from .greedy import infer, timed_stage


class ManagementError(ValueError):
    def __init__(self, index, node, message):
        self.index, self.op_id, self.lineno = index, node.op_id, node.lineno
        super().__init__(f'E_MANAGEMENT operation {index}, PT-DSL op {node.op_id}, line {node.lineno}: {message}')


def check(function):
    with timed_stage("check", nodes=len(function.nodes)):
        return _check(function)


def _check(function):
    """EarthDialect.cpp:203–335; GreedyBootstrapPlacement.cpp:40–45,83–90,
    198–225; Common.cpp:44–140. These are independent A/B/C checks, not a
    replay of placement. No physical levelLower/UpperBound is used.
    """
    from .ir import Node, Type
    p = function.profile
    waterline, factor = p['waterline'], p['rescalingFactor']
    last = p['bootstrapLevelUpperBound']-p['bootstrapLevelLowerBound']
    fallback = function.nodes[0] if function.nodes else Node(0, 'boundary', [], Type(True, 0, 0))

    def fail(index, node, message):
        raise ManagementError(index, node, message)

    types = {}
    for i, t in enumerate(function.inputs):
        if not t.cipher or not 0 <= t.level <= last:
            fail(-i-1, fallback, 'B: function ciphertext input level exceeds lastBootLevel')
        if t.scale != waterline or t.level != 0:
            fail(-i-1, fallback, 'C: input differs from refineInputValues')
        types[-i-1] = t
    for index, n in enumerate(function.nodes):
        if n.id in types or any(a not in types for a in n.args):
            fail(index, n, 'A: duplicate definition or operand used before definition')
        if n.type.scale < 0 or n.type.level < 0:
            fail(index, n, 'A: scale/level must be nonnegative')
        args = [types[a] for a in n.args]
        if n.kind == 'mul' and any(t.cipher and t.level >= last for t in args):
            fail(index, n, 'B: multiplication ciphertext operand level must be < lastBootLevel')
        if n.kind == 'bootstrap' and args[0].scale != factor:
            fail(index, n, 'B: bootstrap input scale must equal rescalingFactor')
        try:
            expected = infer(n.kind, args, n.attrs, p)
        except (ValueError, KeyError, IndexError) as exc:
            fail(index, n, 'A: '+str(exc))
        if n.kind == 'constant':
            # EarthOps.td:122–128 permits locally retuned Plain types.
            if n.type.cipher or n.args or n.data is None:
                fail(index, n, 'A: constant must have a Plain encoding and payload')
        elif n.type != expected:
            fail(index, n, f'A: inferred type {expected} differs from {n.type}')
        if n.type.cipher and not 0 <= n.type.level <= last:
            fail(index, n, 'B: ciphertext level outside [0, lastBootLevel]')
        types[n.id] = n.type
    for snapshot in function.boundaries:
        for value, scale, level, op_id, lineno in snapshot:
            if level > last or scale != waterline:
                owner = Node(value, 'cut', [], Type(True, scale, level), op_id=op_id, lineno=lineno)
                fail(value, owner, 'B: cut frontier exceeds lastBootLevel or differs from waterline')
    if (any(a not in types for a in function.outputs) or
            function.result_types != [types[a] for a in function.outputs]):
        fail(len(function.nodes), fallback, 'C: function result types differ from return values')
    attrs = dict(arg_scale=[t.scale for t in function.inputs],
                 init_level=p['bootstrapLevelUpperBound'],
                 res_scale=[types[a].scale for a in function.outputs])
    if any(function.attrs.get(k) != v for k, v in attrs.items()):
        fail(len(function.nodes), fallback, 'C: arg_scale/init_level/res_scale differ from refineReturnValues')
    # Common.cpp:36–40,73: refining a final output again must add zero levels.
    factor = p['rescalingFactor']
    for a in function.outputs:
        t = types[a]
        numerator = p['bootstrapLevelUpperBound']*factor-(t.level*factor+t.scale+p['output_val'])
        down = abs(numerator)//factor * (-1 if numerator < 0 else 1)
        if down != 0:
            fail(len(function.nodes), fallback, 'C: output level differs from refineReturnValues')
    return True
