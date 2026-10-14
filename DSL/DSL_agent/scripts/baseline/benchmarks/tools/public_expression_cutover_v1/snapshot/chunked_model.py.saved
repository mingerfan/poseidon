"""Schema-4 logical graph -> schema-3 physical graph. Not Agent generation.

Input expansion only: 5..16 elements, rank1..4; vector arithmetic, flatten,
Linear, square/power2/4, fan-out and residual. Output remains 1..4 elements.
No spatial operators, implicit batching, cross-chunk rotate or guessed semantics.
"""
import copy
import math

from chunked_input_abi import binding
from seal_artifact_gate import require

OPS = frozenset(('add', 'subtract', 'multiply', 'negate', 'square', 'power', 'flatten', 'linear'))


def lower(data):
    from model_graph import NAME, CASE_ID, array_shape, validate_graph
    require(type(data) is dict and set(data) == {'schema','id','input_shape','constants','nodes','output'}
            and type(data['schema']) is int and data['schema'] == 4, 'Invalid schema-4 model fields')
    require(type(data['id']) is str and CASE_ID.fullmatch(data['id']), 'Invalid schema-4 id')
    plan = binding(data['input_shape'])
    constants, nodes = data['constants'], data['nodes']
    require(type(constants) is dict and len(constants) <= 32, 'Public constant count limit')
    shapes = {}
    for name, value in constants.items():
        require(type(name) is str and NAME.fullmatch(name) and name != 'x', 'Invalid public name')
        shapes[name] = array_shape(value)
    require(type(nodes) is list and 1 <= len(nodes) <= 64, 'Graph node count limit')
    result = dict(schema=3, id=data['id'], inputs=[dict(name=c['name'],shape=[4]) for c in plan['chunks']],
                  constants={}, nodes=[], output=None)
    # (logical shape, ordered physical value IDs, still block-packed)
    values = {'x': (tuple(data['input_shape']), tuple(c['name'] for c in plan['chunks']), True)}
    declared = {'x', *constants}
    def public(value):
        name = f'p{len(result["constants"])}'
        result['constants'][name] = copy.deepcopy(value)
        return name
    def emit(op, args, **extra):
        name = f'v{len(result["nodes"])}'
        result['nodes'].append(dict(id=name, op=op, inputs=list(args), **extra))
        return name
    for node in nodes:
        require(type(node) is dict and type(node.get('op')) is str and node['op'] in OPS,
                'Unsupported schema-4 operator')
        op = node['op']
        extra = {'weight','bias'} if op == 'linear' else {'exponent'} if op == 'power' else set()
        require(set(node) == {'id','op','inputs'} | extra, 'Unexpected operator fields')
        name, args = node['id'], node['inputs']
        require(type(name) is str and NAME.fullmatch(name) and name not in declared, 'Invalid graph value')
        require(type(args) is list and len(args) == (2 if op in ('add','subtract','multiply') else 1)
                and all(type(a) is str and a in declared for a in args), 'Invalid or forward graph inputs')
        require(args[0] in values, 'Left operand must be ciphertext')
        shape, blocks, packed = values[args[0]]
        if op == 'flatten':
            value = ((math.prod(shape),), blocks, packed)
        elif op == 'linear':
            w, b = node['weight'], node['bias']
            require(type(w) is str and w in constants and len(shape) == 1 and
                    len(shapes[w]) == 2 and shapes[w][1] == shape[0] and 1 <= shapes[w][0] <= 8,
                    'Invalid schema-4 Linear shape')
            width = shapes[w][0]
            require(b is None or type(b) is str and b in constants and shapes[b] == (width,), 'Invalid Linear bias')
            if packed:
                partials = []
                for i, block in enumerate(blocks):
                    rows = [row[4*i:4*i+4] + [0.0]*max(0, 4*i+4-shape[0]) for row in constants[w]]
                    partials.append(emit('linear', [block], weight=public(rows), bias=None))
                total = partials[0]
                for term in partials[1:]:
                    total = emit('add', [total, term])
                if b is not None:
                    total = emit('add', [total, public(constants[b])])
            else:
                total = emit('linear', blocks, weight=public(constants[w]),
                             bias=None if b is None else public(constants[b]))
            value = ((width,), (total,), False)
        else:
            require(len(shape) == 1, 'Flatten multidimensional input before vector arithmetic')
            params = {}
            if op == 'power':
                require(type(node['exponent']) is int and node['exponent'] in (2,4), 'Only powers 2 and 4')
                params['exponent'] = node['exponent']
            right = None
            if len(args) == 2:
                if args[1] in values:
                    rs, right, rp = values[args[1]]
                    require(rs == shape and rp == packed, 'Cipher shape/packing mismatch')
                else:
                    rshape, raw = shapes[args[1]], constants[args[1]]
                    require(rshape in ((), (1,), shape), 'Unsupported public broadcast')
                    if packed and rshape == shape:
                        right = tuple(public(raw[4*i:4*i+4] + [0.0]*max(0,4*i+4-shape[0]))
                                      for i in range(len(blocks)))
                    else:
                        key = public(raw)
                        right = (key,)*len(blocks)
            mapped = tuple(emit(op, [block] if right is None else [block,right[i]], **params)
                           for i,block in enumerate(blocks))
            value = (shape, mapped, packed)
        declared.add(name)
        values[name] = value
    require(type(data['output']) is str and data['output'] in values, 'Invalid graph output')
    outshape, outputs, packed = values[data['output']]
    require(not packed and len(outshape) == 1 and 1 <= outshape[0] <= 4,
            'Schema-4 output must be reduced to 1..4 scalar-neuron elements')
    result['output'] = outputs[0]
    validate_graph(result)  # Existing bounds on physical graph/constants apply too.
    return result, plan


def validate(data):
    from model_graph import validate_graph
    graph, plan = lower(data)
    checked = validate_graph(graph)
    return dict(checked, schema=4, operators=sorted({n['op'] for n in data['nodes']}),
                logical_input_shape=plan['logical_shape'], physical_inputs=len(plan['chunks']))


def test_inputs(shape):
    import numpy as np
    size = binding(shape)['logical_elements']
    # Position-distinct signed sample and random sample expose swapped/dropped chunks.
    flat = np.stack([np.zeros(size),
                     np.array([(-1 if i%2 else 1)*(i+1)/(size+1) for i in range(size)]),
                     np.random.default_rng(42).uniform(-1,1,size),
                     np.array([-1 if i%2 else 1 for i in range(size)])]).astype(np.float64)
    return flat.reshape(4,*shape)
