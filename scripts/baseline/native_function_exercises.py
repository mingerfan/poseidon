"""Frozen native-call exercises: bounded AST observation plus finite sensitivity.

This is a coverage gate, not an FHE backend or an independent model reference.
Candidate Python is never executed. Real tracing and encrypted comparison remain
separate gates; empty returns can establish trace-structural coverage only.
"""
import ast
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from decorated_functions import _analyze, literal
from hecate_contract import rotation_literal
from seal_artifact_gate import require

CATALOG = Path(__file__).with_name('native-function-exercises-v1.json')
EXERCISES = json.loads(CATALOG.read_text())
TASK = 'hecate-native-function-synthesis-v2'
MAX_CALLS = 64
MAX_PROBE_STEPS = 131072


def exercise_spec(name):
    require(type(name) is str and name in EXERCISES, 'Unknown native function exercise')
    return copy.deepcopy(EXERCISES[name])


def descriptor(name):
    exercise_spec(name)
    if name == 'nf-two-inputs':
        return dict(schema=3, id='exercise-'+name,
            inputs=[dict(name='left', shape=[4]), dict(name='right', shape=[4])],
            constants={}, nodes=[dict(id='difference', op='subtract', inputs=['left','right'])],
            output='difference')
    return dict(schema=2, id='exercise-'+name, input_shape=[4],
        constants=dict(weight=[.5], bias=[.375]),
        nodes=[dict(id='scaled', op='multiply', inputs=['x','weight']),
               dict(id='merged', op='add', inputs=['scaled','x']),
               dict(id='out', op='add', inputs=['merged','bias'])], output='out')


def validate_exercise_request(request):
    spec = request.get('construction_exercise')
    require(request.get('task') == TASK and type(spec) is dict and
            spec == exercise_spec(spec.get('id')) and request.get('model') == descriptor(spec['id']),
            'Native function exercise model/specification changed')


def span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


@dataclass(frozen=True)
class Vector:
    data: tuple


def vector(value):
    values = value if type(value) is list else [value]
    return Vector(tuple(float(values[i % len(values)]) for i in range(4)))


def shifted(value):
    return Vector(tuple(a+b for a,b in zip(value.data, (.25,-.5,.75,1.))))


def samples(names):
    # Fixed public coverage probes, NOT the frozen encrypted test input arrays.
    patterns = ((0.,0.,0.,0.), (-.8,.25,.6,1.), (.3,-.5,.125,.7), (.9,.1,-.4,-.6))
    return [{name: vector([0.]*4 if name == 'zero_ct' else list(patterns[(group+i) % 4]))
             for i,name in enumerate(names)} for group in range(3)]


def _run(nodes, plan, constants, inputs, budget, probe=None):
    events = []

    def spend():
        budget[0] += 1
        require(budget[0] <= MAX_PROBE_STEPS, 'Native coverage probe resource limit')

    def invoke(name, args):
        spend()
        fn = nodes[name]
        values = {k: vector(v) for k,v in constants.items()}
        values.update(zip(plan['functions'][name]['parameters'], args))

        def expression(n):
            spend()
            number = literal(n)
            if number is not None:
                return vector(number)
            if type(n) is ast.Name:
                return values[n.id]
            if type(n) in (ast.List, ast.Tuple):
                result = [expression(v) for v in n.elts]
                return result if type(n) is ast.List else tuple(result)
            if type(n) is ast.Subscript:
                return expression(n.value)[rotation_literal(n.slice)]
            if type(n) is ast.UnaryOp:
                return Vector(tuple(-x for x in expression(n.operand).data))
            if type(n) is ast.BinOp:
                a, b = expression(n.left).data, expression(n.right).data
                op = {ast.Add: lambda x,y:x+y, ast.Sub: lambda x,y:x-y, ast.Mult: lambda x,y:x*y}[type(n.op)]
                return Vector(tuple(op(x,y) for x,y in zip(a,b)))
            if type(n) is ast.Call and type(n.func) is ast.Attribute:
                value = expression(n.func.value).data
                step = rotation_literal(n.args[0]) % 4
                return Vector(value[step:]+value[:step])
            require(type(n) is ast.Call and type(n.func) is ast.Name, 'Unexpected checked coverage AST')
            target = n.func.id
            arguments = [expression(a) for a in n.args]
            event_id = len(events)
            require(event_id < MAX_CALLS, 'Native coverage call count limit')
            callee = plan['functions'][target]
            result_type = callee['result']
            features = set()
            if result_type == 'c': features.add('call.scalar_cipher')
            if result_type == 'p': features.add('call.plain_return')
            if type(result_type) is tuple:
                if not result_type[1]: features.add('call.empty_return')
                if result_type[0] == 'tuple' and len(result_type[1]) >= 2 and all(k == 'c' for k in result_type[1]):
                    features.add('call.tuple_multi')
            if name != 'golden': features.add('call.nested')
            if nodes[target].lineno > fn.lineno: features.add('call.forward')
            if callee['kinds'].count('c') >= 2: features.add('call.two_cipher_arguments')
            if 'p' in callee['kinds']: features.add('call.public_argument')
            if not arguments: features.add('call.zero_arguments')
            returned = nodes[target].body[-1].value
            if (len(nodes[target].body) == 1 and type(returned) is ast.Name and result_type == 'c'
                    and returned.id in callee['parameters']
                    and callee['kinds'][callee['parameters'].index(returned.id)] == 'c'):
                features.add('call.identity')
            event = dict(id=event_id, caller=name, callee=target, span=span(n),
                         features=sorted(features), argument_kinds=list(callee['kinds']),
                         result_kind=result_type)
            events.append(event)
            if probe and probe[0] == 'argument' and probe[1] == event_id:
                arguments[probe[2]] = shifted(arguments[probe[2]])
            result = invoke(target, arguments)
            if probe and probe[0] == 'return' and probe[1] == event_id:
                if type(result) is Vector:
                    require(probe[2] == 0, 'Scalar result probe index')
                    result = shifted(result)
                else:
                    changed = list(result)
                    changed[probe[2]] = shifted(changed[probe[2]])
                    result = changed if type(result) is list else tuple(changed)
            return result

        def assign(target, value):
            if type(target) is ast.Name:
                values[target.id] = value
            else:
                for n,v in zip(target.elts,value): values[n.id] = v

        for statement in fn.body[:-1]:
            if type(statement) is ast.Assign:
                assign(statement.targets[0], expression(statement.value))
            else:
                expression(statement.value)
        return expression(fn.body[-1].value)

    value = invoke('golden', [inputs[n] for n in plan['functions']['golden']['parameters']])
    flat = [x for cell in (value if type(value) in (list,tuple) else [value]) for x in cell.data]
    require(all(math.isfinite(x) for x in flat), 'Nonfinite native coverage probe')
    repeated = Counter(e['callee'] for e in events)
    for event in events:
        if repeated[event['callee']] >= 2:
            event['features'] = sorted(event['features']+['call.repeated'])
    return flat, events


def fingerprint(source, constants, expected_outputs=1, *, input_names=('x',)):
    """Public finite probe for regression tests; never a model oracle or FHE run."""
    nodes, plan = _analyze(source, constants, expected_outputs, input_names)
    budget = [0]
    return [x for s in samples(input_names) for x in _run(nodes, plan, constants, s, budget)[0]]


def check_exercise(source, request):
    from candidate_contract import request_input_names
    validate_exercise_request(request)
    names = request_input_names(request)
    constants = request['public_constants']
    nodes, plan = _analyze(source, constants, request['layout']['output_ciphertexts'], names)
    budget = [0]
    probes = samples(names)
    baselines = [_run(nodes, plan, constants, s, budget) for s in probes]
    events = baselines[0][1]
    require(all(e == events for _,e in baselines), 'Native call schedule must be static')
    required = request['construction_exercise']['required_features']
    influence = {}
    cache = {}

    def affects(kind, event, index):
        key = (kind,event['id'],index)
        if key not in cache:
            changed = [_run(nodes, plan, constants, s, budget, key)[0] for s in probes]
            differences = [abs(a-b) for new,(old,_) in zip(changed,baselines) for a,b in zip(new,old)]
            cache[key] = max(differences, default=0.)
        return cache[key] > 1e-9

    for feature in required:
        candidates = [e for e in events if feature in e['features']]
        require(candidates, 'Required native call not reached: '+feature)
        witnesses = []
        for event in candidates:
            result = event['result_kind']
            count = len(result[1]) if type(result) is tuple else 1
            if feature == 'call.empty_return':
                witnesses.append(dict(event=event, status='trace_structural_only'))
                continue
            if not count or not any(affects('return',event,i) for i in range(count)):
                continue
            if feature == 'call.tuple_multi' and not all(affects('return',event,i) for i in range(count)):
                continue
            arguments = []
            if feature in ('call.public_argument', 'call.two_cipher_arguments'):
                kind = 'p' if feature == 'call.public_argument' else 'c'
                arguments = [i for i,k in enumerate(event['argument_kinds'])
                             if k == kind and affects('argument',event,i)]
                if len(arguments) < (1 if kind == 'p' else 2):
                    continue
            witnesses.append(dict(event=event, status='finite_output_changed',
                                  influential_argument_indices=arguments))
        if feature == 'call.repeated':
            counts = Counter(w['event']['callee'] for w in witnesses)
            witnesses = [w for w in witnesses if counts[w['event']['callee']] >= 2]
        require(witnesses, 'Required native call has no demonstrated output influence: '+feature)
        influence[feature] = witnesses
    return dict(schema=1, id=request['construction_exercise']['id'], required=required,
        influence=influence, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest(), probe_steps=budget[0],
        real_native_trace_verified=False, encrypted_execution_verified=False,
        interpretation='Bounded AST call reachability and finite return/argument perturbations; '
            'empty return is structural only. Not all-input equivalence or encrypted execution.')


def verify_trace_coverage(coverage, observed):
    """Bind static witnesses to calls that actually crossed the native boundary."""
    require(type(observed) is list and len(observed) <= 4096, 'Invalid native trace events')
    keys = set()
    for event in observed:
        require(type(event) is dict and set(event) == {'caller','callee','span'}, 'Invalid trace event shape')
        require(type(event['caller']) is str and type(event['callee']) is str and
                type(event['span']) is list and len(event['span']) == 4 and
                all(type(v) is int and v >= 0 for v in event['span']), 'Invalid trace event values')
        keys.add((event['caller'],event['callee'],tuple(event['span'])))
    for feature,witnesses in coverage['influence'].items():
        for witness in witnesses:
            event = witness['event']
            require((event['caller'],event['callee'],tuple(event['span'])) in keys,
                    'Native call witness absent from actual frontend trace: '+feature)
    return dict(verified=True, unique_native_call_sites=len(keys), checked_features=coverage['required'],
                interpretation='Actual createCall source sites; helper bodies trace once, then IR is cloned. '
                    'Not dynamic HEVM call counts or all-input proof.')
