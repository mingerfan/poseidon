"""Bounded array-construction coverage, separate from FHE and model reference.

Only validated native AST is interpreted. Fixed public probes establish finite
cell-result influence, not all-input equivalence or necessity of an operation.
Real frontend observations must separately match the operation/shape/source site.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import native_array_core as storage
from decorated_functions import _analyze, literal
from native_function_exercises import span, samples, vector, descriptor as native_descriptor
from seal_artifact_gate import require

TASK = 'hecate-native-function-synthesis-v4'
CATALOG = Path(__file__)
MAX_EVENTS = 256
MAX_STEPS = 524288
EXERCISES = {
    'na-reverse': ('matrix_reverse', ['return.matrix','index.reverse'],
        'Return a matrix object array from a helper; use a negative-step slice with at least two resulting cells.'),
    'na-transpose': ('matrix_transpose', ['return.matrix','transpose'],
        'Return a matrix object array from a helper and use .T or transpose() on at least two cells.'),
    'na-rank-four': ('rank_four', ['return.rank4','reshape.rank4','transpose','copy'],
        'Reshape storage to rank four, transpose it, return rank-four storage from a helper and use copy().'),
    'na-row-unpack': ('row_unpack', ['return.matrix','unpack.rows'],
        'Return a matrix from a helper and unpack its first axis into row arrays that contribute to the output.'),
    'na-mixed': ('mixed_plain', ['return.mixed'],
        'Return an array containing both Plain and ciphertext cells; at least one of each kind must affect output.'),
    'na-zero-item': ('zero_item', ['return.zero','item.zero'],
        'Return a zero-dimensional object array from a helper and extract its Expr using item().'),
    'na-zero-index': ('zero_index', ['return.zero','index.zero'],
        'Return a zero-dimensional object array from a helper and extract its Expr using [()].'),
    'na-zero-return': ('zero_return', ['return.zero','golden.zero'],
        'Return a zero-dimensional object array from a helper and directly return a zero-dimensional array from golden.'),
    'na-empty': ('empty_array', ['return.empty'],
        'Reachably call a helper returning an empty object array. This requirement is trace-structural only.'),
    'na-nested': ('nested_array', ['call.nested_array','index.reverse'],
        'Within a helper, call another helper returning an object array; use a negative-step slice on its result.'),
}


def exercise_spec(name):
    require(type(name) is str and name in EXERCISES, 'Unknown native array exercise')
    _, features, instruction = EXERCISES[name]
    return dict(id=name, required_features=list(features), instruction=instruction)


def descriptor(name):
    exercise_spec(name)
    result = native_descriptor('nf-scalar')
    result['id'] = 'exercise-'+name
    return result


def validate_exercise_request(request):
    spec = request.get('construction_exercise')
    require(request.get('task') == TASK and type(spec) is dict and
            spec == exercise_spec(spec.get('id')) and request.get('model') == descriptor(spec['id']),
            'Native array exercise model/specification changed')


def trace_record(op, node, caller, result, before=None):
    """Shared trusted observer protocol; no cell values or candidate execution."""
    import numpy as np
    return dict(op=op, caller=caller, span=span(node),
                input_shape=list(before.shape) if type(before) is np.ndarray else None,
                shape=list(result.shape) if type(result) is np.ndarray else None)


@dataclass(frozen=True)
class Cell:
    kind: str
    values: tuple


def flat(value):
    import numpy as np
    if type(value) is np.ndarray: return list(value.flat)
    if type(value) in (list,tuple): return [c for v in value for c in flat(v)]
    require(type(value) is Cell, 'Unexpected coverage value')
    return [value]


def features(op, node, caller, result, before):
    import numpy as np
    out = set()
    arr = type(result) is np.ndarray
    if op == 'return' and arr:
        if caller == 'golden':
            if result.ndim == 0: out.add('golden.zero')
        else:
            if result.ndim == 2 and result.size >= 2: out.add('return.matrix')
            if result.ndim == 4 and result.size >= 2: out.add('return.rank4')
            if result.ndim == 0: out.add('return.zero')
            if result.size == 0: out.add('return.empty')
            if {v.kind for v in result.flat} == {'c','p'}: out.add('return.mixed')
    if op == 'call' and caller != 'golden' and arr and result.size:
        out.add('call.nested_array')
    if op == 'index':
        if before.ndim == 0 and type(node.slice) is ast.Tuple and not node.slice.elts:
            out.add('index.zero')
        key = storage.index(node.slice)
        keys = key if type(key) is tuple else (key,)
        if arr and result.size >= 2 and any(type(k) is slice and k.step is not None and k.step < 0 for k in keys):
            out.add('index.reverse')
    if op == 'transpose' and arr and result.size >= 2: out.add('transpose')
    if op == 'reshape' and arr and result.ndim == 4 and result.size >= 2: out.add('reshape.rank4')
    if op == 'copy' and arr and result.size: out.add('copy')
    if op == 'item' and before.ndim == 0: out.add('item.zero')
    if op == 'unpack' and before.ndim >= 2 and before.size >= 2: out.add('unpack.rows')
    return sorted(out)


def run_probe(nodes, plan, constants, inputs, budget, intervention=None):
    import numpy as np
    events = []

    def spend():
        budget[0] += 1
        require(budget[0] <= MAX_STEPS, 'Native array probe resource bound')

    def observe(op, node, caller, result, before=None):
        spend()
        require(len(events) < MAX_EVENTS, 'Native array event bound')
        cells = flat(result)
        event = dict(id=len(events), trace=trace_record(op,node,caller,result,before),
                     features=features(op,node,caller,result,before), kinds=[c.kind for c in cells])
        events.append(event)
        if intervention is None or intervention[0] != event['id']: return result
        index = intervention[1]
        require(0 <= index < len(cells), 'Array probe cell bound')
        cell = cells[index]
        changed = Cell(cell.kind, tuple(a+b for a,b in zip(cell.values,(.25,-.5,.75,1.))))
        if type(result) is np.ndarray:
            result = result.copy(); result.flat[index] = changed
            return result
        require(type(result) is Cell and index == 0, 'Expected scalar array result')
        return changed

    def invoke(name, args):
        spend()
        fn = nodes[name]
        values = {n:Cell('p',vector(v).data) for n,v in constants.items()}
        values.update(zip(plan['functions'][name]['parameters'],args))

        def expression(n):
            spend()
            number = literal(n)
            if number is not None: return Cell('p',vector(number).data)
            if type(n) is ast.Name: return values[n.id]
            if type(n) in (ast.List,ast.Tuple):
                items = [expression(v) for v in n.elts]
                return items if type(n) is ast.List else tuple(items)
            if type(n) is ast.UnaryOp:
                v = expression(n.operand); return Cell(v.kind,tuple(-x for x in v.values))
            if type(n) is ast.BinOp:
                a,b = expression(n.left),expression(n.right)
                f = {ast.Add:lambda x,y:x+y, ast.Sub:lambda x,y:x-y, ast.Mult:lambda x,y:x*y}[type(n.op)]
                return Cell('c' if 'c' in (a.kind,b.kind) else 'p',tuple(f(x,y) for x,y in zip(a.values,b.values)))
            if type(n) is ast.Subscript:
                v = expression(n.value)
                if type(v) is np.ndarray:
                    return observe('index',n,name,v[storage.index(n.slice)],v)
                return v[storage.integer(n.slice)]
            if type(n) is ast.Attribute:
                v = expression(n.value); return observe('transpose',n,name,v.T,v)
            if storage.constructor(n):
                return observe('array',n,name,np.array(expression(n.args[0]),dtype=object))
            require(type(n) is ast.Call, 'Unexpected array coverage AST')
            if type(n.func) is ast.Name:
                result = invoke(n.func.id,[expression(v) for v in n.args])
                return observe('call',n,name,result) if type(result) is np.ndarray else result
            v = expression(n.func.value)
            if n.func.attr in storage.METHODS:
                return observe(n.func.attr,n,name,storage.apply_method(v,n),v)
            step = storage.integer(n.args[0]) % 4
            return Cell(v.kind,v.values[step:]+v.values[:step])

        def assign(target, value):
            if type(target) is ast.Name: values[target.id] = value
            else:
                if type(value) is np.ndarray: value = observe('unpack',target,name,value,value)
                for n,v in zip(target.elts,value): values[n.id] = v

        for statement in fn.body[:-1]:
            if type(statement) is ast.Assign: assign(statement.targets[0],expression(statement.value))
            else: expression(statement.value)
        returned = expression(fn.body[-1].value)
        if type(returned) is np.ndarray: returned = observe('return',fn.body[-1],name,returned)
        return returned

    result = invoke('golden',[Cell('c',inputs[n].data) for n in plan['functions']['golden']['parameters']])
    output = [x for cell in flat(result) for x in cell.values]
    require(all(math.isfinite(x) for x in output), 'Nonfinite array coverage probe')
    return output, events


def check_exercise(source, request):
    from candidate_contract import request_input_names
    validate_exercise_request(request)
    names = request_input_names(request)
    constants = request['public_constants']
    nodes,plan = _analyze(source,constants,request['layout']['output_ciphertexts'],names,arrays=True)
    probes = samples(names); budget = [0]
    baseline = [run_probe(nodes,plan,constants,s,budget) for s in probes]
    events = baseline[0][1]
    require(all(e == events for _,e in baseline), 'Array event schedule changed across public probes')
    influence = {}; cache = {}
    def affects(event,index):
        key = (event['id'],index)
        if key not in cache:
            altered = [run_probe(nodes,plan,constants,s,budget,key)[0] for s in probes]
            cache[key] = max((abs(a-b) for new,(old,_) in zip(altered,baseline) for a,b in zip(new,old)),default=0.) > 1e-9
        return cache[key]
    required = request['construction_exercise']['required_features']
    for feature in required:
        witnesses = []
        for event in events:
            if feature not in event['features']: continue
            if feature == 'return.empty':
                witnesses.append(dict(event=event,status='trace_structural_only')); continue
            indices = [i for i in range(len(event['kinds'])) if affects(event,i)]
            if not indices: continue
            if feature == 'return.mixed' and {event['kinds'][i] for i in indices} != {'c','p'}: continue
            witnesses.append(dict(event=event,status='finite_cell_result_influence',indices=indices))
        require(witnesses, 'Required native array feature has no output influence: '+feature)
        influence[feature] = witnesses
    return dict(schema=1,id=request['construction_exercise']['id'],required=required,influence=influence,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest(),probe_steps=budget[0],
                real_native_trace_verified=False,encrypted_execution_verified=False,
                interpretation='Finite cell-result influence; empty return is structural. Not all-input proof, '
                    'not proof that an operation cannot be simplified, and not FHE execution.')


def verify_trace_coverage(coverage, observed):
    require(type(observed) is list and len(observed) <= 4096, 'Invalid native array trace size')
    for event in observed:
        require(type(event) is dict and set(event) == {'op','caller','span','input_shape','shape'}, 'Array trace fields')
        require(type(event['op']) is str and type(event['caller']) is str and type(event['span']) is list
                and len(event['span']) == 4 and all(type(v) is int and v >= 0 for v in event['span']), 'Array trace site')
        for key in ('input_shape','shape'):
            require(event[key] is None or type(event[key]) is list, 'Array trace shape type')
            if event[key] is not None: storage.dimensions(event[key])
    for feature,witnesses in coverage['influence'].items():
        for witness in witnesses:
            require(witness['event']['trace'] in observed, 'Array witness absent from real frontend: '+feature)
    return dict(verified=True,checked_features=coverage['required'],observed_operations=len(observed),
                interpretation='Actual frontend operation/source/shape match; not dynamic HEVM instruction counts.')
