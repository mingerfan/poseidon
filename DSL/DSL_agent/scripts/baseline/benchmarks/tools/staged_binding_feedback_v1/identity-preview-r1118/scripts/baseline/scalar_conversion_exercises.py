"""Frozen v21 public scalar extraction exercises; typed observations, no candidate exec."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

CATALOG=Path(__file__).with_name('scalar-conversion-exercises-v1.json')
EXERCISES=json.loads(CATALOG.read_text())
TASK='hecate-function-synthesis-v21'


def exercise_spec(name):
    if type(name) is not str or name not in EXERCISES:
        raise ValueError('Unknown scalar conversion exercise')
    return copy.deepcopy(EXERCISES[name])


def descriptor(name):
    spec=exercise_spec(name)
    nodes=([dict(id='square',op='multiply',inputs=['x','x'])] if spec['family']=='quadratic' else [])
    nodes += [dict(id='scaled',op='multiply',inputs=['square' if nodes else 'x','weight']),
              dict(id='merged',op='add',inputs=['scaled','x']),
              dict(id='out',op='add',inputs=['merged','bias'])]
    return dict(schema=2,id='exercise-'+name,input_shape=[4],
                constants=dict(weight=[.5],bias=[.375]),nodes=nodes,output='out')


def validate_exercise_request(request):
    spec=request.get('construction_exercise')
    if (request.get('task')!=TASK or type(spec) is not dict or
        spec!=exercise_spec(spec.get('id')) or request.get('model')!=descriptor(spec['id'])):
        raise ValueError('Scalar extraction exercise model/specification changed')


def span(node):
    return [node.lineno,node.col_offset,node.end_lineno,node.end_col_offset]


def features(kind,facts):
    if kind=='object_numeric_constructor':return {'object.named_numeric_constructor'}
    if kind=='object_inplace':
        return {'object.overlap_mult'} if facts.get('overlapping') and facts.get('operator')=='Mult' else set()
    storage=facts['storage']
    if kind=='scalar_cast':
        root='cast.'+facts['name']+'.'+storage
        out=set()
        if storage in ('default','bool','str'):out.add(root)
        if facts['name']=='int' and facts['arity']==2:out.add('cast.int.base')
        if storage in ('numeric','object'):
            out.add(root+('.zero' if not facts['shape'] else '.legacy'))
            if facts['negative_fraction']:out.add(root+'.negative_fraction')
        return out
    if kind=='scalar_item':
        out={'item.'+storage+'.'+facts['index_form']}
        if facts['index_form']=='tuple' and len(facts['shape'])!=2:out.clear()
        if facts['index_form']=='flat' and facts['negative_index'] and facts['shape'] and math.prod(facts['shape'])>1:
            out.add('item.'+storage+'.flat_negative')
        if storage=='object' and facts['result_kind']=='cipher':out.add('item.object.cipher')
        return out
    return set()


def check_exercise(source,request):
    from function_construction import normalize
    from construction_exercises import fingerprint
    validate_exercise_request(request)
    events=[]
    def observe(value):
        if type(value) is tuple and value[0] in ('scalar_cast','scalar_item','object_inplace','object_numeric_constructor'):
            kind,node,facts=value
            facts=dict(facts)
            if kind=='object_inplace':facts['operator']=type(node.op).__name__
            events.append(dict(kind=kind,span=span(node),facts=facts,features=sorted(features(kind,facts))))
    expanded=normalize(source,request['public_constants'],request['layout']['output_ciphertexts'],
                       scalar_conversion=True,observe=observe)
    required=request['construction_exercise']['required_features']
    missing=[f for f in required if not any(f in e['features'] for e in events)]
    if missing:raise ValueError('Required scalar construction not executed: '+', '.join(missing))
    baseline=fingerprint(expanded)
    influence={}
    for feature in required:
        for event in [e for e in events if feature in e['features']][:8]:
            class Perturb(ast.NodeTransformer):
                def visit(self,node):
                    if isinstance(node,(ast.Call,ast.AugAssign)) and span(node)==event['span']:
                        if isinstance(node,ast.AugAssign):
                            node.op=ast.Add()
                            return node
                        return ast.copy_location(ast.BinOp(left=node,op=ast.Add(),right=ast.Constant(1)),node)
                    return super().visit(node)
            try:
                altered=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
                probe=fingerprint(normalize(altered,request['public_constants'],
                    request['layout']['output_ciphertexts'],scalar_conversion=True))
                if len(probe)==len(baseline) and max(abs(a-b) for a,b in zip(probe,baseline))>1e-8:
                    influence[feature]=dict(span=event['span'],status='output_changed',
                        probe='add_not_multiply' if event['kind']=='object_inplace' else 'extracted_value_plus_one')
                    break
            except (ValueError,TypeError,IndexError,KeyError,ZeroDivisionError,OverflowError):
                continue
    missing=[f for f in required if f not in influence]
    if missing:raise ValueError('Required extraction has no demonstrated output influence: '+', '.join(missing))
    return dict(schema=3,id=request['construction_exercise']['id'],required=required,events=events,
        influence=influence,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        interpretation='Typed executed extraction plus finite output perturbation; no all-input proof.')
