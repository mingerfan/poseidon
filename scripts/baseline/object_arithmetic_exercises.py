"""Frozen request-v20 exercises. Trusted AST observation and finite probes only."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

CATALOG=Path(__file__).with_name('object-arithmetic-exercises-v1.json')
EXERCISES=json.loads(CATALOG.read_text())
TASK='hecate-function-synthesis-v20'


def exercise_spec(name):
    if type(name) is not str or name not in EXERCISES:
        raise ValueError('Unknown object arithmetic exercise')
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
        raise ValueError('Object arithmetic exercise model/specification changed')


def span(node):
    return [node.lineno,node.col_offset,node.end_lineno,node.end_col_offset]


def features(kind,node,facts):
    out={('binary.' if kind=='object_binary' else 'inplace.')+type(node.op).__name__}
    if facts['overlapping']:out.add('overlap')
    if facts['empty_left']:out.add('empty_left')
    if facts['cipher_pair']:out.add('cipher_pair')
    if kind=='object_binary':
        result=facts['result_shape']
        if not result:out.add('zero_dim')
        if len(result)>=2 and math.prod(result)>max(math.prod(facts['left_shape']),math.prod(facts['right_shape'])):
            out.add('rank_broadcast')
    return out


def check_exercise(source,request):
    from function_construction import normalize
    from construction_exercises import fingerprint
    validate_exercise_request(request)
    events=[]
    def observe(value):
        if type(value) is tuple and value[0] in ('object_binary','object_inplace'):
            kind,node,facts=value
            events.append(dict(kind=kind,span=span(node),facts=facts,
                               features=sorted(features(kind,node,facts))))
    expanded=normalize(source,request['public_constants'],request['layout']['output_ciphertexts'],
                       object_arithmetic=True,observe=observe)
    spec=request['construction_exercise']
    required=spec['required_features']
    missing=[feature for feature in required if not any(feature in e['features'] for e in events)]
    if missing:raise ValueError('Required object arithmetic not executed: '+', '.join(missing))
    baseline=fingerprint(expanded)
    influence={}
    for feature in required:
        candidates=[e for e in events if feature in e['features']][:4]
        for event in candidates:
            tree=ast.parse(source)
            target=next(n for n in ast.walk(tree) if isinstance(n,(ast.BinOp,ast.AugAssign)) and span(n)==event['span'])
            alias=spec['id'].endswith('-alias')
            if alias and not isinstance(target,ast.AugAssign):continue
            if alias and not isinstance(target.target,ast.Name):continue
            if feature=='empty_left' or spec['id']=='oa-empty-subtract':
                replacements=[None]  # Replace Empty construction by public zero.
            elif alias:
                replacements=[ast.Assign(targets=[copy.deepcopy(target.target)],
                    value=ast.BinOp(left=ast.Name(id=target.target.id,ctx=ast.Load()),
                                    op=copy.deepcopy(target.op),right=copy.deepcopy(target.value)))]
            else:
                changed=copy.deepcopy(target)
                changed.op=ast.Sub() if isinstance(target.op,ast.Add) else ast.Add()
                replacements=[changed]
            for replacement in replacements:
                class Perturb(ast.NodeTransformer):
                    def visit(self,node):
                        if replacement is None and isinstance(node,ast.Call) and (
                            isinstance(node.func,ast.Name) and node.func.id=='Empty' or
                            isinstance(node.func,ast.Attribute) and node.func.attr=='Empty'):
                            return ast.copy_location(ast.Constant(0),node)
                        if replacement is not None and isinstance(node,type(target)) and span(node)==event['span']:
                            return ast.copy_location(copy.deepcopy(replacement),node)
                        return super().visit(node)
                try:
                    altered=ast.unparse(ast.fix_missing_locations(Perturb().visit(ast.parse(source))))
                    probe=fingerprint(normalize(altered,request['public_constants'],
                        request['layout']['output_ciphertexts'],object_arithmetic=True))
                    if len(probe)==len(baseline) and max(abs(a-b) for a,b in zip(probe,baseline))>1e-8:
                        influence[feature]=dict(span=event['span'],status='output_changed',
                            probe='empty_as_zero' if replacement is None else 'rebind_not_inplace' if alias else 'change_operator')
                        break
                except (ValueError,TypeError,IndexError,KeyError,ZeroDivisionError):
                    continue
            if feature in influence:break
    missing=[f for f in required if f not in influence]
    if missing:raise ValueError('Required object arithmetic has no demonstrated output influence: '+', '.join(missing))
    return dict(schema=2,id=spec['id'],required=required,events=events,influence=influence,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        interpretation='Actual typed array events and finite output probes; not all-input proof.')

