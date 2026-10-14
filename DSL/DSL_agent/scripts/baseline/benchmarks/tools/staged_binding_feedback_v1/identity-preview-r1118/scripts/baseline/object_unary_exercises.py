"""Frozen v22 unary cases; trusted event observation and finite output probes."""
import ast
import copy
import hashlib
import json
from pathlib import Path

CATALOG=Path(__file__).with_name('object-unary-exercises-v1.json')
EXERCISES=json.loads(CATALOG.read_text())
TASK='hecate-function-synthesis-v22'


def exercise_spec(name):
    if type(name) is not str or name not in EXERCISES:raise ValueError('Unknown object unary exercise')
    return copy.deepcopy(EXERCISES[name])


def descriptor(name):
    exercise_spec(name)
    return dict(schema=2,id='exercise-'+name,input_shape=[4],constants=dict(weight=[.5],bias=[.375]),
        nodes=[dict(id='scaled',op='multiply',inputs=['x','weight']),
               dict(id='merged',op='add',inputs=['scaled','x']),
               dict(id='out',op='add',inputs=['merged','bias'])],output='out')


def validate_exercise_request(request):
    spec=request.get('construction_exercise')
    if (request.get('task')!=TASK or type(spec) is not dict or
        spec!=exercise_spec(spec.get('id')) or request.get('model')!=descriptor(spec['id'])):
        raise ValueError('Object unary exercise model/specification changed')


def span(node):
    return [node.lineno,node.col_offset,node.end_lineno,node.end_col_offset]


def features(facts):
    out={facts['syntax']+'.'+facts['operator']}
    for field in ('cipher_cells','plain_cells','boolean_cells'):
        if facts[field]>0:out.add(field)
    if facts['public_real_cells']>0:out.add('public_cells')
    if not facts['shape']:out.add('zero_dim')
    if facts['input_is_view']:out.add('input_view')
    return out


def perturb(source,event,*,fresh=False):
    tree=ast.parse(source)
    target=next(n for n in ast.walk(tree) if isinstance(n,(ast.Call,ast.UnaryOp)) and span(n)==event['span'])
    if fresh:
        operand=target.operand if isinstance(target,ast.UnaryOp) else target.args[0]
        if not isinstance(operand,ast.Name):raise ValueError('Freshness probe needs a local variable operand')
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and n.value is target and
                     len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id!=operand.id]
        if len(assignments)!=1:raise ValueError('Freshness requires unary result assigned to a distinct local')
        assignment=assignments[0]
        replacement=copy.deepcopy(assignment);replacement.value=copy.deepcopy(operand)
        statements=[replacement]
        if event['facts']['operator']=='USub':
            mutation=ast.AugAssign(target=ast.Name(id=operand.id,ctx=ast.Store()),op=ast.Mult(),value=ast.Constant(-1))
            statements.insert(0,ast.copy_location(mutation,assignment))
        class Change(ast.NodeTransformer):
            def visit_Assign(self,node):
                return statements if node is assignment else self.generic_visit(node)
    else:
        class Change(ast.NodeTransformer):
            def visit(self,node):
                if node is target:
                    result=(ast.UnaryOp(op=ast.USub(),operand=node) if event['facts']['operator']=='USub'
                            else ast.BinOp(left=node,op=ast.Add(),right=ast.Constant(1)))
                    return ast.copy_location(result,node)
                return super().visit(node)
    return ast.unparse(ast.fix_missing_locations(Change().visit(tree)))


def check_exercise(source,request):
    from function_construction import normalize
    from construction_exercises import fingerprint
    validate_exercise_request(request)
    events=[]
    def observe(value):
        if type(value) is tuple and value[0]=='object_unary':
            _,node,facts=value
            events.append(dict(kind='object_unary',span=span(node),facts=facts,features=sorted(features(facts))))
    expanded=normalize(source,request['public_constants'],request['layout']['output_ciphertexts'],
                       object_unary=True,observe=observe)
    required=request['construction_exercise']['required_features']
    missing=[f for f in required if not f.startswith('fresh.') and not any(f in e['features'] for e in events)]
    if missing:raise ValueError('Required typed unary operation not executed: '+', '.join(missing))
    baseline=fingerprint(expanded);influence={}
    for feature in required:
        fresh=feature.startswith('fresh.')
        candidates=[e for e in events if (e['facts']['operator']==feature[6:] if fresh else feature in e['features'])]
        for event in candidates[:8]:
            try:
                cell_kind=dict(cipher_cells='cipher',plain_cells='plain',
                               public_cells='number',boolean_cells='bool').get(feature)
                changed=source if cell_kind else perturb(source,event,fresh=fresh)
                actual=fingerprint(normalize(changed,request['public_constants'],
                    request['layout']['output_ciphertexts'],object_unary=True,
                    _unary_probe=dict(span=event['span'],kind=cell_kind) if cell_kind else None))
                if len(actual)==len(baseline) and max(abs(a-b) for a,b in zip(actual,baseline))>1e-8:
                    influence[feature]=dict(span=event['span'],status='output_changed',
                        probe=('typed_result_'+cell_kind) if cell_kind else
                              ('inplace_and_alias' if event['facts']['operator']=='USub' else 'alias_not_copy') if fresh else
                              'negate_result' if event['facts']['operator']=='USub' else 'result_plus_one')
                    break
            except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,OverflowError):
                continue
    missing=[f for f in required if f not in influence]
    if missing:raise ValueError('Required unary operation lacks output influence/freshness witness: '+', '.join(missing))
    return dict(schema=4,id=request['construction_exercise']['id'],required=required,events=events,influence=influence,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        interpretation='Typed executed unary events and finite result/freshness probes, not all-input proof.')
