"""Frozen starred-call requirements with bounded finite argument influence.

Only checked AST is interpreted. This is neither an FHE backend nor the model
reference. Native frontend call observations and encrypted comparison are
independent gates. Empty expansion establishes reachable structure only.
"""
import ast
import copy
import hashlib
import math
from pathlib import Path

import native_array_core as storage
from decorated_functions import _analyze,literal
from native_array_exercises import Cell,flat
from native_function_exercises import span,samples,vector,descriptor as native_descriptor
from seal_artifact_gate import require

TASK='hecate-native-function-synthesis-v8'
CATALOG=Path(__file__)
MAX_EVENTS=64
MAX_STEPS=524288
EXERCISES={
    'ns-list':('list',['star.list'],'Expand a nonempty list into a decorated helper.'),
    'ns-tuple':('tuple',['star.tuple'],'Expand a nonempty tuple into a decorated helper.'),
    'ns-array':('array',['star.array'],'Expand a one-dimensional object array into a decorated helper.'),
    'ns-reverse':('reverse',['star.reverse'],'Expand a reversed object-array slice with at least two cells.'),
    'ns-matrix-flatten':('matrix_flatten',['star.flatten_matrix'],
        'Explicitly flatten matrix object storage and expand the resulting Expr cells into a helper.'),
    'ns-nested':('nested',['star.call_result'],'Expand the object-array result of a call to another helper.'),
    'ns-multiple':('multiple',['star.multiple','star.mixed'],
        'Use at least two nonempty starred segments mixed with at least one ordinary argument in one helper call.'),
    'ns-empty':('empty',['star.empty'],
        'Reachably call a zero-argument helper using empty storage expansion. This requirement is structural only.'),
}


def exercise_spec(name):
    require(type(name) is str and name in EXERCISES,'Unknown native starred exercise')
    _,features,instruction=EXERCISES[name]
    return dict(id=name,required_features=list(features),instruction=instruction+
        ' For nonempty calls every positional argument must affect the final output; no dead or cancelled padding.')


def descriptor(name):
    exercise_spec(name)
    model=native_descriptor('nf-scalar');model['id']='exercise-'+name
    return model


def validate_exercise_request(request):
    spec=request.get('construction_exercise')
    require(request.get('task')==TASK and type(spec) is dict and
            spec==exercise_spec(spec.get('id')) and request.get('model')==descriptor(spec['id']),
            'Native starred exercise model/specification changed')


def segment(node,value,origin_shapes):
    import numpy as np
    if type(node) is not ast.Starred:
        return dict(kind='ordinary',shape=None,count=1,form=None,origin_shape=None)
    kind='array' if type(value) is np.ndarray else 'list' if type(value) is list else 'tuple'
    form=None;origin=None;n=node.value
    if kind=='array' and type(n) is ast.Subscript:
        key=storage.index(n.slice);parts=key if type(key) is tuple else (key,)
        if any(type(k) is slice and k.step is not None and k.step<0 for k in parts):form='reverse'
    if kind=='array' and type(n) is ast.Call:
        if type(n.func) is ast.Name:form='call_result'
        elif n.func.attr=='flatten':
            form='flatten';origin=origin_shapes.get(id(n))
    return dict(kind=kind,shape=list(value.shape) if kind=='array' else None,
                count=len(value),form=form,origin_shape=origin)


def trace_record(node,caller,segments):
    return dict(caller=caller,callee=node.func.id,span=span(node),segments=segments)


def event_features(event):
    segments=event['trace']['segments'];stars=[s for s in segments if s['kind']!='ordinary']
    nonempty=[s for s in stars if s['count']]
    found={'star.'+s['kind'] for s in nonempty}
    if len(nonempty)>=2:found.add('star.multiple')
    if nonempty and any(s['kind']=='ordinary' for s in segments):found.add('star.mixed')
    if stars and not event['argument_kinds']:found.add('star.empty')
    for s in nonempty:
        if s['form']=='reverse' and s['count']>=2:found.add('star.reverse')
        if s['form']=='call_result':found.add('star.call_result')
        if s['form']=='flatten' and s['origin_shape'] is not None and len(s['origin_shape'])==2 and s['count']>=2:
            found.add('star.flatten_matrix')
    return sorted(found)


def run_probe(nodes,plan,constants,inputs,budget,intervention=None):
    import numpy as np
    events=[]
    def spend():
        budget[0]+=1;require(budget[0]<=MAX_STEPS,'Native starred probe resource bound')
    def invoke(name,args):
        spend();fn=nodes[name];origins={}
        values={n:Cell('p',vector(v).data) for n,v in constants.items()}
        values.update(zip(plan['functions'][name]['parameters'],args))
        def expression(n):
            spend();number=literal(n)
            if number is not None:return Cell('p',vector(number).data)
            if type(n) is ast.Name:return values[n.id]
            if type(n) in (ast.List,ast.Tuple):
                items=[expression(v) for v in n.elts]
                return items if type(n) is ast.List else tuple(items)
            if type(n) is ast.UnaryOp:
                v=expression(n.operand);return Cell(v.kind,tuple(-x for x in v.values))
            if type(n) is ast.BinOp:
                a,b=expression(n.left),expression(n.right)
                op={ast.Add:lambda x,y:x+y,ast.Sub:lambda x,y:x-y,ast.Mult:lambda x,y:x*y}[type(n.op)]
                return Cell('c',tuple(op(x,y) for x,y in zip(a.values,b.values)))
            if type(n) is ast.Subscript:
                v=expression(n.value)
                return v[storage.index(n.slice) if type(v) is np.ndarray else storage.integer(n.slice)]
            if type(n) is ast.Attribute:return expression(n.value).T
            if storage.constructor(n):return np.array(expression(n.args[0]),dtype=object)
            require(type(n) is ast.Call,'Unexpected checked starred AST')
            if type(n.func) is ast.Name:
                arguments=[];segments=[]
                starred=any(type(a) is ast.Starred for a in n.args)
                for a in n.args:
                    v=expression(a.value if type(a) is ast.Starred else a)
                    segments.append(segment(a,v,origins))
                    arguments.extend(list(v) if type(a) is ast.Starred else [v])
                if starred:
                    require(len(events)<MAX_EVENTS,'Native starred event bound')
                    event=dict(id=len(events),trace=trace_record(n,name,segments),argument_kinds=[a.kind for a in arguments])
                    event['features']=event_features(event);events.append(event)
                    if intervention is not None and intervention[0]==event['id']:
                        i=intervention[1];require(0<=i<len(arguments),'Argument intervention bound')
                        v=arguments[i];arguments[i]=Cell(v.kind,tuple(x+y for x,y in zip(v.values,(.25,-.5,.75,1.))))
                return invoke(n.func.id,arguments)
            value=expression(n.func.value)
            if n.func.attr in storage.METHODS:
                origins[id(n)]=list(value.shape)
                return storage.apply_method(value,n)
            step=storage.integer(n.args[0])%4
            return Cell(value.kind,value.values[step:]+value.values[:step])
        def assign(target,value):
            if type(target) is ast.Name:values[target.id]=value
            else:
                for n,v in zip(target.elts,value):values[n.id]=v
        for statement in fn.body[:-1]:
            if type(statement) is ast.Assign:assign(statement.targets[0],expression(statement.value))
            else:expression(statement.value)
        return expression(fn.body[-1].value)
    result=invoke('golden',[Cell('c',inputs[n].data) for n in plan['functions']['golden']['parameters']])
    output=[x for cell in flat(result) for x in cell.values]
    require(all(math.isfinite(x) for x in output),'Nonfinite starred probe')
    return output,events


def check_exercise(source,request):
    from candidate_contract import request_input_names
    validate_exercise_request(request);names=request_input_names(request);constants=request['public_constants']
    nodes,plan=_analyze(source,constants,request['layout']['output_ciphertexts'],names,starred_calls=True)
    probes=samples(names);budget=[0];baseline=[run_probe(nodes,plan,constants,p,budget) for p in probes]
    events=baseline[0][1];require(all(e==events for _,e in baseline),'Star event schedule varies across probes')
    influence={};cache={}
    def affects(event,index):
        key=(event['id'],index)
        if key not in cache:
            changed=[run_probe(nodes,plan,constants,p,budget,key)[0] for p in probes]
            cache[key]=max((abs(a-b) for new,(old,_) in zip(changed,baseline) for a,b in zip(new,old)),default=0.)>1e-9
        return cache[key]
    for feature in request['construction_exercise']['required_features']:
        witnesses=[]
        for event in events:
            if feature not in event['features']:continue
            if feature=='star.empty':witnesses.append(dict(event=event,status='trace_structural_only'));continue
            indices=list(range(len(event['argument_kinds'])))
            if indices and all(affects(event,i) for i in indices):
                witnesses.append(dict(event=event,status='finite_argument_influence',indices=indices))
        require(witnesses,'Required starred feature has no complete argument influence: '+feature)
        influence[feature]=witnesses
    return dict(schema=1,id=request['construction_exercise']['id'],required=request['construction_exercise']['required_features'],
        influence=influence,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest(),probe_steps=budget[0],
        real_native_trace_verified=False,encrypted_execution_verified=False,
        interpretation='Finite per-argument influence, not all-input equivalence or irreducibility; empty is structural only.')


def verify_trace_coverage(coverage,observed):
    require(type(observed) is list and len(observed)<=4096,'Invalid starred frontend observations')
    for record in observed:
        require(type(record) is dict and set(record)=={'caller','callee','span','segments'},'Starred trace fields')
        require(all(type(record[k]) is str for k in ('caller','callee')) and type(record['span']) is list and
                len(record['span'])==4 and all(type(v) is int and v>=0 for v in record['span']),'Starred trace site')
        require(type(record['segments']) is list and len(record['segments'])<=16,'Starred segment bound')
        for s in record['segments']:
            require(type(s) is dict and set(s)=={'kind','shape','count','form','origin_shape'} and
                    s['kind'] in ('ordinary','list','tuple','array') and type(s['count']) is int and
                    0<=s['count']<=16 and s['form'] in (None,'reverse','flatten','call_result'),'Starred segment fields')
            for shape in (s['shape'],s['origin_shape']):
                require(shape is None or type(shape) is list,'Starred shape type')
                if shape is not None:storage.dimensions(shape)
    for feature,witnesses in coverage['influence'].items():
        for w in witnesses:require(w['event']['trace'] in observed,'Starred witness absent from real frontend: '+feature)
    return dict(verified=True,checked_features=coverage['required'],observed_calls=len(observed),
                interpretation='Actual native call site and expanded segment shapes/order, not dynamic HEVM call counts.')
