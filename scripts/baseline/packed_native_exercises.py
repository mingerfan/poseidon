"""Finite construct influence and real-trace witnesses, NOT an FHE backend."""
import ast
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from decorated_functions import _analyze,literal
import native_array_core as storage
from seal_artifact_gate import require

TASK='hecate-periodic-packed-native-synthesis-v2'
CATALOG=Path(__file__).with_name('packed-native-exercises-v1.json')
EXERCISES=json.loads(CATALOG.read_text())
MAX_STEPS=524288
MAX_EVENTS=512

def exercise_spec(name):
    require(type(name) is str and name in EXERCISES,'Unknown packed-native exercise')
    return copy.deepcopy(EXERCISES[name]['spec'])

def descriptor(name):
    exercise_spec(name)
    return copy.deepcopy(EXERCISES[name]['model'])

def validate_exercise_request(request):
    spec=request.get('construction_exercise')
    require(request.get('task')==TASK and type(spec) is dict and spec==exercise_spec(spec.get('id')) and
            request.get('model')==descriptor(spec['id']),'Packed-native exercise model/specification changed')

def span(n):return [n.lineno,n.col_offset,n.end_lineno,n.end_col_offset]

@dataclass(frozen=True)
class Cell:
    kind:str
    values:tuple
    def binary(self,other,op):
        require(type(other) is Cell,'Probe arithmetic type')
        return Cell('c',tuple(op(a,b) for a,b in zip(self.values,other.values)))
    def __add__(self,other):return self.binary(other,lambda a,b:a+b)
    def __sub__(self,other):return self.binary(other,lambda a,b:a-b)
    def __mul__(self,other):return self.binary(other,lambda a,b:a*b)
    def __neg__(self):return Cell('c',tuple(-x for x in self.values))
    def rotate(self,k):return Cell(self.kind,self.values[k:]+self.values[:k])

def flat(value):
    import numpy as np
    if type(value) is np.ndarray:return list(value.flat)
    if type(value) in (tuple,list):return [c for v in value for c in flat(v)]
    require(type(value) is Cell,'Unexpected probe value')
    return [value]

def run_probe(nodes,plan,request,probe,budget,loops,intervention=None):
    import numpy as np
    period=request['layout']['input_slot_period'];events=[];invocations=[0]
    def spend():
        budget[0]+=1;require(budget[0]<=MAX_STEPS,'Packed-native influence work bound')
    def vector(value):
        values=value if type(value) is list else [value]
        require(len(values) in (1,period),'Probe constant period')
        return Cell('p',tuple(values*period if len(values)==1 else values))
    def changed(cell):
        return Cell(cell.kind,tuple(x+(.173,-.239,.317,-.419)[i%4] for i,x in enumerate(cell.values)))
    def observe(trace,features,result,invocation,*,mutates=False,arguments=False):
        spend();require(len(events)<MAX_EVENTS,'Packed-native influence event bound')
        cells=flat(result)
        event=dict(id=len(events),trace=trace,features=features,cells=len(cells),invocation=invocation)
        events.append(event)
        if intervention is None or event['id']!=intervention[0]:return result
        index=intervention[1];require(0<=index<len(cells),'Probe intervention index')
        if arguments:
            result=list(result);result[index]=changed(result[index]);return result
        if type(result) is np.ndarray:
            if not mutates:result=result.copy()
            result.flat[index]=changed(cells[index]);return result
        require(type(result) is Cell and index==0,'Probe scalar intervention')
        return changed(result)
    def invoke(name,args):
        spend();invocations[0]+=1;invocation=invocations[0];fn=nodes[name]
        values={k:vector(v) for k,v in request['public_constants'].items()}
        values.update(zip(plan['functions'][name]['parameters'],args))
        def stored(op,n,result,before=None):
            features=[]
            if op=='return' and name!='golden' and type(result) is np.ndarray and result.ndim==2 and result.size>=2:
                features.append('return.matrix')
            if op=='item' and type(before) is np.ndarray and before.ndim==0:features.append('item.zero')
            return observe(dict(kind='storage',op=op,caller=name,span=span(n),
                input_shape=list(before.shape) if type(before) is np.ndarray else None,
                shape=list(result.shape) if type(result) is np.ndarray else None),features,result,invocation)
        def expression(n):
            spend();number=literal(n)
            if number is not None:return vector(number)
            if type(n) is ast.Name:return values[n.id]
            if type(n) in (ast.List,ast.Tuple):
                items=[expression(x) for x in n.elts]
                return items if type(n) is ast.List else tuple(items)
            if type(n) is ast.UnaryOp:return -expression(n.operand)
            if type(n) is ast.BinOp:
                a,b=expression(n.left),expression(n.right)
                if type(n.op) is ast.Add:return a+b
                if type(n.op) is ast.Sub:return a-b
                return a*b
            if type(n) is ast.Subscript:
                v=expression(n.value);out=v[storage.index(n.slice)]
                return stored('index',n,out,v) if type(v) is np.ndarray else out
            if type(n) is ast.Attribute:
                v=expression(n.value);return stored('transpose',n,v.T,v)
            require(type(n) is ast.Call,'Unexpected checked packed-native AST')
            if storage.constructor(n):return stored('array',n,np.array(expression(n.args[0]),dtype=object))
            if type(n.func) is ast.Name:
                args=[]
                for part in n.args:
                    value=expression(part.value if type(part) is ast.Starred else part)
                    args.extend(list(value) if type(part) is ast.Starred else [value])
                if any(type(a) is ast.Starred for a in n.args):
                    args=observe(dict(kind='star',caller=name,callee=n.func.id,span=span(n)),
                        ['star.arguments'] if len(args)>=2 else [],args,invocation,arguments=True)
                return invoke(n.func.id,args)
            value=expression(n.func.value)
            if n.func.attr in storage.METHODS:return stored(n.func.attr,n,storage.apply_method(value,n),value)
            return value.rotate(storage.integer(n.args[0]))
        def assign(target,value):
            if type(target) is ast.Name:values[target.id]=value
            else:
                for n,v in zip(target.elts,value):values[n.id]=v
        for statement in fn.body[:-1]:
            if type(statement) is ast.AugAssign:
                target=statement.target.id;left=values[target];right=expression(statement.value)
                array=type(left) is np.ndarray
                shared=array and any(type(v) is np.ndarray and v is not left and np.shares_memory(v,left)
                                     for v in values.values())
                if type(statement.op) is ast.Add:left+=right
                elif type(statement.op) is ast.Sub:left-=right
                else:left*=right
                features=['array.view_inplace'] if shared else []
                if not array:
                    features=['scalar.augmented']
                    if any(start<=statement.lineno<=end for start,end in loops.get(name,[])):
                        features.append('loop.augmented')
                values[target]=observe(dict(kind='mutation' if array else 'augmented',function=name,
                    target=target,span=span(statement)),features,left,invocation,mutates=array)
            elif type(statement) is ast.Assign:assign(statement.targets[0],expression(statement.value))
            else:expression(statement.value)
        return stored('return',fn.body[-1],expression(fn.body[-1].value))
    count=request['layout']['model_input_binding']['logical_elements']
    x=Cell('c',tuple((((i*i+3*i+7*probe+1)%23)-11)/32 if i<count else 0.
                     for i in range(period)))
    args=[x if n=='x' else Cell('c',(0.,)*period) for n in plan['functions']['golden']['parameters']]
    output=flat(invoke('golden',args))
    selected=[output[c].values[s] for c,s in request['layout']['output_selectors']]
    require(all(math.isfinite(x) for x in selected),'Nonfinite packed-native probe')
    return selected,events

def check_exercise(source,request):
    from candidate_contract import request_input_names
    validate_exercise_request(request)
    nodes,plan=_analyze(source,request['public_constants'],request['layout']['output_ciphertexts'],
        request_input_names(request),array_mutation=True,slot_period=request['layout']['input_slot_period'])
    loops={fn.name:[(n.lineno,n.end_lineno) for n in ast.walk(fn) if type(n) is ast.For]
           for fn in ast.parse(source).body}
    budget=[0];baseline=[run_probe(nodes,plan,request,k,budget,loops) for k in range(3)]
    events=baseline[0][1]
    require(all(es==events for _,es in baseline),'Probe event schedule changed with public samples')
    cache={}
    def affects(event,index):
        key=(event['id'],index)
        if key not in cache:
            changed=[run_probe(nodes,plan,request,k,budget,loops,key)[0] for k in range(3)]
            cache[key]=max((abs(a-b) for new,(old,_) in zip(changed,baseline)
                            for a,b in zip(new,old)),default=0.)>1e-9
        return cache[key]
    witnesses={}
    for feature in request['construction_exercise']['required_features']:
        for event in events:
            if feature not in event['features']:continue
            if feature=='loop.augmented' and sum(e['trace']==event['trace'] and
                    e['invocation']==event['invocation'] for e in events)<2:continue
            indices=list(range(event['cells']))
            if feature=='star.arguments':
                selected=indices if all(affects(event,i) for i in indices) else []
            else:
                selected=[]
                for index in indices:
                    if affects(event,index):selected.append(index)
                    if len(selected)>=(2 if feature=='return.matrix' else 1):break
            if len(selected)>=(2 if feature in ('return.matrix','star.arguments') else 1):
                witnesses[feature]=dict(trace=event['trace'],influencing_cells=selected,
                    evidence='finite_public_probe_influence_not_semantic_proof')
                break
        require(feature in witnesses,'Missing contributing construction feature: '+feature)
    return dict(schema=1,id=request['construction_exercise']['id'],witnesses=witnesses,
                probe_count=3,slot_period=request['layout']['input_slot_period'],
                work=budget[0],plaintext_reference_used=False,real_frontend_checked=False)

def verify_trace_coverage(checked,records):
    observed=[]
    for event in records['storage']:
        observed.append(dict(kind='storage',**event))
    for event in records['star']:
        observed.append({k:event[k] for k in ('caller','callee','span')}|{'kind':'star'})
    for group,kind in (('augmented','augmented'),('mutation','mutation')):
        for event in records[group]:
            observed.append({k:event[k] for k in ('function','target','span')}|{'kind':kind})
    for feature,witness in checked['witnesses'].items():
        require(witness['trace'] in observed,'Missing real native trace witness: '+feature)
    return dict(schema=1,id=checked['id'],features=list(checked['witnesses']),
                actual_frontend_checked=True,finite_influence_checked=True,
                all_input_semantic_proof=False)

