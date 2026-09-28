"""Unified native witnesses over arbitrary bounded logical input/output bindings.

Candidate Python remains inert AST. Reuse the native validators and observer
protocols; finite public interventions are separate from real Hecate/FHE evidence.
Structural empty containers/expansions are explicitly excluded from numeric credit.
"""
import ast
from collections import Counter
import math
import json
import numpy as np
from decorated_functions import _analyze,literal
import native_array_core as storage
from native_array_exercises import features as array_features,trace_record as array_trace
from native_star_exercises import segment,trace_record as star_trace,event_features as star_features
from packed_native_exercises import Cell,flat,span,MAX_STEPS,MAX_EVENTS
from unified_graph_exercises import STRUCTURAL
from seal_artifact_gate import require

def run_probe(nodes,plan,request,probe,budget,loops,intervention=None,helper_evaluator=None,helper_group=None):
    period=request['layout']['input_slot_period'];events=[];invocations=[0]
    def spend():
        budget[0]+=1;require(budget[0]<=MAX_STEPS,'Unified native influence work bound')
    def vector(value):
        values=value if type(value) is list else [value]
        require(len(values) in (1,period),'Probe constant period')
        return Cell('p',tuple(values*period if len(values)==1 else values))
    def changed(cell):
        return Cell(cell.kind,tuple(x+(.173,-.239,.317,-.419)[i%4] for i,x in enumerate(cell.values)))
    def observe(trace,features,result,invocation,*,mutates=False,arguments=False):
        spend();require(len(events)<MAX_EVENTS,'Unified native influence event bound')
        cells=flat(result)
        event=dict(id=len(events),trace=trace,features=sorted(set(features)),cells=len(cells),
                   kinds=[c.kind for c in cells],invocation=invocation,arguments=arguments)
        events.append(event)
        if intervention is None or event['id']!=intervention[0]:return result
        index=intervention[1];require(0<=index<len(cells),'Probe intervention index')
        if arguments:
            result=list(result);result[index]=changed(result[index]);return result
        if type(result) is np.ndarray:
            if not mutates:result=result.copy()
            result.flat[index]=changed(cells[index]);return result
        if type(result) in (list,tuple):
            # Decorated native flat sequence results are guaranteed by _analyze.
            original_type=type(result);result=list(result);result[index]=changed(cells[index]);return tuple(result) if original_type is tuple else result
        require(type(result) is Cell and index==0,'Probe scalar intervention')
        return changed(result)
    def invoke(name,args):
        spend();invocations[0]+=1;invocation=invocations[0];fn=nodes[name];origins={}
        values={k:vector(v) for k,v in request['public_constants'].items()}
        values.update(zip(plan['functions'][name]['parameters'],args))
        def stored(op,n,result,before=None):
            found=array_features(op,n,name,result,before)
            return observe(dict(kind='storage',**array_trace(op,n,name,result,before)),
                           found,result,invocation)
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
            require(type(n) is ast.Call,'Unexpected checked unified native AST')
            if storage.constructor(n):return stored('array',n,np.array(expression(n.args[0]),dtype=object))
            if type(n.func) is ast.Name:
                args=[];segments=[]
                for part in n.args:
                    value=expression(part.value if type(part) is ast.Starred else part)
                    segments.append(segment(part,value,origins))
                    args.extend(list(value) if type(part) is ast.Starred else [value])
                if any(type(a) is ast.Starred for a in n.args):
                    record=star_trace(n,name,segments)
                    found=star_features(dict(trace=record,argument_kinds=[a.kind for a in args]))
                    if len(args)>=2:found.append('star.arguments')
                    args=observe(dict(kind='star',**record),found,args,invocation,arguments=True)
                target=n.func.id
                if helper_evaluator is not None and target in request['upstream_helpers']['helpers']:
                    result=helper_evaluator(target,args)
                    trace=dict(kind='upstream',caller=name,callee=target,span=span(n))
                    result=observe(trace,['upstream.'+target],result,invocation)
                    return changed(result) if helper_group==target else result
                callee=plan['functions'][target]
                trace=dict(kind='call',caller=name,callee=target,span=span(n))
                arg_features=[]
                if callee['kinds'].count('c')>=2:arg_features.append('call.two_cipher_arguments')
                if 'p' in callee['kinds']:arg_features.append('call.public_argument')
                if arg_features:args=observe(trace,arg_features,args,invocation,arguments=True)
                result=invoke(target,args);found=[]
                if type(result) is Cell:
                    found.append('call.scalar_cipher' if result.kind=='c' else 'call.plain_return')
                if type(result) in (tuple,list) and not result:found.append('call.empty_return')
                if type(result) is tuple and len(result)>=2 and all(c.kind=='c' for c in result):
                    found.append('call.tuple_multi')
                if name!='golden':found.append('call.nested')
                if nodes[target].lineno>fn.lineno:found.append('call.forward')
                if not args and flat(result):found.append('call.zero_arguments')
                returned=nodes[target].body[-1].value
                if (len(nodes[target].body)==1 and type(returned) is ast.Name and
                        type(result) is Cell and result.kind=='c' and returned.id in callee['parameters'] and
                        callee['kinds'][callee['parameters'].index(returned.id)]=='c'):
                    found.append('call.identity')
                result=observe(trace,found,result,invocation)
                if type(result) is np.ndarray:result=stored('call',n,result)
                return result
            value=expression(n.func.value)
            if n.func.attr in storage.METHODS:
                origins[id(n)]=list(value.shape)
                return stored(n.func.attr,n,storage.apply_method(value,n),value)
            return value.rotate(storage.integer(n.args[0]))
        def assign(target,value):
            if type(target) is ast.Name:values[target.id]=value
            else:
                if type(value) is np.ndarray:value=stored('unpack',target,value,value)
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
                found=['array.view_inplace'] if shared else []
                if not array:
                    found=['scalar.augmented']
                    if any(start<=statement.lineno<=end for start,end in loops.get(name,[])):
                        found.append('loop.augmented')
                values[target]=observe(dict(kind='mutation' if array else 'augmented',function=name,
                    target=target,span=span(statement)),found,left,invocation,mutates=array)
            elif type(statement) is ast.Assign:assign(statement.targets[0],expression(statement.value))
            else:expression(statement.value)
        return stored('return',fn.body[-1],expression(fn.body[-1].value))
    bindings=request['layout']['inputs'];inputs={}
    for j,b in enumerate(bindings):
        count=b.get('elements',math.prod(b['shape']))
        inputs[b['dsl_name']]=Cell('c',tuple((((i*i+3*i+7*probe+5*j+1)%23)-11)/32
                                            if i<count else 0. for i in range(period)))
    inputs['zero_ct']=Cell('c',(0.,)*period)
    args=[inputs[n] for n in plan['functions']['golden']['parameters']]
    output=flat(invoke('golden',args))
    # Count reachable result events, not argument events or unused definitions.
    repeats=Counter(e['trace']['callee'] for e in events if e['trace']['kind']=='call' and not e['arguments'])
    for e in events:
        if e['trace']['kind']=='call' and not e['arguments'] and repeats[e['trace']['callee']]>=2 and e['cells']:
            e['features']=sorted(set(e['features'])|{'call.repeated'})
    selected=[output[c].values[s] for c,s in request['layout']['output_selectors']]
    require(all(math.isfinite(x) for x in selected),'Nonfinite unified native probe')
    return selected,events

def check_exercise(source,request):
    from candidate_contract import request_input_names
    from unified_graph_contract import validate_request
    validate_request(request)
    nodes,plan=_analyze(source,request['public_constants'],request['layout']['output_ciphertexts'],
                        request_input_names(request),array_mutation=True,slot_period=request['layout']['input_slot_period'],
                        upstream_helpers='upstream_helpers' in request,
                        upstream_request=request if 'upstream_helpers' in request else None)
    loops={fn.name:[(n.lineno,n.end_lineno) for n in ast.walk(fn) if type(n) is ast.For]
           for fn in ast.parse(source).body}
    evaluator=None
    if 'upstream_helpers' in request:
        from upstream_helper_coverage import helper_evaluator
        evaluator=helper_evaluator(request)
    budget=[0];baseline=[run_probe(nodes,plan,request,k,budget,loops,helper_evaluator=evaluator) for k in range(3)]
    events=baseline[0][1]
    require(all(es==events for _,es in baseline),'Probe event schedule changed')
    cache={}
    def affects(event,index):
        key=(event['id'],index)
        if key not in cache:
            changed=[run_probe(nodes,plan,request,k,budget,loops,key,helper_evaluator=evaluator)[0] for k in range(3)]
            cache[key]=max((abs(a-b) for new,(old,_) in zip(changed,baseline)
                           for a,b in zip(new,old)),default=0.)>1e-9
        return cache[key]
    witnesses={}
    for feature in request['construction_exercise']['required_features']:
        if feature in STRUCTURAL:
            event=next((e for e in events if feature in e['features'] and e['cells']==0),None)
            require(event is not None,'Missing reachable structural feature: '+feature)
            witnesses[feature]=dict(trace=event['trace'],influencing_cells=[],
                                    evidence='reachable_structure_only_no_numeric_credit')
            continue
        matches=[]
        for event in events:
            if feature not in event['features']:continue
            if feature=='loop.augmented' and sum(e['trace']==event['trace'] and
                    e['invocation']==event['invocation'] for e in events)<2:continue
            indices=list(range(event['cells']))
            if event['arguments']:
                needed=[i for i,k in enumerate(event['kinds']) if
                    feature!='call.public_argument' and feature!='call.two_cipher_arguments' or
                    feature=='call.public_argument' and k=='p' or feature=='call.two_cipher_arguments' and k=='c']
                chosen=needed if needed and all(affects(event,i) for i in needed) else []
                minimum=2 if feature in ('star.arguments','call.two_cipher_arguments') else 1
            else:
                minimum=2 if feature in ('return.matrix','return.rank4','reshape.rank4','transpose','index.reverse','unpack.rows','call.tuple_multi','return.mixed') else 1
                chosen=[i for i in indices if affects(event,i)]
            if len(chosen)<minimum:continue
            if feature=='return.mixed' and {event['kinds'][i] for i in chosen}!={'c','p'}:continue
            if feature=='unpack.rows':
                row_size=math.prod(event['trace']['shape'][1:])
                if len({i//row_size for i in chosen})<2:continue
            matches.append((event,chosen))
            if feature!='call.repeated':break
        if feature=='call.repeated':
            groups={}
            for e,ii in matches:groups.setdefault(e['trace']['callee'],[]).append((e,ii))
            matches=next((v for v in groups.values() if len(v)>=2),[])[:2]
        require(matches and (feature!='call.repeated' or len(matches)>=2),'Missing contributing construction feature: '+feature)
        event,chosen=matches[0]
        witness=dict(trace=event['trace'],influencing_cells=chosen,
                     evidence='finite_public_probe_influence_not_semantic_proof')
        if len(matches)>1:witness['additional_traces']=[e['trace'] for e,_ in matches[1:]]
        witnesses[feature]=witness
    structural=[f for f in witnesses if f in STRUCTURAL]
    return dict(schema=2,checker_contract='unified-native-witness-v3',id=request['construction_exercise']['id'],
                witnesses=witnesses,structural_features=structural,
                numeric_features=[f for f in witnesses if f not in STRUCTURAL],
                probe_count=3,slot_period=request['layout']['input_slot_period'],work=budget[0],
                plaintext_reference_used=False,real_frontend_checked=False)

def verify_trace_coverage(checked,records):
    observed=[]
    observed.extend(dict(kind='storage',**e) for e in records['storage'])
    observed.extend(dict(kind='star',**e) for e in records['star'])
    observed.extend(dict(kind='call',**e) for e in records.get('calls',[]))
    for group,kind in (('augmented','augmented'),('mutation','mutation')):
        observed.extend({k:e[k] for k in ('function','target','span')}|{'kind':kind} for e in records[group])
    counts=Counter(json.dumps(e,sort_keys=True) for e in observed)
    for feature,w in checked['witnesses'].items():
        needed=Counter(json.dumps(e,sort_keys=True) for e in [w['trace'],*w.get('additional_traces',[])])
        require(all(counts[e]>=n for e,n in needed.items()),'Missing real native trace witness: '+feature)
    return dict(schema=2,id=checked['id'],features=list(checked['witnesses']),
                numeric_features=checked['numeric_features'],structural_features=checked['structural_features'],
                actual_frontend_checked=True,finite_influence_checked=bool(checked['numeric_features']),
                structural_only=not bool(checked['numeric_features']),all_input_semantic_proof=False)
