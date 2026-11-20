"""Declared-function domain checks using the unchanged mathematical primitives.

No compiler, CKKS, candidate Python, original float32 model or preprocessing is
executed. Bounds are selected from the frozen approximation-definition JSON.
"""
import copy
import math
from pathlib import Path

import numpy as np
from benchmark_graph import validate
from benchmark_math import operation
from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha
from retained_dual_reference_rows import _json

FORMAT='poseidon-retained-declared-domain-contract-v1'
RESULT='poseidon-retained-declared-domain-observation-v1'


class DomainWorkLimit(ValueError):pass


def _interval(document,pointer):
    require(type(pointer)is list and 1<=len(pointer)<=16,'Explicit domain JSON pointer')
    value=document
    for key in pointer:
        require(type(key)in(str,int),'Domain pointer component')
        if type(value)is dict:
            require(type(key)is str and key in value,'Domain object field');value=value[key]
        else:
            require(type(value)is list and type(key)is int and 0<=key<len(value),'Domain array index');value=value[key]
    require(type(value)is list and len(value)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in value)
        and value[0]<=value[1],'Original finite domain interval')
    return list(value)


def contract(model,definition,polynomial_domains):
    validate(model)
    nodes=[n for n in model['nodes']if n['op']=='polynomial']
    require(type(polynomial_domains)is dict and set(polynomial_domains)=={n['id']for n in nodes},'Every declared polynomial needs an explicit domain mapping')
    if definition is None:
        require(not nodes,'Polynomial domain definition required');document={}
    else:
        require(type(definition)is dict and Path(definition['path']).suffix=='.json','Original domain definition file')
        raw=read_bytes(definition['path'],16*1024**2);require(sha(raw)==definition['sha256'],'Original domain definition changed')
        document=_json(raw)
    predicates=[]
    for node in nodes:
        pointer=polynomial_domains[node['id']];interval=_interval(document,pointer)
        predicates.append(dict(id=node['id']+':arg0',node_id=node['id'],reference=node['inputs'][0],
            definition_pointer=copy.deepcopy(pointer),interval=interval))
    return bound(dict(format=FORMAT,declared_reference_model_sha256=sha(canonical(model)),definition=copy.deepcopy(definition),
        predicates=predicates,all_declared_polynomial_nodes_covered=True,scope='declared_function_mathematics_only',
        runtime_or_refresh_domain_certified=False,bounds_or_tolerances_changed=False))


def verify_contract(value,model):
    checked_bound(value);require(value.get('format')==FORMAT,'Declared domain contract format')
    mapping={p['node_id']:p['definition_pointer']for p in value['predicates']}
    require(len(mapping)==len(value['predicates'])and value==contract(model,value['definition'],mapping),'Declared domain mapping/model/bounds changed')
    return value


def _cost(node,values,shapes):
    op=node['op'];out=sum(math.prod(shapes[n])for n in node['outputs']);xs=[values[k]for k in node['inputs']]
    if op=='linear':return out*(2*xs[1].shape[-1]+2)
    if op.startswith('conv'):
        return out*(2*(xs[0].shape[1]if xs[0].ndim==xs[1].ndim else xs[0].shape[0])//node['attrs']['groups']*math.prod(xs[1].shape[2:])+2)
    if op.startswith('avg_pool'):return out*(math.prod(node['attrs']['kernel'])+2)
    if op=='polynomial':return out*(4*xs[1].size+8)
    if op=='batch_norm':return out*16
    if op in('sum','mean'):return xs[0].size*2+out
    if op=='power':return out*(node['attrs']['exponent']+1)
    return sum(x.size for x in xs)+2*out


def evaluate(model,inputs,value,*,maximum_scalar_operations):
    """One pruned reference traversal; never use CKKS observations as domains."""
    verify_contract(value,model);checked=validate(model)
    require(type(maximum_scalar_operations)is int and 1<=maximum_scalar_operations<=2**28,'Original finite domain mathematical work limit')
    require(set(inputs)=={s['name']for s in model['inputs']},'Declared domain input names')
    arrays={k:np.asarray(v,dtype=np.float64)for k,v in model['constants'].items()}
    for spec in model['inputs']:
        x=inputs[spec['name']]
        require(type(x)is np.ndarray and x.dtype==np.float64 and list(x.shape)==spec['shape']and np.isfinite(x).all(),'Declared domain input dtype/shape')
        arrays[spec['name']]=x.copy()
    producers={out:node for node in model['nodes']for out in node['outputs']}
    required=set();pending=[p['reference']for p in value['predicates']];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);node=producers.get(name)
        if node is not None:required.add(node['id']);pending.extend(node['inputs'])
    used=calls=0;failure=None
    def charge(count):
        nonlocal used
        if used+count>maximum_scalar_operations:raise DomainWorkLimit('Declared domain mathematical work exhausted')
        used+=count
    for node in model['nodes']:
        if node['id']not in required:continue
        charge(_cost(node,arrays,checked['shapes']))
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            result=operation(node['op'],[arrays[k]for k in node['inputs']],node['attrs'])
        parts=result if node['op']=='split'else[result];calls+=1
        for name,item in zip(node['outputs'],parts):
            x=np.asarray(item,dtype=np.float64)
            require(list(x.shape)==checked['shapes'][name],'Declared domain operation shape')
            arrays[name]=x
            if not np.isfinite(x).all():failure=node['id']
        if failure is not None:break
    observations=[]
    for predicate in value['predicates']:
        x=arrays.get(predicate['reference']);finite=x is not None and np.isfinite(x).all()
        if x is not None:charge(3*x.size)
        low=float(np.min(x))if finite else None;high=float(np.max(x))if finite else None
        lower,upper=predicate['interval']
        observations.append(dict(predicate_id=predicate['id'],reference=predicate['reference'],
            elements=int(x.size)if x is not None else None,all_finite=bool(finite),minimum=low,maximum=high,
            expected_interval=list(predicate['interval']),passed=bool(finite and lower<=low and high<=upper)))
    return bound(dict(format=RESULT,domain_contract_binding=value['binding'],declared_reference_model_sha256=sha(canonical(model)),
        observations=observations,passed=failure is None and all(p['passed']for p in observations),failed_reference_node=failure,
        reference_operation_calls=calls,accounted_scalar_work=used,maximum_scalar_operations=maximum_scalar_operations,
        candidate_python_executed=False,CKKS_outputs_used=False,encrypted_execution=False,runtime_or_refresh_domain_certified=False))
