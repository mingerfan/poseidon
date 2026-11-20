"""Invertible topological ordering of a pure public tensor DAG.

Grouping ready operators can improve the existing lossless metadata columns.
Node payloads and all arithmetic remain exact. The logical liveness estimate
counts float64 payloads and is not a native ciphertext/RSS certificate.
"""
from collections import Counter
import copy
import heapq
import math
from benchmark_graph import validate,canonical,digest,require
from delivery_component import bound,checked_bound

VERSION='poseidon-topological-metadata-scheduling-v1'
GROUPS=(('flatten','reshape','transpose','permute','rotate','slice','split'),
        ('add','subtract','multiply','negate','square','power','batch_norm','polynomial'),
        ('linear','conv1d','conv2d','avg_pool1d','avg_pool2d'),('concat','stack'),('sum','mean'))
PRIORITY={op:i for i,group in enumerate(GROUPS)for op in group}


def logical_liveness(model):
    checked=validate(model);uses=Counter(ref for n in model['nodes']for ref in n['inputs'])
    uses.update(o['value']for o in model['outputs'])
    live={s['name']:math.prod(s['shape'])for s in model['inputs']}
    maximum_elements=sum(live.values());maximum_values=len(live)
    for node in model['nodes']:
        for value in node['outputs']:
            if checked['kinds'][value]!='plain':live[value]=math.prod(checked['shapes'][value])
        maximum_elements=max(maximum_elements,sum(live.values()));maximum_values=max(maximum_values,len(live))
        for value in node['inputs']:
            uses[value]-=1
            if uses[value]==0:live.pop(value,None)
        for value in node['outputs']:
            if uses[value]==0:live.pop(value,None)
    return dict(peak_logical_elements=maximum_elements,peak_logical_values=maximum_values,
        float64_payload_bytes=8*maximum_elements,view_aliases_counted_as_separate_values=True,
        native_ciphertext_or_RSS_bound=False)


def restore(model,receipt):
    checked_bound(receipt);require(receipt['format']==VERSION,'Frozen topological schedule receipt')
    require(digest(model)==receipt['scheduled_model_sha256'],'Scheduled graph changed')
    nodes={n['id']:n for n in model['nodes']};order=receipt['original_order']
    require(len(order)==len(nodes)and set(order)==set(nodes),'Complete original node order')
    original=copy.deepcopy(model);original['nodes']=[copy.deepcopy(nodes[i])for i in order]
    validate(original);require(digest(original)==receipt['original_model_sha256'],'Original graph cannot be recovered')
    return original


def normalize(model,*,maximum_nodes=1024):
    require(type(maximum_nodes)is int and 1<=maximum_nodes<=1024,'Finite topological node budget')
    checked=validate(model);require(len(model['nodes'])<=maximum_nodes,'Topological node budget')
    before=canonical(model);nodes=model['nodes'];indices={n['id']:i for i,n in enumerate(nodes)}
    producers={value:i for i,n in enumerate(nodes)for value in n['outputs']}
    dependencies=[{producers[r]for r in n['inputs']if r in producers}for n in nodes]
    successors=[set()for _ in nodes];degrees=[len(d)for d in dependencies]
    for index,parents in enumerate(dependencies):
        for parent in parents:successors[parent].add(index)
    ready=[(PRIORITY[n['op']],i)for i,n in enumerate(nodes)if degrees[i]==0];heapq.heapify(ready);order=[]
    while ready:
        _,index=heapq.heappop(ready);order.append(index)
        for child in sorted(successors[index]):
            degrees[child]-=1
            if degrees[child]==0:heapq.heappush(ready,(PRIORITY[nodes[child]['op']],child))
    require(len(order)==len(nodes),'Complete acyclic topological schedule')
    new=copy.deepcopy(model);new['nodes']=[copy.deepcopy(nodes[i])for i in order];final=validate(new)
    require(canonical({k:v for k,v in model.items()if k!='nodes'})==canonical({k:v for k,v in new.items()if k!='nodes'}),
        'Topological schedule changed public graph payload')
    require(len({n['id']for n in new['nodes']})==len(nodes)and all(canonical(n)==canonical(nodes[indices[n['id']]])for n in new['nodes']),
        'Topological schedule changed a node or constant')
    require(final['output_shapes']==checked['output_shapes']and canonical(model)==before,'Topological schedule mutated signature or input')
    receipt=bound(dict(format=VERSION,original_model_sha256=digest(model),scheduled_model_sha256=digest(new),
        original_order=[n['id']for n in nodes],scheduled_order=[n['id']for n in new['nodes']],
        priority_groups=[list(g)for g in GROUPS],tie_break='original topological ordinal',
        original_liveness=logical_liveness(model),scheduled_liveness=logical_liveness(new),
        exact_node_payloads_and_constants=True,arithmetic_and_named_outputs_unchanged=True,
        model_name_dispatch=False,numerical_reference_used=False,compiler_feedback_used=False,DSL_generated=False,
        native_memory_certified=False,maximum_nodes=maximum_nodes))
    require(canonical(restore(new,receipt))==before,'Topological schedule is not invertible')
    return new,receipt
