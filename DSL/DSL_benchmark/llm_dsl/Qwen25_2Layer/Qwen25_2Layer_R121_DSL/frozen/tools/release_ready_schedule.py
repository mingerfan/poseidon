"""Draft ready-list host schedule prioritizing immediate CT lifetime reductions.

No arithmetic, operand order, metadata, constant, or recomputation changes.
PT cache and actual capacity must be measured after scheduling; no gain assumed.
"""
from collections import Counter,defaultdict
import heapq
from defer_cpu_encodes import SECTIONS,footprint
from reorder_runtime_plan import graph_hash,transform_dfs


def transform(plan):
    baseline,admission=transform_dfs(plan)
    original=[op for op in plan['execution'] if op['kind']=='compute']
    nodes={op['output']:op for op in original}
    values={v['id']:v for v in plan['values']}
    roots=set(plan['external_inputs'])|{o['output'] for o in plan['initialization']}
    external=set(plan['external_inputs']);finals=set(plan['final_outputs'])
    sizes={v:65536*8*(row['level']+1)*row['components'] if row['kind']=='ciphertext' else 0
           for v,row in values.items()}
    inputs={v:Counter(op['inputs']) for v,op in nodes.items()}
    consumers=defaultdict(set)
    for out,counts in inputs.items():
        for value in counts:consumers[value].add(out)
    remaining=Counter(v for op in original for v in op['inputs'])
    missing={v:len(set(op['inputs'])-roots) for v,op in nodes.items()}
    positions={op['output']:i for i,op in enumerate(original)}
    emitted=set(roots);ready=set();heap=[];versions=Counter()

    def push(value):
        assert value in ready
        freed=sum(sizes[v] for v,n in inputs[value].items()
                  if remaining[v]==n and v not in finals and v not in external)
        net=sizes[value]-freed
        versions[value]+=1
        heapq.heappush(heap,(net,-freed,sizes[value],positions[value],versions[value],value))

    for value,number in missing.items():
        if number==0:ready.add(value);push(value)
    ordered=[]
    while heap:
        *_,version,value=heapq.heappop(heap)
        if value not in ready or versions[value]!=version:continue
        ready.remove(value);op=nodes[value]
        assert all(v in emitted for v in op['inputs']) and value not in emitted
        ordered.append(op);emitted.add(value)
        affected=set()
        for input_value,count in inputs[value].items():
            remaining[input_value]-=count;assert remaining[input_value]>=0
            affected.update(consumers[input_value]&ready)
        for child in consumers[value]:
            missing[child]-=1;assert missing[child]>=0
            if missing[child]==0:
                assert child not in ready and child not in emitted
                ready.add(child);affected.add(child)
        for candidate in affected:push(candidate)
    assert not ready and emitted==set(values) and len(ordered)==len(original)
    assert graph_hash(ordered)==graph_hash(original) and not any(remaining.values())
    remaining=Counter(v for op in ordered for v in op['inputs'])
    live=set(roots);execution=[]
    for op in ordered:
        assert all(v in live for v in op['inputs'])
        out=op['output'];assert out not in live;live.add(out);execution.append(op)
        for v in op['inputs']:
            remaining[v]-=1
            if remaining[v]==0 and v not in finals:
                execution.append(dict(kind='release',value=v));live.remove(v)
        if remaining[out]==0 and out not in finals:
            execution.append(dict(kind='release',value=out));live.remove(out)
    assert live==finals
    result=dict(plan,initialization=list(plan['initialization']),execution=execution,finalization=[])
    ordinal=0
    for section in SECTIONS:
        result[section]=[dict(op,ordinal=ordinal+i) for i,op in enumerate(result[section])]
        ordinal+=len(result[section])
    before=Counter(op['value'] for s in SECTIONS for op in plan[s] if op['kind']=='release')
    after=Counter(op['value'] for s in SECTIONS for op in result[s] if op['kind']=='release')
    assert before==after and all(n==1 for n in after.values())
    for key in plan:
        if key not in SECTIONS:assert result[key]==plan[key],key
    old,new=footprint(baseline),footprint(result)
    assert old['counts']==new['counts'] and old['by_level']==new['by_level']
    return result,dict(passed=True,policy='release_ready',all_arithmetic_nodes_identical=True,
        operand_order_preserved=True,values_levels_scales_identical=True,
        encode_payloads_identical=True,same_released_values=True,no_recomputation=True,
        arithmetic_graph_sha256=graph_hash(ordered),baseline=old,candidate=new,
        note='Draft host schedule; requires cache-aware measurement, native verifier, Mock and CKKS')
