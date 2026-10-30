"""Experimental host DAG schedule; preserve every arithmetic node and operand order."""
from collections import Counter
import hashlib
import json
from defer_cpu_encodes import SECTIONS,canonical,footprint

def graph_hash(ops):
    rows=sorted((canonical(o) for o in ops if o['kind']=='compute'),key=lambda o:int(o['output']))
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def transform_dfs(plan):
    assert plan['target']['world_size']==1 and plan['target']['device_counts']==[0]
    assert all(v['place']==dict(kind='host',rank=0) for v in plan['values'])
    assert all(o['kind']=='encode' for o in plan['initialization'])
    assert all(o['kind'] in ('compute','release') for o in plan['execution'])
    assert all(o['kind']=='release' for o in plan['finalization'])
    assert all('reuse_input' not in o for s in SECTIONS for o in plan[s])
    baseline=footprint(plan)
    original=[o for o in plan['execution'] if o['kind']=='compute']
    nodes={o['output']:o for o in original};assert len(nodes)==len(original)
    roots=set(plan['external_inputs'])|{o['output'] for o in plan['initialization']}
    values={v['id'] for v in plan['values']};assert set(nodes)|roots==values
    emitted=set(roots);visiting=set();ordered=[]
    # Original operand order is retained in each node; this only chooses when
    # to evaluate independent producers. Iterative traversal handles deep sums.
    for target in [*plan['final_outputs'],*(o['output'] for o in original)]:
        stack=[(target,False)]
        while stack:
            value,finish=stack.pop()
            if value in emitted:continue
            assert value in nodes,'Missing producer'
            op=nodes[value]
            if finish:
                assert all(v in emitted for v in op['inputs'])
                ordered.append(op);emitted.add(value);visiting.remove(value)
            else:
                assert value not in visiting,'Cycle in dependency graph'
                visiting.add(value);stack.append((value,True))
                for dependency in reversed(list(dict.fromkeys(op['inputs']))):
                    if dependency not in emitted:stack.append((dependency,False))
    assert emitted==values and len(ordered)==len(original)
    assert graph_hash(ordered)==graph_hash(original)
    remaining=Counter(v for op in ordered for v in op['inputs'])
    finals=set(plan['final_outputs']);live=set(roots);execution=[]
    for v in sorted(roots,key=int):
        assert remaining[v]>0 or v in finals,'Unused initial value outside bounded-cache scope'
    for op in ordered:
        assert all(v in live for v in op['inputs'])
        out=op['output'];assert out not in live;live.add(out);execution.append(op)
        for v in op['inputs']:
            remaining[v]-=1;assert remaining[v]>=0
            if remaining[v]==0 and v not in finals:
                assert v in live;execution.append(dict(kind='release',value=v));live.remove(v)
        if remaining[out]==0 and out not in finals:
            execution.append(dict(kind='release',value=out));live.remove(out)
    assert live==finals and all(v==0 for v in remaining.values())
    result=dict(plan,initialization=list(plan['initialization']),execution=execution,finalization=[])
    ordinal=0
    for section in SECTIONS:
        result[section]=[dict(o,ordinal=ordinal+i) for i,o in enumerate(result[section])]
        ordinal+=len(result[section])
    assert result['values']==plan['values']
    for key in plan:
        if key not in SECTIONS:assert result[key]==plan[key]
    before_releases=Counter(o['value'] for s in SECTIONS for o in plan[s] if o['kind']=='release')
    after_releases=Counter(o['value'] for o in result['execution'] if o['kind']=='release')
    assert before_releases==after_releases and all(n==1 for n in after_releases.values())
    candidate=footprint(result)
    assert baseline['counts']==candidate['counts']
    return result,dict(passed=True,arithmetic_graph_sha256=graph_hash(original),all_arithmetic_nodes_identical=True,
        operand_order_preserved=True,values_levels_scales_identical=True,encode_payloads_identical=True,
        same_released_values=True,changed_compute_schedule=True,no_recomputation=True,
        baseline=baseline,candidate=candidate,
        note='Experimental host-only schedule post-pass; not an official compiler change and not CKKS precision proof')
