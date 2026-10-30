"""Experimental bounded CPU plaintext cache using immutable re-encoding.

Only Encode roots are cached. Compute order and ciphertext lifetime stay fixed.
Evicted plaintexts are encoded again with fresh ValueIds before later consumers.
Eviction uses public next-use positions, with size as a tiebreak.
No secret key, ciphertext refresh, data-dependent branching or API modification.
"""
from collections import OrderedDict, deque
import hashlib
import json
from defer_cpu_encodes import SECTIONS, canonical, footprint, semantic_hash, transform

def normalized_computes(plan, aliases):
    h=hashlib.sha256()
    for section in SECTIONS:
        for op in plan[section]:
            if op['kind']!='compute':continue
            row=canonical(op)
            row['inputs']=[aliases.get(v,v) for v in row['inputs']]
            h.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode());h.update(b'\n')
    return h.hexdigest()

def transform_bounded(plan,budget_bytes):
    # Reuse strict admission and original lifecycle verification from round012.
    deferred,deferred_audit=transform(plan)
    del deferred
    assert budget_bytes>0
    roots={o['output']:o for o in plan['initialization']}
    original_values={v['id']:v for v in plan['values']}
    assert all(isinstance(v,str) and str(int(v))==v for v in original_values)
    next_id=max(map(int,original_values))+1
    size=lambda v:65536*8*(v['level']+1)*v['components']
    root_sizes={v:size(original_values[v]) for v in roots}
    cache=OrderedDict();resident=0;peak=0;seen=set();aliases={};new_values=[]
    reencodes=0;evictions=0;new_execution=[];new_finalization=[]
    actual_encodes={};payload_counts={};definition_sources={}
    # All use positions are public compiler schedule metadata, not slot values.
    use_positions={root:deque() for root in roots}
    total_compute=0
    for op in plan['execution']:
        if op['kind']=='compute':
            for root in dict.fromkeys(v for v in op['inputs'] if v in roots):
                use_positions[root].append(total_compute)
            total_compute+=1
    compute_index=0


    def encode(root,out):
        nonlocal next_id,resident,peak,reencodes
        if root not in seen:identifier=root;seen.add(root)
        else:
            identifier=str(next_id);next_id+=1;reencodes+=1
            new_values.append(dict(original_values[root],id=identifier));aliases[identifier]=root
        definition_sources[identifier]=root
        out.append(dict(roots[root],output=identifier))
        cache[root]=identifier;resident+=root_sizes[root];peak=max(peak,resident)
        actual_encodes[root]=actual_encodes.get(root,0)+1
        kind=roots[root]['payload']['kind'];assert kind in ('inline','bundle')
        payload_counts[kind]=payload_counts.get(kind,0)+1

    def process(op,out):
        nonlocal resident,evictions,compute_index
        if op['kind']=='release':
            value=op['value']
            if value not in roots:out.append(op);return
            assert value in seen,'Unused Encode root is outside this experiment scope'
            if value in cache:
                out.append(dict(op,value=cache.pop(value)));resident-=root_sizes[value]
            return
        assert op['kind']=='compute'
        needed=list(dict.fromkeys(v for v in op['inputs'] if v in roots))
        pinned=set(needed)
        for root in needed:
            assert use_positions[root] and use_positions[root][0]==compute_index
            use_positions[root].popleft()

        assert sum(root_sizes[v] for v in needed)<=budget_bytes,'Operation needs more simultaneous encoded plaintext than cache budget'
        for root in needed:
            if root in cache:continue
            while resident+root_sizes[root]>budget_bytes:
                candidates=[v for v in cache if v not in pinned]
                victim=max(candidates,key=lambda v:(use_positions[v][0] if use_positions[v] else float('inf'),root_sizes[v],int(v))) if candidates else None
                assert victim is not None,'No safe cache eviction'
                out.append(dict(kind='release',value=cache.pop(victim)))
                resident-=root_sizes[victim];evictions+=1
            encode(root,out)
        for root in needed:cache.move_to_end(root)
        out.append(dict(op,inputs=[cache[v] if v in roots else v for v in op['inputs']]))
        compute_index+=1

    for op in plan['execution']:process(op,new_execution)
    for op in plan['finalization']:process(op,new_finalization)
    assert not cache and resident==0 and seen==set(roots)
    assert compute_index==total_compute and all(not q for q in use_positions.values())
    result=dict(plan,format_version=3,values=plan['values']+new_values,
        initialization=[],execution=new_execution,finalization=new_finalization)
    ordinal=0
    for section in SECTIONS:
        ops=[]
        for op in result[section]:ops.append(dict(op,ordinal=ordinal));ordinal+=1
        result[section]=ops
    assert normalized_computes(result,aliases)==semantic_hash(plan,'compute')
    # Preserve every non-Encode-root Release, especially every ciphertext Release.
    original_releases=[canonical(o) for s in SECTIONS for o in plan[s] if o['kind']=='release' and o['value'] not in roots]
    new_releases=[canonical(o) for s in SECTIONS for o in result[s]
        if o['kind']=='release' and aliases.get(o['value'],o['value']) not in roots]
    assert new_releases==original_releases
    for v in new_values:
        root=aliases[v['id']];assert dict(v,id=root)==original_values[root]
    for s in SECTIONS:
        for op in result[s]:
            if op['kind']=='encode':
                root=definition_sources[op['output']]
                assert canonical(dict(op,output=root))==canonical(roots[root])
    for key in plan:
        if key not in (*SECTIONS,'format_version','values'):assert result[key]==plan[key]
    audit=footprint(result)
    assert audit['ciphertext_peak_coefficient_bytes']==deferred_audit['candidate']['ciphertext_peak_coefficient_bytes']
    assert peak<=budget_bytes
    report=dict(passed=True,scope='CPU V3 bounded immutable Encode-root cache',budget_bytes=budget_bytes,
        eviction_policy='farthest_next_use_with_size_tiebreak',
        encoded_cache_peak_bytes=peak,reencodes=reencodes,evictions=evictions,extra_value_ids=len(aliases),
        original_encode_count=len(roots),candidate_encode_count=sum(actual_encodes.values()),
        payload_counts=payload_counts,compute_semantic_sha256=semantic_hash(plan,'compute'),
        normalized_compute_hash_matches=True,all_constant_payloads_identical=True,
        ciphertext_lifetimes_unchanged=True,all_original_value_metadata_unchanged=True,
        boot_count_unchanged=audit['counts'].get('boot',0)==deferred_audit['candidate']['counts'].get('boot',0),
        deferred_baseline=deferred_audit['candidate'],candidate=audit,
        note='Budget covers Encode roots only; report counts derived plaintext separately through full live-set audit. Not measured RSS.')
    assert report['boot_count_unchanged']
    return result,report
