"""Experimental CPU V3 post-pass. Only defer Encode; preserve all arithmetic.

This is not the compiler's GPU streaming pass. Native V3 Runtime performs all
execution. One rank, zero GPU, Host-only plans with explicit Release are accepted.
"""
import copy
import hashlib
import json

SECTIONS=('initialization','execution','finalization')

def canonical(op):
    return {k:v for k,v in op.items() if k!='ordinal'}

def semantic_hash(plan, kind):
    h=hashlib.sha256()
    for section in SECTIONS:
        for op in plan[section]:
            if op['kind']==kind:
                h.update(json.dumps(canonical(op),sort_keys=True,separators=(',',':')).encode())
                h.update(b'\n')
    return h.hexdigest()

def footprint(plan):
    values={v['id']:v for v in plan['values']}
    assert len(values)==len(plan['values'])
    size=lambda v:65536*8*(v['level']+1)*v['components']
    live=set(plan['external_inputs']);external=set(live)
    resident=sum(size(values[i]) for i in live);peak=resident
    pt=0;pt_peak=0;ct_peak=resident
    encoded=0;initial_encoded=0;counts={};distribution={}
    for section in SECTIONS:
        for op in plan[section]:
            key=op.get('op',op['kind']);counts[key]=counts.get(key,0)+1
            if op['kind']=='release':
                v=op['value'];assert v in live;live.remove(v)
                if v not in external:
                    resident-=size(values[v])
                    if values[v]['kind']=='plaintext':pt-=size(values[v])
            else:
                assert all(v in live for v in op.get('inputs',[]))
                v=op['output'];assert v not in live;live.add(v);resident+=size(values[v])
                if values[v]['kind']=='plaintext':pt+=size(values[v])
                if op['kind']=='encode':
                    encoded+=size(values[v])
                    if section=='initialization':initial_encoded+=size(values[v])
                if op.get('op')=='boot':
                    src,dst=values[op['inputs'][0]],values[v]
                    assert (src['level'],src['scale_log2'],src['components'])==(1,43,2)
                    assert (dst['level'],dst['scale_log2'],dst['components'])==(5,43,2)
                peak=max(peak,resident);pt_peak=max(pt_peak,pt);ct_peak=max(ct_peak,resident-pt)
    assert live==set(plan['final_outputs']) and counts.get('release',0)>0
    for v in values.values():
        key=f"{v['kind']}:L{v['level']}:c{v['components']}"
        row=distribution.setdefault(key,dict(count=0,coefficient_bytes=0))
        row['count']+=1;row['coefficient_bytes']+=size(v)
    return dict(release_peak_coefficient_bytes=peak,plaintext_peak_coefficient_bytes=pt_peak,
        ciphertext_peak_coefficient_bytes=ct_peak,eager_initialization_encode_bytes=initial_encoded,
        total_encoded_coefficient_bytes=encoded,counts=counts,by_level=distribution,
        input_levels=[values[v]['level'] for v in plan['external_inputs']],
        output_levels=[values[v]['level'] for v in plan['final_outputs']],
        coefficients_only=True,excludes='keys, bootstrap/encoder workspace, allocator, plan and raw bundle objects')

def transform(plan):
    assert plan['format_version'] in (1,2,3)
    assert plan['target']['world_size']==1 and plan['target']['device_counts']==[0]
    assert all(v['place']==dict(kind='host',rank=0) for v in plan['values'])
    assert all(o['kind']=='encode' for o in plan['initialization'])
    assert all(o['kind'] in ('compute','release') for o in plan['execution'])
    assert all(o['kind']=='release' for o in plan['finalization'])
    assert not any('reuse_input' in o for o in plan['execution'])
    assert all(o['output'] not in plan['final_outputs'] for o in plan['initialization'])
    before=footprint(plan)
    # Shallow copy sections; instruction dictionaries are copied when ordinals change.
    result=dict(plan);pending={o['output']:o for o in plan['initialization']}
    assert len(pending)==len(plan['initialization'])
    execution=[];unused_before_release=0
    for op in plan['execution']:
        needed=op.get('inputs',[])
        if op['kind']=='release' and op['value'] in pending:
            needed=[op['value']];unused_before_release+=1
        for v in needed:
            if v in pending:execution.append(pending.pop(v))
        execution.append(op)
    # Unused plaintext released only in finalization is initialized immediately
    # before finalization, keeping every original definition and release intact.
    for op in plan['finalization']:
        if op['value'] in pending:
            execution.append(pending.pop(op['value']));unused_before_release+=1
    assert not pending,'Unconsumed original Encode definitions'
    result.update(format_version=3,initialization=[],execution=execution)
    ordinal=0
    for section in SECTIONS:
        revised=[]
        for op in result[section]:
            revised.append(dict(op,ordinal=ordinal));ordinal+=1
        result[section]=revised
    for key in plan:
        if key not in (*SECTIONS,'format_version'):assert result[key]==plan[key],key
    assert semantic_hash(result,'compute')==semantic_hash(plan,'compute')
    assert semantic_hash(result,'release')==semantic_hash(plan,'release')
    original_encodes={o['output']:canonical(o) for o in plan['initialization']}
    new_encodes={o['output']:canonical(o) for o in result['execution'] if o['kind']=='encode'}
    assert new_encodes==original_encodes
    after=footprint(result)
    assert before['counts']==after['counts']
    assert before['by_level']==after['by_level']
    assert after['release_peak_coefficient_bytes']<=before['release_peak_coefficient_bytes']
    report=dict(passed=True,scope='CPU-only Encode deferral post-pass; native V3 Runtime',
        arithmetic_unchanged=True,values_and_levels_unchanged=True,encodes_unchanged=True,
        releases_unchanged=True,boot_placement_unchanged=True,no_compute_reorder=True,
        original_format=plan['format_version'],format_version=3,
        compute_semantic_sha256=semantic_hash(plan,'compute'),
        release_semantic_sha256=semantic_hash(plan,'release'),
        moved_encodes=len(original_encodes),unused_encodes=unused_before_release,
        baseline=before,candidate=after)
    return result,report
