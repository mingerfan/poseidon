"""Uninstalled draft: preserve return metadata using a direct downward switch.

Only terminal down-switch / native Boot / down-switch chains are eligible.
Never raises a source's level and never rewrites arithmetic consumers.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

SECTIONS=('initialization','execution','finalization')


def transform(plan):
    assert plan['target']['world_size']==1 and plan['target']['device_counts']==[0]
    assert all(o['kind']=='encode' for o in plan['initialization'])
    assert all(o['kind'] in ('compute','release') for o in plan['execution'])
    assert all(o['kind']=='release' for o in plan['finalization'])
    assert all('reuse_input' not in o for s in SECTIONS for o in plan[s])
    before_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    values={v['id']:v for v in plan['values']}
    nodes={o['output']:o for o in plan['execution'] if o['kind']=='compute'}
    users={}
    for op in nodes.values():
        for value in op['inputs']:users.setdefault(value,[]).append(op['output'])
    finals=list(plan['final_outputs']);assert len(finals)==len(set(finals))
    removed=set();rewired={};candidates=[]
    for i,final in enumerate(finals):
        end=nodes.get(final,{})
        if end.get('op')!='mod_switch':continue
        boot=nodes.get(end['inputs'][0],{})
        if boot.get('op')!='boot' or boot.get('attrs',{}).get('implementation')!='native':continue
        down=nodes.get(boot['inputs'][0],{})
        if down.get('op')!='mod_switch':continue
        source=down['inputs'][0]
        if source in finals or source in removed:continue
        source_meta={k:v for k,v in values[source].items() if k!='id'}
        final_meta={k:v for k,v in values[final].items() if k!='id'}
        if source_meta['kind']!='ciphertext' or source_meta['level']<final_meta['level']:continue
        if {k:v for k,v in source_meta.items() if k!='level'}!={k:v for k,v in final_meta.items() if k!='level'}:continue
        if users.get(boot['output'])!=[final] or users.get(down['output'])!=[boot['output']]:continue
        if users.get(final):continue
        chain={final,boot['output'],down['output']}
        if chain & set(finals)!={final}:continue
        assert not chain & removed
        if source_meta['level']==final_meta['level']:
            finals[i]=source;gone=chain;mode='equal_metadata_bypass'
        else:
            # Keep the original returned ValueId, attrs and metadata. Only
            # its mod_switch input changes to a source with enough primes.
            rewired[final]=source;gone=chain-{final};mode='direct_downward_switch'
        removed.update(gone)
        candidates.append(dict(output_index=i,original_output=final,replacement_output=finals[i],
            removed_boot=boot['output'],removed_values=sorted(gone,key=int),
            source=source,source_level=source_meta['level'],metadata=final_meta,mode=mode))
    result=deepcopy(plan);result['final_outputs']=finals
    result['values']=[v for v in result['values'] if v['id'] not in removed]
    ordered=[deepcopy(o) for o in plan['execution'] if o['kind']=='compute' and o['output'] not in removed]
    for op in ordered:
        if op['output'] in rewired:
            assert op['op']=='mod_switch';op['inputs']=[rewired[op['output']]]
    # A source now survives until its direct return switch. Rebuild all
    # releases at true last use rather than retaining a stale early release.
    remaining=Counter(v for o in ordered for v in o['inputs'])
    live=set(plan['external_inputs'])|{o['output'] for o in plan['initialization']}
    final_set=set(finals);assert live<=set(remaining)|final_set
    execution=[]
    for op in ordered:
        assert all(v in live for v in op['inputs']) and op['output'] not in live
        live.add(op['output']);execution.append(op)
        for value in op['inputs']:
            remaining[value]-=1
            if remaining[value]==0 and value not in final_set:
                execution.append(dict(kind='release',value=value));live.remove(value)
        if remaining[op['output']]==0 and op['output'] not in final_set:
            execution.append(dict(kind='release',value=op['output']));live.remove(op['output'])
    assert live==final_set and not any(remaining.values())
    result['execution']=execution;result['finalization']=[]
    ordinal=0
    for section in SECTIONS:
        for op in result[section]:op['ordinal']=ordinal;ordinal+=1
    strip=lambda o:{k:v for k,v in o.items() if k!='ordinal'}
    before=[strip(o) for o in plan['execution'] if o['kind']=='compute' and o['output'] not in removed]
    after=[strip(o) for o in ordered]
    for old,new in zip(before,after):
        if old['output'] in rewired:
            expected=dict(old,inputs=[rewired[old['output']]])
            assert new==expected and old['op']=='mod_switch'
        else:assert old==new
    for old,new in zip(plan['final_outputs'],finals):
        assert {k:v for k,v in values[old].items() if k!='id'}=={k:v for k,v in values[new].items() if k!='id'}
    assert before_hash==hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    return result,dict(candidate_count=len(candidates),candidates=candidates,
        removed_value_count=len(removed),returned_metadata_unchanged=True,
        remaining_arithmetic_unchanged=True,only_mod_switch_inputs_rewired=True,
        original_input_not_mutated=True,releases_rebuilt=True,
        scope='Uninstalled candidate; requires native PlanVerifier, Mock and CKKS')
