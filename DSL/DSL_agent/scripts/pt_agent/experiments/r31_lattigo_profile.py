"""Build an independent finite native profile plan from full operand metadata.

This module only constructs protocol dictionaries. It imports no numerical or
cryptographic package and performs no model, key or ciphertext execution.
"""
import copy
from collections import Counter
import json

from .r31_lattigo_binding import NativeBinding


def rotation_cache_inventory(plan, *, capacity=1, logical_slots=14):
    """Static original-order cache schedule; durations remain unmeasured.

    The adapter normalizes steps modulo logical slots before requesting the
    corresponding residual-context Galois key. Full-level key identity is
    independent of the ciphertext level. No native getter is called here.
    """
    if type(capacity) is not int or capacity<1:raise ValueError('Positive LRU capacity required')
    order=[];hits=misses=evictions=zero=0;steps=Counter()
    for instruction in plan['execution']:
        if instruction['op']!='rotate':continue
        step=instruction['attrs']['steps']%(1<<logical_slots)
        if step==0:zero+=1;continue
        steps[step]+=1
        if step in order:
            hits+=1;order.remove(step)
        else:
            misses+=1
            if len(order)>=capacity:order.pop(0);evictions+=1
        order.append(step)
    return dict(capacity=capacity,logical_slots=logical_slots,nonzero_rotation_calls=sum(steps.values()),
        zero_rotation_copies=zero,unique_normalized_steps=len(steps),hits=hits,misses=misses,
        evictions=evictions,terminal_resident_steps=order,step_counts=dict(sorted(steps.items())),
        predicted_keygen_seconds=None,predicted_encoding_seconds=None,
        scope='Static LRU schedule only; native key generation and encoding costs need independent measurements')


def profile_cells(plan):
    """Keep output/target metadata so distinct modulus-switch targets stay apart."""
    values={item['id']:item for item in plan['values']}; rows={}
    def meta(value):
        return {k:value[k] for k in ('kind','level','scale_log2','components')}
    for op in plan['execution']:
        attrs={k:v for k,v in op.get('attrs',{}).items()
               if k in ('target_level','target_scale_log2','target_components','implementation')}
        cell=dict(op=op['op'],inputs=[meta(values[name]) for name in op['inputs']],
                  output=meta(values[op['output']]),attrs=attrs)
        key=json.dumps(cell,sort_keys=True,separators=(',',':'))
        if key not in rows:rows[key]=dict(cell,source_count=0,source_ordinals=[],
            representative_attrs=copy.deepcopy(op.get('attrs',{})))
        rows[key]['source_count']+=1;rows[key]['source_ordinals'].append(op['ordinal'])
    return [dict(index=index,**row) for index,(_,row) in enumerate(sorted(rows.items()))]


def make_profile_plan(original,*,binding=None):
    """One target operation per cell; setup/aggregation operations are separate.

All targets share one original-parameter context and one encrypted input. A
single final sum makes every target observable at the final decrypt. Native
execution and its independent numeric reference are separate future actions.
"""
    binding=binding or NativeBinding();unit=binding.precision_bits
    cells=profile_cells(original)
    if not cells:raise ValueError('The source plan contains no native operations')
    context=original['values'][0]['context'];place=dict(kind='host',rank=0)
    values=[];initialization=[];execution=[];blobs=[];by_id={};bases={};plaintexts={};targets=[]
    next_ordinal=len(cells)*8+100
    def desc(name,level,scale=None,components=2,kind='ciphertext'):
        if scale is None:scale=unit
        value=dict(id=name,kind=kind,context=context,level=level,scale_log2=scale,
                   components=components,ntt=True,place=dict(place))
        values.append(value);by_id[name]=value;return name
    desc('profile-input',5);bases[5]='profile-input'
    def operation(kind,inputs,level,scale=None,components=2,**attrs):
        nonlocal next_ordinal
        name='profile-value-'+str(len(execution));desc(name,level,scale,components)
        item=dict(kind='compute',ordinal=next_ordinal,op=kind,inputs=list(inputs),output=name,place=dict(place))
        next_ordinal+=1
        if attrs:item['attrs']=attrs
        execution.append(item);return name,item
    def base(level):
        if level not in bases:
            bases[level]=operation('mod_switch',['profile-input'],level,target_level=level)[0]
        return bases[level]
    def plain(level,number):
        key=(level,number)
        if key not in plaintexts:
            name='profile-plain-'+str(len(plaintexts));desc(name,level,components=1,kind='plaintext')
            content='profile:'+name
            initialization.append(dict(kind='encode',ordinal=len(initialization),output=name,
                                       payload=dict(kind='bundle',content=content)))
            blobs.append(dict(content=content,byte_length=16384*8,value=number))
            plaintexts[key]=name
        return plaintexts[key]
    def operand(meta):
        if meta['kind']=='plaintext':
            if (meta['scale_log2'],meta['components'])!=(unit,1):
                raise ValueError('Plaintext metadata differs from the explicit profile binding')
            return plain(meta['level'],.125)
        name=base(meta['level'])
        if meta['scale_log2']==2*unit and meta['components']==3:
            name=operation('mul_cc',[name,name],meta['level'],2*unit,3)[0]
        elif meta['scale_log2']==2*unit and meta['components']==2:
            name=operation('mul_cp',[name,plain(meta['level'],.5)],meta['level'],2*unit)[0]
        elif (meta['scale_log2'],meta['components'])!=(unit,2):
            raise ValueError('Unsupported source operand metadata for finite profiling')
        return name
    accumulator=None
    for cell in cells:
        inputs=[operand(meta) for meta in cell['inputs']]
        out=cell['output'];attrs=dict(cell['representative_attrs'])
        name,item=operation(cell['op'],inputs,out['level'],out['scale_log2'],out['components'],**attrs)
        targets.append(dict(cell_index=cell['index'],target_ordinal=item['ordinal'],target_output=name,
                            source_count=cell['source_count']))
        level,scale,components=out['level'],out['scale_log2'],out['components']
        if components==3:name,_=operation('relinearize',[name],level,scale);components=2
        if scale==2*unit:
            name,_=operation('rescale',[name],level-1,unit,target_level=level-1,target_scale_log2=unit)
            level-=1;scale=unit
        if level>1:name,_=operation('mod_switch',[name],1,target_level=1)
        if level<1 or scale!=unit:raise ValueError('Target output cannot enter the final profile sum')
        if accumulator is None:accumulator=name
        else:accumulator,_=operation('add_cc',[accumulator,name],1)
    plan=dict(format_version=1,plan_id=1,target=dict(original['target'],adapter_profile=True),
        values=values,external_inputs=['profile-input'],initialization=initialization,execution=execution,
        finalization=[],final_outputs=[accumulator],plaintext_bundle=dict(
            id='r31-independent-native-cells'+('' if unit==43 else '-precision'+str(unit)),version=1))
    manifest=dict(bundle_format_version=1,bundle_id=plan['plaintext_bundle']['id'],version=1,
                  blobs=[{k:v for k,v in row.items() if k!='value'} for row in blobs])
    metadata=dict(cells=cells,target_operations=targets,constant_vectors=blobs,native_binding=binding.metadata(),
        cell_count=len(cells),target_execution_count=len(targets),
        setup_and_aggregation_count=len(execution)-len(targets),source_execution_count=len(original['execution']),
        source_op_counts=dict(Counter(op['op'] for op in original['execution'])),
        scope='Independent one-context finite target-cell profiling, with setup and final aggregation separately identified; no source-model execution or prediction yet',
        rotations='Representative original step per metadata cell; key generation and cache events must be reported separately',
        residual_rotation_cache=rotation_cache_inventory(original,capacity=1),
        bootstraps='Original target level/scale/components; one native bootstrap target cell; constructor/keygen separately recorded',
        actual_native_executions=0,security_acceptance=None,extra_integrity_checks=0)
    return plan,manifest,metadata
