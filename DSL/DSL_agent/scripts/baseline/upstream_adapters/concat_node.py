"""Real HE_Concat with immutable channel closure and periodic rotation lowering.

The helper and patched upstream algorithm are unchanged. Only its right input's
rotate method maps a signed step to existing positive binary-period key steps.
This is exact for P-periodic slots; it cannot be used with nonperiodic values.
"""
import math
from benchmark_graph import validate,require,canonical

SLOTS=16384

def bind_node(model,node_id,period):
    check=validate(model)
    require(type(period) is int and period in (4,8,16,32,64,128,256),'Concat period')
    selected=[n for n in model['nodes'] if n['id']==node_id]
    require(len(selected)==1 and selected[0]['op']=='concat','Concat node identity')
    node=selected[0];refs=node['inputs']
    require(len(refs)==2 and all(r not in model['constants'] for r in refs),'HE_Concat requires two encrypted values')
    shapes=[check['shapes'][r] for r in refs];shape=shapes[0];rank=len(shape)
    require(shapes[0]==shapes[1],'HE_Concat requires equal input shapes')
    axis=node['attrs']['axis']%rank
    require((rank==1 and axis==0) or (2<=rank<=4 and shape[0]==1 and axis==1),
            'HE_Concat requires flat vectors or batch-one channel concat')
    count=math.prod(shape);output_shape=check['shapes'][node['outputs'][0]]
    require(2*count<=period,'Concat intermediate does not fit common ciphertext period')
    channels=shape[0] if rank==1 else shape[1]
    hi=shape[2] if rank==4 else 1;wi=shape[-1] if rank>=3 else 1
    closure=1<<(2*count-1).bit_length();inp_closure=1<<(count-1).bit_length()
    geometry=dict(nt=SLOTS,bb=1.,fh=1,fw=1,s=1,hi=hi,wi=wi,ki=1,ci=channels,co=2*channels,
                  ho=hi,wo=wi,ko=1,ti=channels,to=2*channels,ni=1,no=1,
                  pi=SLOTS//inp_closure,po=SLOTS//closure,q=(2*channels+SLOTS//inp_closure-1)//(SLOTS//inp_closure))
    normalized=(-count)%period
    steps=[1<<i for i in range(period.bit_length()-1) if normalized&(1<<i)]
    return dict(schema=1,kind='concat',node_id=node_id,input_values=list(refs),output_value=node['outputs'][0],
                input_shape=list(shape),output_shape=list(output_shape),logical_input_elements=count,
                slot_period=period,closure_period=closure,expected_geometry=geometry,
                rotation=dict(requested=-count,normalized=normalized,steps=steps,
                              equivalence='P-periodic rotation modulo P then binary composition'))


class PeriodicRotationInput:
    """Narrow, trusted adapter; no candidate can instantiate this type."""
    def __init__(self,value,binding,records):self.value=value;self.binding=binding;self.records=records
    def rotate(self,step):
        expected=self.binding['rotation']
        require(type(step) is int and step==expected['requested'],'Unexpected upstream concat rotation')
        result=self.value
        for part in expected['steps']:result=result.rotate(part)
        self.records.append(dict(expected))
        return result


def expected_record(binding):
    return dict(helper='HE_Concat',source='poly/Func.py',closure='MPCB.shapeClosure.CC',
                geometry=binding['expected_geometry'],input_shape=binding['input_shape'],
                output_shape=binding['output_shape'],closure_period=binding['closure_period'],
                slot_count=SLOTS,rotations=[binding['rotation']],
                returned_to_golden=False,unchanged_upstream_functions=True,bootstrap_removed=False,
                agent_generated=False,output_contribution_proven=False)


def apply(binding,left,right,helpers,mpcb):
    import numpy as np
    keys=('nt','bb','fh','fw','s','hi','wi','ki','ci','co')
    actual=mpcb.InferShapes({k:binding['expected_geometry'][k] for k in keys})
    require(canonical(actual)==canonical(binding['expected_geometry']),'Actual concat closure geometry changed')
    records=[];proxy=PeriodicRotationInput(right,binding,records)
    result=helpers.HE_Concat(mpcb.shapeClosure(**actual),np.array([left],dtype=object),np.array([proxy],dtype=object))
    require(type(result) is np.ndarray and result.shape==(1,),'Actual HE_Concat whole-Expr result')
    record=expected_record(binding);record['geometry']=actual;record['rotations']=records
    require(canonical(record)==canonical(expected_record(binding)),'Actual concat invocation differs')
    return result[0],record
