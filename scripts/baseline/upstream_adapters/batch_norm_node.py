"""Bind one mathematical BatchNorm node from any legal DAG to real HE_BN.

The immutable standalone view reuses the checked upstream closure adapter, not
an alternate arithmetic implementation. Candidate code receives only a callee.
"""
import copy,math
from benchmark_graph import FORMAT,validate,require,digest,canonical
from upstream_adapters.batch_norm import bind as standalone_bind,apply as standalone_apply

def bind_node(model,node_id,period):
    check=validate(model)
    require(type(period) is int and period in (4,8,16,32,64,128,256),'BN slot period')
    nodes=[n for n in model['nodes'] if n['id']==node_id]
    require(len(nodes)==1 and nodes[0]['op']=='batch_norm','BN node identity')
    node=nodes[0];shape=check['shapes'][node['inputs'][0]]
    require(math.prod(shape)<=period,'BN intermediate does not fit common ciphertext period')
    require(all(name in model['constants'] for name in node['inputs'][1:]),'BN parameters must be fixed public constants')
    names=node['inputs'][1:];constants={name:copy.deepcopy(model['constants'][name]) for name in names}
    inp='bound_input'
    while inp in constants:inp+='x'
    result='bound_result'
    while result in constants or result==inp:result+='x'
    view=dict(format=FORMAT,id='bound_bn',inputs=[dict(name=inp,shape=list(shape))],constants=constants,
              nodes=[dict(id='bound_node',op='batch_norm',inputs=[inp,*names],attrs=copy.deepcopy(node['attrs']),outputs=[result])],
              outputs=[dict(name='bound_output',value=result)])
    spec=standalone_bind(view)
    require(period%spec['closure_period']==0,'BN closure period mismatch')
    geometry=dict(spec['geometry']);copies=16384//spec['closure_period'];channels=shape[1]
    geometry.update(ho=geometry['hi'],wo=geometry['wi'],ko=1,ti=channels,to=channels,ni=1,no=1,
                    pi=copies,po=copies,q=(channels+copies-1)//copies)
    return dict(schema=1,expected_geometry=geometry,node_id=node_id,input_value=node['inputs'][0],output_value=node['outputs'][0],
                input_shape=list(shape),slot_period=period,closure_period=spec['closure_period'],
                mathematical_view=view,view_sha256=digest(view))

def apply(binding,cipher,helpers,mpcb):
    # The parent request recomputes binding from its validated complete graph.
    require(binding['view_sha256']==digest(binding['mathematical_view']),'BN mathematical view hash')
    result,record=standalone_apply(binding['mathematical_view'],cipher,helpers,mpcb)
    record['returned_to_golden']=False  # The surrounding candidate may discard it.
    record['output_contribution_proven']=False
    require(record['input_shape']==binding['input_shape'] and record['closure_period']==binding['closure_period'],
            'BN actual closure differs from binding')
    require(canonical(record)==canonical(expected_record(binding)),'Actual BN closure record differs')
    return result,record

def expected_record(binding):
    return dict(helper='HE_BN',source='poly/Func.py',closure='MPCB.shapeClosure.BN',
                parameter_derivation='MPCB.abstractBN',geometry=binding['expected_geometry'],
                input_shape=binding['input_shape'],closure_period=binding['closure_period'],slot_count=16384,
                returned_to_golden=False,unchanged_upstream_functions=True,bootstrap_removed=False,
                agent_generated=False,output_contribution_proven=False)
