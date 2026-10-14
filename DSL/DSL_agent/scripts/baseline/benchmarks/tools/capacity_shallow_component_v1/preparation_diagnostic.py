from component_contract import prepare_task
from application_planning.layout import plan_model
g=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='broadcast_inner',inputs=[dict(name='x',shape=[2,3,4]),dict(name='y',shape=[1,3,1])],constants={},nodes=[dict(id='add',op='add',inputs=['x','y'],outputs=['v'],attrs={}),dict(id='square',op='square',inputs=['v'],outputs=['z'],attrs={})],outputs=[dict(name='out',value='z')])
o=dict(chunk_period=16,capacity_packing_version='capacity-public-tensors-v10',compiler_configuration='poseidon-cpu-eva-w45-compact-v1',generation_guidance='explicit-v15',public_weight_policy='descriptors-compact-origins-v3')
p=plan_model(g,options=o);print(p['failure']);print(p['candidates'])
prepare_task(g,o)
