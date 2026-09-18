"""Opt-in variable-period ABI. Does not change CKKS security parameters."""
import json
import math

from seal_artifact_gate import require

ABI = 'periodic-packed-v1'
TASK = 'hecate-periodic-packed-synthesis-v1'
CONTRACT = 'hecate-periodic-packed-v1'
NATIVE_TASK = 'hecate-periodic-packed-native-synthesis-v1'
NATIVE_CONTRACT = 'hecate-periodic-packed-native-v1'
NATIVE_EXERCISE_TASK = 'hecate-periodic-packed-native-synthesis-v2'
NATIVE_TASKS = (NATIVE_TASK,NATIVE_EXERCISE_TASK)
TASKS = (TASK,*NATIVE_TASKS)
PERIODS = (4,8,16,32,64,128,256)
MAX_CONSTANT_ELEMENTS = 4096
MAX_HIDDEN_WIDTH = 16
MAX_OPERATIONS = 1024


def binding(shape):
    require(type(shape) is list and 1 <= len(shape) <= 4 and
            all(type(n) is int and 1 <= n <= 256 for n in shape), 'Packed input needs positive rank1..4 shape')
    count=math.prod(shape)
    require(1 <= count <= 256,'Packed input v1 supports 1..256 logical elements')
    period=next(p for p in PERIODS if p>=count)
    return dict(schema=1,kind=ABI,logical_shape=list(shape),logical_elements=count,
                input_order='C-row-major',slot_period=period,padding=0)


def rotations(period):
    require(type(period) is int and period in PERIODS,'Invalid packed slot period')
    return tuple(1<<i for i in range(period.bit_length()-1))


def zero_argument(period):
    rotations(period)
    return dict(dsl_name='zero_ct',kind='encrypted_zero',source='trusted_client',slot_period=period)


def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def validate_layout(layout):
    required={'execution_abi','input_shape','input_slot_period','input_order','model_input_binding',
              'output_shape','output_ciphertexts','output_selectors','output_representation'}
    require(type(layout) is dict and required <= set(layout) <= required|{'auxiliary_ciphertexts'},
            'Invalid packed layout fields')
    plan=binding(layout['input_shape']);period=plan['slot_period']
    require(layout['execution_abi']==ABI and same(layout['model_input_binding'],plan) and
            type(layout['input_slot_period']) is int and layout['input_slot_period']==period and
            layout['input_order']=='C-row-major','Packed input binding changed')
    if 'auxiliary_ciphertexts' in layout:
        require(same(layout['auxiliary_ciphertexts'],[zero_argument(period)]),'Invalid packed encrypted-zero input')
    shape=layout['output_shape'];count=layout['output_ciphertexts'];mode=layout['output_representation']
    require(type(shape) is list and 1<=len(shape)<=4 and all(type(n) is int and 1<=n<=256 for n in shape) and
            type(count) is int,'Invalid packed output shape/count')
    elements=math.prod(shape)
    if mode=='packed_prefix':
        require(1<=elements<=plan['logical_elements'] and count==1,'Invalid packed-prefix output')
        expected=[[0,i] for i in range(elements)]
    else:
        require(mode=='scalar_neurons' and 1<=elements<=16 and count==elements,'Invalid scalar-neuron output')
        expected=[[i,0] for i in range(count)]
    require(same(layout['output_selectors'],expected),'Packed output selectors changed')
    return plan


def validate_request(request):
    model=request.get('model')
    require(type(model) is dict and type(model.get('schema')) is int and model['schema']==5,
            'Packed request requires schema5 original model')
    plan=validate_layout(request['layout'])
    require(same(plan,binding(model.get('input_shape'))),'Packed model/layout shape mismatch')
    return plan


def gate_options(layout):
    plan=validate_layout(layout)
    return dict(execution_abi=ABI,input_period=plan['slot_period'])


def pack_inputs(logical,shape):
    import numpy as np
    plan=binding(shape)
    require(type(logical) is np.ndarray and logical.dtype==np.float64 and
            logical.shape==(4,*shape) and np.isfinite(logical).all(),'Invalid packed logical inputs')
    flat=logical.reshape(4,-1)
    return np.pad(flat,((0,0),(0,plan['slot_period']-flat.shape[1])))


def decode_output(flat,layout):
    """Restore logical output dimensions after declared row-major selection."""
    import numpy as np
    validate_layout(layout)
    require(type(flat) is np.ndarray and flat.dtype==np.float64 and
            flat.shape==(4,math.prod(layout['output_shape'])) and np.isfinite(flat).all(),
            'Invalid flat decoded output')
    return flat.reshape(4,*layout['output_shape'])
