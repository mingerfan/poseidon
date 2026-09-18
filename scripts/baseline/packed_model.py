"""Schema5 bounded logical models for one variable-period encrypted input."""
import math

from packed_input_abi import binding, MAX_CONSTANT_ELEMENTS, MAX_HIDDEN_WIDTH
from seal_artifact_gate import require
from spatial_ops import OPS as SPATIAL_OPS

OPS=frozenset(('add','subtract','multiply','negate','square','power','flatten','linear','reshape','batch_norm','concat','permute','transpose'))|SPATIAL_OPS


def broadcast_to_shape(plain_shape,cipher_shape):
    """Plain trailing-axis broadcasting, without enlarging the ciphertext shape."""
    return (len(plain_shape)<=len(cipher_shape) and
            all(a==1 or a==b for a,b in zip(reversed(plain_shape),reversed(cipher_shape))))


def validate(data):
    from model_graph import NAME,CASE_ID,array_shape
    require(type(data) is dict and set(data)=={'schema','id','input_shape','constants','nodes','output'}
            and type(data['schema']) is int and data['schema']==5,'Invalid packed model fields')
    plan=binding(data['input_shape'])
    require(type(data['id']) is str and CASE_ID.fullmatch(data['id']),'Invalid model id')
    constants=data['constants'];nodes=data['nodes']
    require(type(constants) is dict and len(constants)<=32,'Public constant count limit')
    values={'x':('cipher',tuple(data['input_shape']),True)}
    for name,value in constants.items():
        require(type(name) is str and NAME.fullmatch(name) and name not in values,'Invalid public constant name')
        values[name]=('plain',array_shape(value,max_elements=MAX_CONSTANT_ELEMENTS),False)
    require(type(nodes) is list and 1<=len(nodes)<=64,'Graph node count limit')
    for node in nodes:
        require(type(node) is dict and type(node.get('op')) is str and node['op'] in OPS,'Unsupported packed model operator')
        op=node['op'];extra={'weight','bias'} if op=='linear' else {'exponent'} if op=='power' else {'shape'} if op=='reshape' else set()
        if op=='batch_norm':extra={'running_mean','running_var','weight','bias','eps'}
        if op=='concat':extra={'axis'}
        if op=='permute':extra={'dims'}
        if op=='transpose':extra={'dim0','dim1'}
        if op in SPATIAL_OPS:
            extra=({'weight','bias','stride','padding'} | (set(node)&{'dilation','groups'}) if op.startswith('conv')
                   else {'kernel','stride','padding','count_include_pad'})
        require(set(node)=={'id','op','inputs'}|extra,'Unexpected packed operator fields')
        name,args=node['id'],node['inputs']
        require(type(name) is str and NAME.fullmatch(name) and name not in values,'Invalid graph node name')
        require(type(args) is list and (1<=len(args)<=8 if op=='concat' else len(args)==(2 if op in ('add','subtract','multiply') else 1))
                and all(type(a) is str and a in values for a in args),'Invalid/forward graph input')
        left=values[args[0]]
        require(left[0]=='cipher','Packed operation requires ciphertext')
        if op in ('permute','transpose'):
            from tensor_permutation import axes,transpose_axes
            dims=axes(left[1],node['dims']) if op=='permute' else transpose_axes(left[1],node['dim0'],node['dim1'])
            result=('cipher',tuple(left[1][d] for d in dims),left[2])
        elif op=='batch_norm':
            from batch_norm_ops import coefficients
            parameters=[]
            for field in ('running_mean','running_var','weight','bias'):
                key=node[field]
                require(key is None and field in ('weight','bias') or type(key) is str and key in constants,
                        'Invalid BatchNorm public '+field)
                parameters.append(None if key is None else constants[key])
            coefficients(left[1],*parameters,node['eps'],max_elements=256)
            result=left
        elif op=='concat':
            from concat_ops import geometry
            require(all(values[a][0]=='cipher' for a in args),'Concat requires ciphertext inputs')
            shape,_=geometry([values[a][1] for a in args],node['axis'],max_elements=16)
            result=('cipher',shape,False)
        elif op in SPATIAL_OPS:
            from packed_spatial import geometry
            ws=None;bias=None
            if op.startswith('conv'):
                w,b=node['weight'],node['bias']
                require(type(w) is str and w in constants,'Invalid Conv public weight')
                ws=values[w][1];kernel=ws[2:]
                require(len(ws)==(3 if op=='conv1d' else 4),'Invalid Conv weight rank')
                require(b is None or type(b) is str and b in constants and values[b][1]==(ws[0],),'Invalid Conv bias')
            else:kernel=node['kernel']
            output_shape=geometry(op,left[1],ws,kernel,node['stride'],node['padding'],node.get('count_include_pad',True),
                dilation=node.get('dilation'),groups=node.get('groups',1))
            result=('cipher',output_shape,False)
        elif op=='flatten':
            result=('cipher',(math.prod(left[1]),),left[2])
        elif op=='reshape':
            from logical_reshape import reshape_shape
            result=('cipher',reshape_shape(left[1],node['shape'],max_elements=256),left[2])
        else:
            result=left
            if op=='linear':
                w,b=node['weight'],node['bias']
                require(type(w) is str and w in constants,'Invalid Linear public weight')
                ws=values[w][1]
                require(len(ws)==2 and ws[1]==left[1][-1] and 1<=ws[0]<=MAX_HIDDEN_WIDTH,'Invalid Linear shape')
                require(b is None or type(b) is str and b in constants and values[b][1]==(ws[0],),'Invalid Linear bias')
                result=('cipher',(*left[1][:-1],ws[0]),False)
                require(math.prod(result[1])<=MAX_HIDDEN_WIDTH,'Scalar-neuron output budget exceeded by batched Linear')
            elif op in ('add','subtract','multiply'):
                right=values[args[1]]
                require(right[1:]==left[1:] if right[0]=='cipher' else broadcast_to_shape(right[1],left[1]),
                        'Packed binary shape/layout mismatch')
            elif op=='power':
                require(type(node['exponent']) is int and node['exponent'] in (2,4),'Only powers2/4 supported')
        values[name]=result
    output=data['output']
    require(type(output) is str and output in values and values[output][0]=='cipher' and 1<=len(values[output][1])<=4,
            'Output must be a rank1..4 ciphertext tensor')
    result=values[output]
    require(1<=math.prod(result[1])<=(256 if result[2] else 16),'Packed output size limit')
    return dict(schema=5,operators=sorted({n['op'] for n in nodes}),output_shape=list(result[1]),
                input_slot_period=plan['slot_period'],compiled=False,encrypted_execution=False)


def test_inputs(shape):
    import numpy as np
    n=binding(shape)['logical_elements']
    return np.stack([np.zeros(n),np.array([(-1 if i%2 else 1)*(i+1)/(n+1) for i in range(n)]),
        np.random.default_rng(42).uniform(-1,1,n),np.array([-1 if i%2 else 1 for i in range(n)])
        ]).astype(np.float64).reshape(4,*shape)
