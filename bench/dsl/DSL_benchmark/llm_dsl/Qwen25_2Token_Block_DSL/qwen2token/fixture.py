"""Reproduce the historical synthetic inputs/weights and an exact NumPy reference."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def config():
    return json.loads((ROOT/'config/model.json').read_text(encoding='utf-8'))

def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()

def shapes():
    c=config(); h,f,k=c['hidden_size'],c['intermediate_size'],c['kv_heads']*c['head_dim']
    # Preserve the exact historical weight_shapes iteration order, including the
    # unused final norm RNG draw before the block weights.
    result={'final_normal.weight': (h,)}
    result.update({'model_list.0.'+n:s for n,s in {
        'pre_Normal.weight':(h,), 'post_Normal.weight':(h,),
        'attention.q_weight.weight':(h,h), 'attention.q_weight.bias':(h,),
        'attention.k_weight.weight':(k,h), 'attention.k_weight.bias':(k,),
        'attention.v_weight.weight':(k,h), 'attention.v_weight.bias':(k,),
        'attention.o_weight.weight':(h,h), 'mlplayer.gate_proj.weight':(f,h),
        'mlplayer.up_proj.weight':(f,h), 'mlplayer.down_proj.weight':(h,f)
    }.items()})
    return result

def generate():
    rng=np.random.default_rng(20260922)
    weights={}
    for name,shape in shapes().items():
        if 'Normal.weight' in name or name=='final_normal.weight':
            weights[name]=rng.uniform(.9,1.1,size=shape)
        else:
            weights[name]=rng.normal(scale=.15/math.sqrt(shape[-1]) if len(shape)==2 else .02,size=shape)
    x=rng.normal(size=(2,896))
    x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True))
    return weights,x

def reference(weights,x):
    c=config(); seq=2; positions=np.array([7,8]); dim=c['head_dim']
    w=lambda n:weights['model_list.0.'+n]
    def norm(v,g):
        variance=np.mean(v*v,axis=-1,keepdims=True)+c['rms_norm_eps']
        if not np.all((variance>=.5)&(variance<=2.)):
            raise ValueError('RMS variance outside frozen [.5,2] profile')
        return v/np.sqrt(variance)*g
    n=norm(x,w('pre_Normal.weight'))
    def rope(v,heads):
        v=v.reshape(seq,heads,dim)
        angles=np.outer(positions,1/(c['rope_theta']**(np.arange(0,dim,2)/dim)))
        angles=np.tile(angles,(1,2))[:,None,:]
        return v*np.cos(angles)+np.concatenate((-v[:,:,dim//2:],v[:,:,:dim//2]),axis=-1)*np.sin(angles)
    q=rope(n@w('attention.q_weight.weight').T+w('attention.q_weight.bias'),c['query_heads'])
    k=rope(n@w('attention.k_weight.weight').T+w('attention.k_weight.bias'),c['kv_heads'])
    v=(n@w('attention.v_weight.weight').T+w('attention.v_weight.bias')).reshape(seq,c['kv_heads'],dim)
    attended=[]; max_centered=0.
    for i in range(seq):
        heads=[]
        for h in range(c['query_heads']):
            kv=h//(c['query_heads']//c['kv_heads'])
            scores=k[:i+1,kv]@q[i,h]/math.sqrt(dim)
            max_centered=max(max_centered,float(np.max(np.abs(scores-scores.mean()))))
            p=np.exp(scores-scores.max()); p/=p.sum()
            heads.append(p@v[:i+1,kv])
        attended.append(w('attention.o_weight.weight')@np.concatenate(heads))
    residual=x+np.asarray(attended)
    post=norm(residual,w('post_Normal.weight'))
    gate=post@w('mlplayer.gate_proj.weight').T
    gate_max=float(np.max(np.abs(gate)))
    if max_centered>.5 or gate_max>2:
        raise ValueError('Attention/SiLU outside frozen profile')
    ffn=(gate/(1+np.exp(-gate))*(post@w('mlplayer.up_proj.weight').T))@w('mlplayer.down_proj.weight').T
    return residual+ffn,dict(observed_centered_bound=max_centered,observed_gate_abs_max=gate_max)

def prepare(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new fixture directory; existing files are preserved')
    weights,x=generate()
    target,bounds=reference(weights,x)
    frozen=np.load(ROOT/'data/reference_hidden.npy',allow_pickle=False)
    np.testing.assert_allclose(target,frozen,rtol=0,atol=1e-12)
    frozen_inputs=ROOT/'data/input_hidden.npy'
    if frozen_inputs.exists():
        np.testing.assert_allclose(x,np.load(frozen_inputs,allow_pickle=False),rtol=0,atol=1e-14)
    output.mkdir(parents=True)
    with (output/'weights.npz').open('xb') as stream:
        np.savez(stream,**weights)
    packed=np.zeros((2,32768),dtype=np.float64)
    packed[:,:896]=x
    packed[:,896:898]=[.3,-.2]
    for name,value in [('input_hidden.npy',x),('input_slots.npy',packed),('reference_hidden.npy',target)]:
        with (output/name).open('xb') as stream:
            np.save(stream,value)
    record=dict(seed=20260922,positions=[7,8],numpy_version=np.__version__,synthetic_weights=True,
        input_shape=[2,896],physical_slot_shape=[2,32768],input_tail_probe=[.3,-.2],
        input_sha256=array_sha(x),reference_sha256=array_sha(target),
        weight_arrays={n:dict(shape=list(a.shape),sha256=array_sha(a)) for n,a in weights.items()},
        bounds=bounds,reference_difference_from_frozen=float(np.max(np.abs(target-frozen))))
    with (output/'fixture_manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2)
    return record
