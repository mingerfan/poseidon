#!/usr/bin/env python3
"""Bind pinned pretrained Qwen3-0.6B weights, two token inputs and dense references."""
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np
from qwen3 import spec, reference
from qwen3.weights import file_hash

REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
WEIGHT_SHA256='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'

def mapping(c):
    result={'emb_weight.weight':'model.embed_tokens.weight','final_normal.weight':'model.norm.weight','lm_head.weight':'lm_head.weight'}
    names={'pre_Normal.weight':'input_layernorm.weight','post_Normal.weight':'post_attention_layernorm.weight',
           'attention.q_weight.weight':'self_attn.q_proj.weight','attention.k_weight.weight':'self_attn.k_proj.weight',
           'attention.v_weight.weight':'self_attn.v_proj.weight','attention.o_weight.weight':'self_attn.o_proj.weight',
           'attention.q_norm.weight':'self_attn.q_norm.weight','attention.k_norm.weight':'self_attn.k_norm.weight',
           'mlplayer.gate_proj.weight':'mlp.gate_proj.weight','mlplayer.up_proj.weight':'mlp.up_proj.weight',
           'mlplayer.down_proj.weight':'mlp.down_proj.weight'}
    for layer in range(c['num_layers']):
        for local,hf in names.items():result[f'model_list.{layer}.{local}']=f'model.layers.{layer}.{hf}'
    return result

def cheby(function,domain,degree):
    lo,hi=domain
    def evaluate(t):
        x=(t+1)*(hi-lo)/2+lo
        if function=='rsqrt':return 1/np.sqrt(x)
        if function=='silu':return x/(1+np.exp(-x))
        raise ValueError(function)
    count=degree+1
    samples=evaluate(np.cos(np.pi*(np.arange(count)+.5)/count))
    # DCT-II via an even FFT extension avoids a quadratic interpolation
    # matrix and roundoff from summing thousands of cosine terms.
    transformed=np.fft.fft(np.concatenate((samples,samples[::-1])))[:count]
    coefficients=(transformed*np.exp(-.5j*np.pi*np.arange(count)/count)).real/count
    coefficients[0]*=.5
    return dict(method='chebyshev',function=function,coefficients=coefficients.tolist(),domain=[lo,hi],evaluation='blocked',block_size=8)

def norms(bounds):
    lo,hi=bounds
    domain=(max(lo*.5,1e-12),max(hi*2,lo*2))
    if domain[1]/domain[0]<=8:
        return cheby('rsqrt',domain,24)
    return dict(method='newton_rsqrt',function='rsqrt',domain=list(domain),
                iterations=math.ceil(math.log(domain[1]/domain[0])/math.log(2.25))+8,
                evaluation='residual',refinement_steps=2)

def silu_profile(bound):
    radius=max(bound*1.5,2)
    return dict(method='normalize_square_silu',function='silu',domain=[-radius,radius],
                squarings=max(0,math.ceil(math.log2(radius))),reciprocal_iterations=8,
                intermediate_iterations=4,exp_degree=12)

def calibrate(bounds):
    rows=[]
    for row in bounds['layers']:
        center=max(row['centered_score_abs_max']*1.5,.125)
        squarings=max(0,math.ceil(math.log2(center/.5)))
        q_norm=norms((min(b[0] for b in row['q_norm_variance']),max(b[1] for b in row['q_norm_variance'])))
        k_norm=norms((min(b[0] for b in row['k_norm_variance']),max(b[1] for b in row['k_norm_variance'])))
        rows.append(dict(pre_norm=norms(row['pre_norm_variance']),post_norm=norms(row['post_norm_variance']),
            q_norm=[q_norm]*len(row['q_norm_variance']),k_norm=[k_norm]*len(row['k_norm_variance']),
            silu=silu_profile(row['gate_abs_max']),
            softmax=dict(method='normalize_square',centered_bound=center,squarings=squarings,reciprocal_iterations=8,
                intermediate_iterations=4,max_length=2,exp_degree=12,exp_bound=.5)))
    return dict(blocks=rows,final_norm=norms(bounds['final_norm_variance']))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,nargs=2,default=[9707,11])
    a=parser.parse_args();checkpoint=a.checkpoint.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    config=json.loads((checkpoint/'config.json').read_text())
    expected={'model_type':'qwen3','num_hidden_layers':28,'hidden_size':1024,'intermediate_size':3072,
              'num_attention_heads':16,'num_key_value_heads':8,'head_dim':128,'vocab_size':151936,
              'attention_bias':False,'tie_word_embeddings':True}
    for key,value in expected.items():
        if config[key]!=value:raise ValueError(f'{key}: {config[key]} != target {value}')
    c=dict(num_layers=config['num_hidden_layers'],hidden_size=config['hidden_size'],intermediate_size=config['intermediate_size'],
           query_heads=config['num_attention_heads'],kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
           vocab_size=config['vocab_size'],max_sequence_length=config['max_position_embeddings'],
           rope_theta=config['rope_theta'],rms_norm_eps=config['rms_norm_eps'])
    path=checkpoint/'model.safetensors'
    if file_hash(path)!=WEIGHT_SHA256:raise ValueError('checkpoint weight SHA-256 mismatch')
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length))
    names=mapping(c);shapes=spec.weight_shapes(c)
    if set(header)-{'__metadata__'}!=set(names.values()):raise ValueError('checkpoint tensor names do not match complete Qwen3 schema')
    out.mkdir();weights_dir=out/'weights';weights_dir.mkdir();arrays={};manifest={'config':c,'arrays':{}}
    for index,(local,hf) in enumerate(names.items()):
        record=header[hf]
        if record['dtype']!='BF16' or tuple(record['shape'])!=shapes[local]:raise ValueError(f'invalid checkpoint tensor {hf}')
        start,end=record['data_offsets'];shape=tuple(record['shape'])
        if end-start!=math.prod(shape)*2:raise ValueError(f'tensor byte range {hf}')
        raw=np.memmap(path,dtype='<u2',mode='r',offset=8+length+start,shape=shape)
        data=(raw.astype(np.uint32)<<16).view(np.float32)
        if not np.isfinite(data).all():raise ValueError(f'nonfinite checkpoint tensor {hf}')
        destination=weights_dir/f'{index:03d}.npy';np.save(destination,data,allow_pickle=False)
        manifest['arrays'][local]={'file':destination.name,'sha256':file_hash(destination),'checkpoint_tensor':hf}
        arrays[local]=np.load(destination,mmap_mode='r')
    if not np.array_equal(arrays['emb_weight.weight'],arrays['lm_head.weight']):raise ValueError('tied embedding/LM head config disagrees with stored weights')
    (weights_dir/'weights.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'model.json').write_text(json.dumps(c,indent=2)+'\n')
    embedding=np.array(arrays['emb_weight.weight'][a.tokens],dtype=np.float64)
    np.save(out/'embeddings.npy',embedding,allow_pickle=False)
    result,bounds=reference.forward(arrays,embedding,[0,1],c)
    np.savez(out/'reference-exact.npz',**result)
    (out/'activation-bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
    profiles=calibrate(bounds);(out/'approximations.json').write_text(json.dumps(profiles,indent=2)+'\n')
    approximate,observed=reference.forward(arrays,embedding,[0,1],c,profiles=profiles)
    np.savez(out/'reference-approximate.npz',**approximate)
    (out/'approximate-activation-bounds.json').write_text(json.dumps(observed,indent=2)+'\n')
    error=approximate['logits']-result['logits']
    tokenizer=json.loads((checkpoint/'tokenizer.json').read_text());inverse={v:k for k,v in tokenizer['model']['vocab'].items()}
    provenance=dict(model='Qwen/Qwen3-0.6B',revision=REVISION,weight_sha256=WEIGHT_SHA256,
        config=config,tensor_count=len(names),stored_parameters=sum(math.prod(r['shape']) for k,r in header.items() if k!='__metadata__'),
        tied_weights_identical=True,tokens=a.tokens,token_strings=[inverse[t] for t in a.tokens],positions=[0,1],
        reference='independent dense NumPy FP64 equations; original rsqrt/SiLU/causal softmax and a separate polynomial reference',
        profile_scope='calibrated on this fixed two-token prefill; not a general input accuracy certificate',
        ckks_tolerance={'absolute':.001,'relative':.0001},
        exact_top1=int(np.argmax(result['logits'])),approximate_top1=int(np.argmax(approximate['logits'])),
        approximate_vs_exact={'logit_max_abs':float(np.max(np.abs(error))),'logit_rms':float(np.sqrt(np.mean(error**2))),
            'cosine':float(np.dot(approximate['logits'],result['logits'])/(np.linalg.norm(approximate['logits'])*np.linalg.norm(result['logits'])))})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2),flush=True)

if __name__=='__main__':main()
