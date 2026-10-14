"""Two-token full-size Qwen block pilot, with synthetic weights and disk vectors."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import export_qwen_layer_probe as tr
from export_qwen_block_probe import schedule


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


class VectorGraph(list):
    def __init__(self, stream):
        super().__init__()
        self.stream=stream

    def store(self, values):
        values=np.asarray(values,dtype='<f8')
        mask=values!=0
        boundaries=np.flatnonzero(np.diff(np.r_[False,mask,False]))
        segments=[]
        for start,end in boundaries.reshape(-1,2):
            offset=self.stream.tell()
            self.stream.write(values[start:end].tobytes())
            segments.append([int(start),offset,int(end-start)])
        return dict(size=len(values),segments=segments)

    def append(self,node):
        node=dict(node,expected=self.store(node['expected']))
        if isinstance(node.get('constant'),list):
            node['constant']=self.store(node['constant'])
        super().append(node)
        if len(self)%10000==0:
            print('trace nodes',len(self),'vector bytes',self.stream.tell(),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',choices=('smoke','qwen'),required=True)
    parser.add_argument('--mode',choices=('prefill','decode'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap-config',type=Path,required=True)
    args=parser.parse_args()
    if sys.byteorder!='little':
        raise RuntimeError('diagnostic vector files require little-endian doubles')
    tr.SLOTS=32768
    ql=tr.load_layers()
    qm=importlib.import_module('_qwen_ckks_layer_probe.qwen_model')
    cfg=(qm.QwenConfig(num_layers=1) if args.shape=='qwen' else
         qm.QwenConfig(num_layers=1,hidden_size=4,intermediate_size=8,
                       query_heads=2,kv_heads=1,head_dim=2,vocab_size=8))
    def fit(function,fn,domain):
        p=np.polynomial.Chebyshev.interpolate(fn,12,domain=domain)
        grid=np.linspace(*domain,8193)
        if np.max(np.abs(p(grid)-fn(grid)))>1e-5:
            raise AssertionError(function+' approximation failed')
        return ql.ChebyshevApproximation(function,tuple(p.coef),domain,evaluation='blocked')
    rsqrt=fit('rsqrt',lambda x:1/np.sqrt(x),(.5,2.))
    silu=fit('silu',lambda x:x/(1+np.exp(-x)),(-2.,2.))
    soft=ql.NormalizeSquareSoftmax(.5,0,6,2,exp_degree=8,exp_bound=.5)
    profile=qm.BlockApproximations(rsqrt,rsqrt,silu,softmax=soft)
    profiles=qm.ModelApproximations((profile,),rsqrt)
    rng=np.random.default_rng(20260922)
    weights={}
    for name,shape in qm.weight_shapes(cfg).items():
        if name in ('emb_weight.weight','lm_head.weight'):
            # These tensors are not used by the isolated decoder block.
            weights[name]=np.broadcast_to(np.float64(0),shape)
        elif 'Normal.weight' in name or name=='final_normal.weight':
            weights[name]=rng.uniform(.9,1.1,size=shape)
        else:
            weights[name]=rng.normal(scale=.15/math.sqrt(shape[-1]) if len(shape)==2 else .02,size=shape)
    x=rng.normal(size=(2,cfg.hidden_size))
    x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True))
    positions=(7,8)
    w=lambda name:weights['model_list.0.'+name]
    def norm(v,gain):
        variance=np.mean(v*v,axis=-1,keepdims=True)+cfg.rms_norm_eps
        if not np.all((variance>=rsqrt.domain[0])&(variance<=rsqrt.domain[1])):
            raise AssertionError('variance outside fixed profile')
        return v/np.sqrt(variance)*gain
    n=norm(x,w('pre_Normal.weight'))
    q=n@w('attention.q_weight.weight').T+w('attention.q_weight.bias')
    k=n@w('attention.k_weight.weight').T+w('attention.k_weight.bias')
    v=(n@w('attention.v_weight.weight').T+w('attention.v_weight.bias')).reshape(2,cfg.kv_heads,cfg.head_dim)
    def rope(a,heads):
        a=a.reshape(2,heads,cfg.head_dim)
        angles=np.outer(positions,1/(cfg.rope_theta**(np.arange(0,cfg.head_dim,2)/cfg.head_dim)))
        angles=np.tile(angles,(1,2))[:,None,:]
        half=cfg.head_dim//2
        return a*np.cos(angles)+np.concatenate((-a[:,:,half:],a[:,:,:half]),axis=-1)*np.sin(angles)
    q,k=rope(q,cfg.query_heads),rope(k,cfg.kv_heads)
    attended=[]
    max_centered=0.
    for i in range(2):
        heads=[]
        for h in range(cfg.query_heads):
            kv=h//(cfg.query_heads//cfg.kv_heads)
            scores=k[:i+1,kv]@q[i,h]/math.sqrt(cfg.head_dim)
            max_centered=max(max_centered,float(np.max(np.abs(scores-scores.mean()))))
            p=np.exp(scores-scores.max());p/=p.sum()
            heads.append(p@v[:i+1,kv])
        attended.append(w('attention.o_weight.weight')@np.concatenate(heads))
    if max_centered>soft.centered_bound:
        raise AssertionError('scores outside fixed profile')
    residual=x+np.asarray(attended)
    post=norm(residual,w('post_Normal.weight'))
    gate=post@w('mlplayer.gate_proj.weight').T
    if np.max(np.abs(gate))>2:
        raise AssertionError('gate outside fixed profile')
    ffn=(gate/(1+np.exp(-gate))*(post@w('mlplayer.up_proj.weight').T))@w('mlplayer.down_proj.weight').T
    target=(residual+ffn).ravel()
    model=qm.Qwen25Model(weights,profiles,config=cfg,slots=tr.SLOTS,logit_chunk_size=8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    vector_path=args.output.with_suffix('.vectors.bin')
    with vector_path.open('xb') as vector_stream:
        graph=VectorGraph(vector_stream)
        packed=tuple(ql.PackedVector(tr.Expr(graph,'input',np.r_[row,[.3,-.2]]),cfg.hidden_size,tr.SLOTS) for row in x)
        empty=qm.LayerKVCache((),(),())
        if args.mode=='prefill':
            outputs,_=model._decoder_block(packed,positions,empty,0)
        else:
            first,cache=model._decoder_block(packed[:1],positions[:1],empty,0)
            second,_=model._decoder_block(packed[1:],positions[1:],cache,0)
            outputs=first+second
        output=ql.concatenate(outputs)
        err=float(np.max(np.abs(output.expr.values[:len(target)]-target)))
        if not np.isfinite(output.expr.values).all() or err>1e-5:
            raise AssertionError(('plaintext approximation failed',err))
        np.testing.assert_allclose(output.expr.values[len(target):],0,atol=1e-10)
    nodes,last=schedule(graph,output.expr.index,8,5)
    boots=sum(n['op']=='bootstrap' for n in nodes)
    steps={n['steps'] for n in nodes if n['op']=='rotate'}
    hashes={name:digest(tr.SOURCE/name) for name in ('qwen_layers.py','qwen_model.py','qwen_nonlinear.py','ops.py')}
    boot=json.loads(args.bootstrap_config.read_text())
    boot['output_level']=5
    program=dict(name=f'{args.shape}_block_{args.mode}',packing='prefix',slots=tr.SLOTS,nodes=nodes,
        output=last,target=target.tolist(),output_size=len(target),zero_output_tail=True,max_abs_error=1e-3,
        planned_bootstraps=boots,plaintext_approximation_error=err)
    data=dict(format='qwen-ckks-diagnostic-dag-v1',probe_kind='two_token_fullsize_block_pilot',
        context=dict(log_n=16,scale_bits=boot['scale_bits'],q_bits=boot['q_bits'],p_bits=boot['p_bits'],
                     initial_level=8,max_scale_snap_relative=1e-5,power_of_two_rotation_keys=True),
        bootstrap=boot,vector_file=vector_path.name,vector_format='little_endian_float64_segments',
        vector_sha256=digest(vector_path),candidate_sha256=hashlib.sha256(json.dumps(profiles.to_dict(),sort_keys=True).encode()).hexdigest(),
        kernel_sha256=hashes['qwen_layers.py'],source_sha256=hashes,synthetic_weights=True,
        contract=dict(config=asdict(cfg),profiles=profiles.to_dict(),positions=positions,
                      observed_centered_bound=max_centered,observed_gate_abs_max=float(np.max(np.abs(gate)))),
        programs=[program])
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,allow_nan=False,separators=(',',':'))
    summary=dict(name=program['name'],source_sha256=hashes,fixture_sha256=digest(args.output),
        vector_file=vector_path.name,vector_sha256=data['vector_sha256'],vector_bytes=vector_path.stat().st_size,
        node_count=len(nodes),planned_bootstraps=boots,distinct_rotation_steps=len(steps),
        power_of_two_key_steps=30,plaintext_approximation_error=err,contract=data['contract'])
    with args.output.with_suffix('.summary.json').open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
