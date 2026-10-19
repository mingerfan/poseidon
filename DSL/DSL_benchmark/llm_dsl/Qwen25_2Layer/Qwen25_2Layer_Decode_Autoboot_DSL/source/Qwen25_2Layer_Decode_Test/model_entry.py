"""Two real-width Decoder Blocks, two-token decode, optional final norm/full LM head.

Uses the frozen Qwen arithmetic and the independently validated packed Linear.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'Qwen25_24Layer_DSL'))
sys.path.insert(0,str(ROOT.parent/'Qwen_DSL_Memory_Study'))

def configuration():
    c=json.loads((ROOT.parent/'Qwen25_24Layer_DSL/config/model.json').read_text())
    c['num_layers']=2
    return c

def profiles():
    p=json.loads((ROOT.parent/'Qwen25_24Layer_DSL/config/approximations.provisional.json').read_text())
    p['blocks']=p['blocks'][:2]
    return p

def model(weights):
    from qwen24.kernels import qwen_model as qm
    from qwen24.kernels import qwen_layers as ql
    from qwen24.weights import load
    import packed_linear
    ql.linear=packed_linear.linear
    import packed_lm_head
    ql.lm_head=packed_lm_head.lm_head
    return qm.Qwen25Model(load(weights,configuration()),qm.ModelApproximations.from_dict(profiles()),
        config=qm.QwenConfig(**configuration()),slots=32768,logit_chunk_size=1024)

def signature(scope):
    out=[]
    if scope=='full':
        for start in range(0,151936,1024):
            out.append(dict(name=f'logits.{start}',prefix_size=min(1024,151936-start)))
    elif scope=='backbone':
        out=[dict(name=f'hidden.{p}',prefix_size=896) for p in (0,1)]
    else:raise ValueError('Unknown scope')
    for layer in range(2):
        for kind in ('key','value'):
            for position in (0,1):out.append(dict(name=f'layer{layer}.{kind}.{position}',prefix_size=128))
    return dict(config=configuration(),positions=[0,1],slots=32768,mode='decode-pair',scope=scope,
        inputs=[dict(name=f'embedding.{i}',prefix_size=896) for i in (0,1)],outputs=out,
        note='Two independent Decoder Blocks. Full scope adds final RMSNorm and all 151936 logits. Synthetic weights.')

def evaluate(m,expressions,scope):
    from qwen24.kernels import qwen_model as qm
    from qwen24.kernels import qwen_layers as ql
    assert len(expressions)==2 and m.config.num_layers==2
    cache=[qm.LayerKVCache((),(),()) for _ in range(2)]
    hidden_outputs=[]
    for position,expr in enumerate(expressions):
        hidden=(ql.PackedVector(expr,896,32768),)
        for layer in range(2):
            previous=cache[layer]
            hidden,cache[layer]=m._decoder_block(hidden,(position,),previous,layer)
            assert cache[layer].positions==tuple(range(position+1))
            if position:
                assert cache[layer].keys[0] is previous.keys[0] and cache[layer].values[0] is previous.values[0]
        hidden_outputs.append(hidden[0])
    if scope=='full':
        last=ql.rms_norm(hidden_outputs[-1],m.weights['final_normal.weight'],
            rsqrt=m.approximations.final_norm,eps=m.config.rms_norm_eps)
        out=list(ql.lm_head(last,m.weights['lm_head.weight'],chunk_size=1024))
    else:out=hidden_outputs
    for entry in cache:out.extend(entry.keys);out.extend(entry.values)
    assert [x.size for x in out]==[x['prefix_size'] for x in signature(scope)['outputs']]
    return [x.expr for x in out]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--scope',choices=('backbone','full'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    assert not a.output_dir.exists(),'Preserve previous output'
    import hecate as hc
    m=model(a.weights)
    @hc.func('c,c')
    def qwen2layer_decode(x0,x1):return evaluate(m,(x0,x1),a.scope)
    a.output_dir.mkdir(parents=True)
    started=time.monotonic();hc.save(str(a.output_dir),str(a.output_dir),bytecode=True)
    sig=signature(a.scope);sig['native_trace_seconds']=time.monotonic()-started
    (a.output_dir/'signature.json').write_text(json.dumps(sig,indent=2))
    print(json.dumps(sig),flush=True)

if __name__=='__main__':main()
