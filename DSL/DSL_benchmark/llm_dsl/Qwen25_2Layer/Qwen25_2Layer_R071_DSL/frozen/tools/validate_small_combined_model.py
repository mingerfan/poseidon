"""Server-side 2 small blocks using the packed ABI, independent dense math."""
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
sys.path.insert(0,str(ROOT/'package/verification'))
from validate_numpy import Expr,Plain
from prepare_fixture import prepare
from qwen24 import spec
from qwen24.weights import load
from qwen24.reference import forward,assert_provisional_bounds
from check_outputs import check

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'degree10p2_pair13'
    assert mode in ('degree10p2_pair13', 'degree10p2_pair13_silu_canonical_fold2', 'degree10p2_pair13_silu_pre_fold2')
    config_path=ROOT/'package/verification/smoke_config.json'
    c=spec.config(config_path);directory=ROOT/('small-model-'+mode)
    directory.mkdir();fixture=directory/'fixture'
    prepare(fixture,c,slots=32)
    weights=load(fixture/'weights',c)
    embeddings=np.load(fixture/'input_embeddings.npy')
    module=types.ModuleType('qwen24.kernels.expr');module.Expr=Expr;module.Plain=Plain
    sys.modules[module.__name__]=module
    import combined_runtime as combined
    from qwen24.kernels import qwen_model as qm
    model=combined.load_model(fixture/'weights',config_path=config_path,slots=32,rms_degree_strategy=mode)
    inputs=[]
    for row in embeddings:
        dirty=np.full(32,.3);dirty[:c['hidden_size']]=row;inputs.append(Expr(dirty))
    summaries=[]
    def run(mode,positions,values,cached=()):
        sig=combined.signature(c,mode,positions,cached,32)
        funcs=[];calls=[]
        def func(typespec):
            assert typespec==','.join('c' for _ in sig['inputs'])
            def decorate(fn):funcs.append(fn);return fn
            return decorate
        original=type(model)._decoder_block
        def observe(obj,hidden,pos,cache,index):
            calls.append(index);return original(obj,hidden,pos,cache,index)
        started=time.monotonic()
        combined.register(types.SimpleNamespace(func=func),model,sig)
        with patch.object(type(model),'_decoder_block',observe):output=funcs[0](*values)
        assert calls==list(range(2))*(2 if mode=='decode-pair' else 1)
        assert len(output)==len(sig['outputs'])
        arrays=np.stack([e.values for e in output])
        tails=[float(np.max(np.abs(v[row['prefix_size']:]),initial=0)) for v,row in zip(arrays,sig['outputs'])]
        assert max(tails)<1e-10
        n=sig['logit_ciphertexts'];kv=c['kv_heads']*c['head_dim'];tokens=len(cached)+len(positions)
        logits=np.concatenate([arrays[i,:sig['outputs'][i]['prefix_size']] for i in range(n)])
        caches=np.stack([row[:tokens*2*kv].reshape(tokens,2,kv) for row in arrays[n:]])
        actual=dict(logits=logits,keys=caches[:,:,0,:],values=caches[:,:,1,:])
        expected,bounds=forward(weights,embeddings[:tokens],tuple(cached)+tuple(positions),c)
        assert_provisional_bounds(bounds)
        result=check(actual,expected,atol=1e-5);assert result['passed'],result
        summaries.append(dict(mode=mode,new_positions=list(positions),cached_positions=list(cached),
            executed_blocks=len(calls),inputs=len(values),outputs=len(output),comparison=result,
            max_tail=max(tails),seconds=time.monotonic()-started))
        return output,actual,sig
    _,prefill,_=run('prefill',(0,1),inputs)
    _,pair,_=run('decode-pair',(0,1),inputs)
    first,_,sig=run('prefill',(0,),inputs[:1])
    cache=first[sig['logit_ciphertexts']:];snapshots=[x.values.copy() for x in cache]
    _,external,_=run('decode',(1,),inputs[1:]+cache,(0,))
    for value,before in zip(cache,snapshots):np.testing.assert_array_equal(value.values,before)
    differences={}
    for name,candidate in [('pair',pair),('external',external)]:
        differences[name]={k:float(np.max(np.abs(prefill[k]-candidate[k]))) for k in prefill}
        assert max(differences[name].values())<1e-11
    report=dict(passed=True,scope='2 blocks, reduced dimensions, packed ABI arithmetic only',
        config=c,slots=32,checks=summaries,differences=differences,cache_immutable=True,
        native_trace=False,ckks=False,gpu=False,external_cache_level_handoff_verified=False)
    (ROOT/('small-model-'+mode+'-result.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
