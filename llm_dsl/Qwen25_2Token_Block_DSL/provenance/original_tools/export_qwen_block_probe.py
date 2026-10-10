"""Trace a whole decoder block and insert public, level-only bootstrap nodes.

No decrypted values are used for scheduling or evaluation. Small synthetic
weights and bounded profiles are a composition pilot, not full Qwen coverage.
"""
import argparse
import hashlib
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import export_qwen_layer_probe as tr


def cost(node):
    if node['op'] in ('multiply','multiply_vector'):
        return 1
    if node['op'] == 'multiply_plain':
        c = node['constant']
        return int(not (c == math.trunc(c) and abs(c) <= 1024))
    return 0


def schedule(graph, output, initial_level, restored_level):
    needed = {output}
    for i in range(len(graph)-1,-1,-1):
        if i in needed:
            needed.update(graph[i][k] for k in ('a','b') if k in graph[i])
    nodes, latest, levels = [], {}, {}
    def refresh(old):
        if levels[old] >= 1:
            return
        original = graph[old]
        # Rotations and addition of public constants need no level. Refresh
        # their shared input once, then recompute these cheap operations.
        if original['op'] in ('rotate', 'add_plain'):
            refresh(original['a'])
            replacement = dict(original, a=latest[original['a']],
                               planned_level=levels[original['a']])
            levels[old] = levels[original['a']]
        else:
            prior = nodes[latest[old]]
            replacement = dict(op='bootstrap', a=latest[old], expected=prior['expected'],
                               expected_tail=prior.get('expected_tail', 0),
                               planned_level=restored_level, source_node=old)
            levels[old] = restored_level
        latest[old] = len(nodes)
        nodes.append(replacement)
    for i,node in enumerate(graph):
        if i not in needed:
            continue
        mapped = dict(node)
        for operand in ('a','b'):
            if operand not in node:
                continue
            old = node[operand]
            if levels[old] < cost(node):
                refresh(old)
            mapped[operand] = latest[old]
        level = min((levels[node[k]] for k in ('a','b') if k in node),default=initial_level) - cost(node)
        mapped['planned_level'] = level
        latest[i],levels[i] = len(nodes),level
        nodes.append(mapped)
    needed = {latest[output]}
    for i in range(len(nodes)-1,-1,-1):
        if i in needed:
            needed.update(nodes[i][k] for k in ('a','b') if k in nodes[i])
    remap, compact = {}, []
    for i,node in enumerate(nodes):
        if i in needed:
            remap[i] = len(compact)
            compact.append({k:remap[v] if k in ('a','b') else v for k,v in node.items()})
    return compact,remap[latest[output]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap-config',type=Path,required=True)
    args = parser.parse_args()
    tr.SLOTS = 32768
    ql = tr.load_layers()
    qm = importlib.import_module('_qwen_ckks_layer_probe.qwen_model')
    cfg = qm.QwenConfig(num_layers=1,hidden_size=4,intermediate_size=8,
                        query_heads=2,kv_heads=1,head_dim=2,vocab_size=8)
    def fit(function, fn, domain, degree):
        p = np.polynomial.Chebyshev.interpolate(fn,degree,domain=domain)
        grid = np.linspace(*domain,8193)
        if np.max(np.abs(p(grid)-fn(grid))) > 1e-5:
            raise AssertionError('profile grid accuracy failed: '+function)
        return ql.ChebyshevApproximation(function,tuple(p.coef),domain,evaluation='blocked')
    rsqrt = fit('rsqrt',lambda x:1/np.sqrt(x),(.5,2.),12)
    silu = fit('silu',lambda x:x/(1+np.exp(-x)),(-2.,2.),12)
    soft = ql.NormalizeSquareSoftmax(.5,0,6,2,exp_degree=8,exp_bound=.5)
    profile = qm.BlockApproximations(rsqrt,rsqrt,silu,softmax=soft)
    profiles = qm.ModelApproximations((profile,),rsqrt)
    rng = np.random.default_rng(20260920)
    weights = {name:rng.normal(scale=.15,size=shape) for name,shape in qm.weight_shapes(cfg).items()}
    for name in weights:
        if 'Normal.weight' in name or name=='final_normal.weight':
            weights[name] = rng.uniform(.9,1.1,size=weights[name].shape)
    x = rng.normal(size=(2,4))
    x /= np.sqrt(np.mean(x*x,axis=1,keepdims=True))
    positions=(7,8)
    w = lambda name: weights['model_list.0.'+name]
    def norm(v,gain):
        variance=np.mean(v*v,axis=-1,keepdims=True)+1e-6
        assert np.all((variance>=rsqrt.domain[0]) & (variance<=rsqrt.domain[1]))
        return v/np.sqrt(variance)*gain
    n=norm(x,w('pre_Normal.weight'))
    q=n@w('attention.q_weight.weight').T+w('attention.q_weight.bias')
    k=n@w('attention.k_weight.weight').T+w('attention.k_weight.bias')
    v=n@w('attention.v_weight.weight').T+w('attention.v_weight.bias')
    def rope(a,heads):
        a=a.reshape(2,heads,2)
        angle=np.asarray(positions)[:,None,None]
        return a*np.cos(angle)+np.stack((-a[:,:,1],a[:,:,0]),axis=-1)*np.sin(angle)
    q,k=rope(q,2),rope(k,1)
    attended=[]
    max_centered=0.
    for i in range(2):
        heads=[]
        for h in range(2):
            scores=k[:i+1,0]@q[i,h]/math.sqrt(2)
            max_centered=max(max_centered,float(np.max(np.abs(scores-scores.mean()))))
            p=np.exp(scores-scores.max());p/=p.sum()
            heads.append(p@v[:i+1])
        attended.append(w('attention.o_weight.weight')@np.concatenate(heads))
    assert max_centered<=soft.centered_bound
    residual=x+np.asarray(attended)
    post=norm(residual,w('post_Normal.weight'))
    gate=post@w('mlplayer.gate_proj.weight').T
    assert np.max(np.abs(gate))<=2
    ffn=(gate/(1+np.exp(-gate))*(post@w('mlplayer.up_proj.weight').T))@w('mlplayer.down_proj.weight').T
    target=(residual+ffn).ravel()
    model=qm.Qwen25Model(weights,profiles,config=cfg,slots=tr.SLOTS,logit_chunk_size=8)
    boot=json.loads(args.bootstrap_config.read_text())
    boot['output_level']=5
    programs=[]
    for mode in ('prefill','decode'):
        graph=[]
        packed=tuple(ql.PackedVector(tr.Expr(graph,'input',np.r_[row,[.3,-.2]]),4,tr.SLOTS) for row in x)
        empty=qm.LayerKVCache((),(),())
        if mode=='prefill':
            outputs,_=model._decoder_block(packed,positions,empty,0)
        else:
            first,cache=model._decoder_block(packed[:1],positions[:1],empty,0)
            second,_=model._decoder_block(packed[1:],positions[1:],cache,0)
            outputs=first+second
        output=ql.concatenate(outputs)
        err=float(np.max(np.abs(output.expr.values[:8]-target)))
        assert err<1e-5,err
        np.testing.assert_allclose(output.expr.values[8:],0,atol=1e-10)
        nodes,last=schedule(graph,output.expr.index,8,5)
        count=sum(n['op']=='bootstrap' for n in nodes)
        print(mode,'nodes',len(nodes),'bootstraps',count,'approximation_error',err,flush=True)
        programs.append(dict(name='decoder_block_'+mode,packing='prefix',slots=tr.SLOTS,
                             nodes=nodes,output=last,target=target.tolist(),output_size=8,
                             zero_output_tail=True,max_abs_error=1e-3,
                             planned_bootstraps=count,plaintext_approximation_error=err))
    hashes={name:hashlib.sha256((tr.SOURCE/name).read_bytes()).hexdigest()
            for name in ('qwen_layers.py','qwen_model.py','qwen_nonlinear.py','ops.py')}
    data=dict(format='qwen-ckks-diagnostic-dag-v1',probe_kind='complete_decoder_block_with_bootstrap',
              context=dict(log_n=16,scale_bits=boot['scale_bits'],q_bits=boot['q_bits'],p_bits=boot['p_bits'],
                           initial_level=8,max_scale_snap_relative=1e-5),bootstrap=boot,
              candidate_sha256=hashlib.sha256(json.dumps(profiles.to_dict(),sort_keys=True).encode()).hexdigest(),
              kernel_sha256=hashes['qwen_layers.py'],source_sha256=hashes,synthetic_weights=True,
              contract=dict(config=asdict(cfg),profiles=profiles.to_dict(),positions=positions,
                            observed_centered_bound=max_centered,inputs=x.tolist()),programs=programs)
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(data,out,allow_nan=False)


if __name__=='__main__':
    main()
