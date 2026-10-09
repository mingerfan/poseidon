"""New tracing glue around the byte-identical, CPU-validated block implementation."""
import json
from pathlib import Path
import numpy as np
from .fixture import ROOT,config,shapes
from .kernels import qwen_layers as ql
from .kernels import qwen_model as qm

SLOTS=32768
POSITIONS=(7,8)

def load_model(weights_path):
    cfg=qm.QwenConfig(**config())
    with np.load(weights_path,allow_pickle=False) as archive:
        if set(archive.files)!=set(shapes()):
            raise ValueError('Expected the generated one-block weight set')
        weights={n:np.asarray(archive[n],dtype=np.float64) for n in shapes()}
    for name in ('emb_weight.weight','lm_head.weight'):
        # Unused constructor placeholders; never part of this block graph.
        weights[name]=np.broadcast_to(np.float64(0),(cfg.vocab_size,cfg.hidden_size))
    profiles=qm.ModelApproximations.from_dict(json.loads((ROOT/'config/approximations.json').read_text()))
    return qm.Qwen25Model(weights,profiles,config=cfg,slots=SLOTS,logit_chunk_size=8)

def signature(mode):
    if mode not in ('prefill','decode-pair'):
        raise ValueError('Expected prefill or decode-pair')
    return dict(format='qwen-2token-block-handoff-v1',mode=mode,
        function='_hecate_qwen_block_2tokens',slots=SLOTS,positions=list(POSITIONS),
        inputs=[dict(name='hidden_position_'+str(p),ciphertext=i,prefix_size=896) for i,p in enumerate(POSITIONS)],
        outputs=[dict(name='concatenated_hidden',ciphertext=0,prefix_size=1792,
                      order=['position_7[0:896]','position_8[0:896]'],tail='zero')],
        kv='remains an encrypted expression inside decode-pair; no external cache input/output',
        refresh='GPU backend must schedule real bootstrap/rescale; no explicit bootstrap in source DSL',
        scope='one decoder block, synthetic weights; no embedding/final norm/LM head')

def evaluate(model,mode,expressions):
    signature(mode)
    if len(expressions)!=2:
        raise ValueError('Exactly two encrypted hidden vectors required')
    hidden=tuple(ql.PackedVector(e,896,SLOTS) for e in expressions)
    cache=qm.LayerKVCache((),(),())
    if mode=='prefill':
        outputs,cache=model._decoder_block(hidden,POSITIONS,cache,0)
    else:
        outputs=()
        for token,position in zip(hidden,POSITIONS):
            step,cache=model._decoder_block((token,),(position,),cache,0)
            outputs+=step
    if cache.positions!=POSITIONS or len(cache.keys)!=2 or len(cache.values)!=2:
        raise AssertionError('KV cache contract mismatch')
    # Exactly the final packing operation in the historical CPU diagnostic.
    return ql.concatenate(outputs).expr

def register(hc,model,mode):
    signature(mode)
    @hc.func('c,c')
    def qwen_block_2tokens(x0,x1):
        return [evaluate(model,mode,(x0,x1))]
    return qwen_block_2tokens
