"""Candidate tiled joint Q/K/V projection with separate clean cache output."""
import numpy as np
from qwen24.kernels import qwen_layers as ql,ops,tiled_down


def project_rope(x,q_weight,k_weight,v_weight,*,position,head_dim,theta,
                 q_bias=None,k_bias=None,v_bias=None,return_raw_v=False,combine_query=True):
    weights=tuple(ops._real_array(w,'QKV weight') for w in (q_weight,k_weight,v_weight))
    if any(w.ndim!=2 or w.shape[1]!=x.size or not 0<w.shape[0]<=x.slots for w in weights):
        raise ValueError('QKV weights require [output,input] within slot capacity')
    sizes=tuple(w.shape[0] for w in weights)
    if sizes[1]!=sizes[2]:raise ValueError('K and V dimensions must match')
    biases=[]
    for bias,size in zip((q_bias,k_bias,v_bias),sizes):
        value=None if bias is None else ops._real_array(bias,'QKV bias')
        if value is not None and value.shape!=(size,):raise ValueError('Bias shape mismatch')
        biases.append(value)
    begin=0 if combine_query else 1
    if sum(sizes[begin:])>x.slots:
        q,k,v=(ql.linear(x,w,b) for w,b in zip(weights,biases))
        raw_v=v
    else:
        joined_weight=np.concatenate(weights[begin:],axis=0)
        joined_bias=None if all(b is None for b in biases[begin:]) else np.concatenate([
            np.zeros(size) if b is None else b for size,b in zip(sizes[begin:],biases[begin:])])
        # Public column tiles reduce repeated row reductions. Full-size tile
        # is 128; smaller capacities keep the same arithmetic for validation.
        joined=tiled_down.linear(x,joined_weight,joined_bias,
            tile=min(128,max(1,x.slots//4)))
        # Q remains mathematically correct even when split. For the
        # first causal row, GQA never consumes its score, so compiler DCE can
        # discard this independent projection while retaining the KV result.
        key_offset=sizes[0] if combine_query else 0
        q=ql.PackedVector(joined.expr,sizes[0],x.slots) if combine_query else ql.linear(x,weights[0],biases[0])
        k=ql.PackedVector(joined.expr.rotate(key_offset) if key_offset else joined.expr,sizes[1],x.slots)
        # q/k views have dirty tails and carry NO clean-prefix certificate.
        # RoPE's existing masks isolate both split halves of every head.
        # The raw V view is only for internal GQA, whose existing head
        # extracts mask it. Never certify its tail. Returned cache stays clean.
        raw_v=ql.PackedVector(joined.expr.rotate(key_offset+sizes[1]),sizes[2],x.slots)
        v=ql.extract(raw_v,0,sizes[2])
    result=(ql.rope(q,position,head_dim=head_dim,theta=theta),
            ql.rope(k,position,head_dim=head_dim,theta=theta),v)
    return result+(raw_v,) if return_raw_v else result
