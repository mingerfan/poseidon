"""Experimental per-model prefill RMS batching for independent round 041.

Requires a later packed_rms_batch.rms_norm_batch(..., strategy=...) interface.
Frozen rounds 033/039 do not use this class. Server validation is still required.
"""
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import qwen_model as qm
from qwen24.kernels import packed_rms_batch as batch


class PrefillRMSModel(qm.Qwen25Model):
    def __init__(self,*args,rms_batch_mode='pre',rms_strategy='tiled16',**kwargs):
        if rms_batch_mode not in ('none','pre','both','both_except_last'):
            raise ValueError('RMS batching mode must be none, pre or both')
        if rms_strategy not in batch.STRATEGIES:
            raise ValueError('Unknown RMS strategy')
        super().__init__(*args,**kwargs)
        self.rms_batch_mode=rms_batch_mode
        self.rms_strategy=rms_strategy

    def _decoder_block(self,hidden,positions,cache,index):
        # One-token prefill/decode uses the exact established implementation.
        if len(hidden)==1 or self.rms_batch_mode=='none':
            return super()._decoder_block(hidden,positions,cache,index)
        cfg=self.config;profile=self.approximations.blocks[index]
        prefix=f'model_list.{index}.'
        def weight(name):return self.weights[prefix+name]
        normalized=batch.rms_norm_batch(hidden,weight('pre_Normal.weight'),
            rsqrt=profile.pre_norm,eps=cfg.rms_norm_eps,strategy=self.rms_strategy)
        attention,keys,values=ql.attention(
            normalized,weight('attention.q_weight.weight'),weight('attention.k_weight.weight'),
            weight('attention.v_weight.weight'),weight('attention.o_weight.weight'),positions=positions,
            q_bias=weight('attention.q_weight.bias'),k_bias=weight('attention.k_weight.bias'),
            v_bias=weight('attention.v_weight.bias'),cached_positions=cache.positions,
            cached_keys=cache.keys,cached_values=cache.values,query_heads=cfg.query_heads,
            kv_heads=cfg.kv_heads,head_dim=cfg.head_dim,theta=cfg.rope_theta,
            exp=profile.exp,reciprocal=profile.reciprocal,public_shift=profile.public_shift,
            normalization=profile.softmax)
        updated=qm.LayerKVCache(cache.positions+positions,keys,values)
        residuals=tuple(ql.add(x,y) for x,y in zip(hidden,attention))
        if self.rms_batch_mode=='both' or (self.rms_batch_mode=='both_except_last' and index<cfg.num_layers-1):
            normalized=batch.rms_norm_batch(residuals,weight('post_Normal.weight'),
                rsqrt=profile.post_norm,eps=cfg.rms_norm_eps,strategy=self.rms_strategy)
        else:
            normalized=tuple(ql.rms_norm(x,weight('post_Normal.weight'),
                rsqrt=profile.post_norm,eps=cfg.rms_norm_eps) for x in residuals)
        if getattr(self,'mlp_pair',False) and len(normalized)==2 and index<cfg.num_layers-1:
            from .packed_mlp_pair import swiglu_pair
            ffns=swiglu_pair(normalized,weight('mlplayer.gate_proj.weight'),
                weight('mlplayer.up_proj.weight'),weight('mlplayer.down_proj.weight'),
                silu=profile.silu)
            return tuple(ql.add(x,f) for x,f in zip(residuals,ffns)),updated
        output=[]
        for x,n in zip(residuals,normalized):
            ffn=ql.swiglu(n,weight('mlplayer.gate_proj.weight'),weight('mlplayer.up_proj.weight'),
                         weight('mlplayer.down_proj.weight'),silu=profile.silu)
            output.append(ql.add(x,ffn))
        return tuple(output),updated
