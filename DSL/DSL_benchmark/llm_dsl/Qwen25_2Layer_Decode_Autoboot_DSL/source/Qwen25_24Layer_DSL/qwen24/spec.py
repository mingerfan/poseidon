"""Public model and ciphertext interface; no Hecate/native imports."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/model.json'
PROFILES = ROOT / 'config/approximations.provisional.json'


def config(path=CONFIG):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('num_layers', 'hidden_size', 'intermediate_size', 'query_heads',
                'kv_heads', 'head_dim', 'vocab_size', 'max_sequence_length'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError('Positive integer required: ' + key)
    if c['num_layers'] != 24:
        raise ValueError('This package requires exactly 24 decoder blocks')
    if c['hidden_size'] != c['query_heads'] * c['head_dim'] or c['head_dim'] % 2:
        raise ValueError('Invalid head dimensions')
    if c['query_heads'] % c['kv_heads']:
        raise ValueError('Query heads must be divisible by KV heads')
    for key in ('rope_theta', 'rms_norm_eps'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Positive finite number required: ' + key)
    return c


def weight_shapes(c):
    h, f, k = c['hidden_size'], c['intermediate_size'], c['kv_heads'] * c['head_dim']
    result = {'emb_weight.weight': (c['vocab_size'], h), 'final_normal.weight': (h,),
              'lm_head.weight': (c['vocab_size'], h)}
    for layer in range(c['num_layers']):
        result.update({f'model_list.{layer}.' + name: shape for name, shape in {
            'pre_Normal.weight': (h,), 'post_Normal.weight': (h,),
            'attention.q_weight.weight': (h, h), 'attention.q_weight.bias': (h,),
            'attention.k_weight.weight': (k, h), 'attention.k_weight.bias': (k,),
            'attention.v_weight.weight': (k, h), 'attention.v_weight.bias': (k,),
            'attention.o_weight.weight': (h, h), 'mlplayer.gate_proj.weight': (f, h),
            'mlplayer.up_proj.weight': (f, h), 'mlplayer.down_proj.weight': (h, f),
        }.items()})
    return result


def signature(c, mode, positions, cached=(), slots=32768, chunk=1024):
    positions, cached = tuple(positions), tuple(cached)
    if mode not in ('prefill', 'decode', 'decode-pair'):
        raise ValueError('Unknown mode')
    if not positions or any(type(p) is not int or p < 0 for p in positions + cached):
        raise ValueError('Positions must be nonnegative integers')
    if any(a >= b for a, b in zip(cached + positions, (cached + positions)[1:])):
        raise ValueError('Positions must be strictly increasing')
    if mode == 'decode' and (len(positions) != 1 or not cached):
        raise ValueError('decode needs one new position and nonempty cached positions')
    if mode != 'decode' and cached:
        raise ValueError('Only decode accepts external cached positions')
    if mode == 'decode-pair' and len(positions) != 2:
        raise ValueError('decode-pair requires two positions')
    if positions[-1] >= c['max_sequence_length']:
        raise ValueError('Position exceeds model context')
    if type(slots) is not int or slots <= 0 or slots & (slots - 1):
        raise ValueError('slots must be a positive power of two')
    if max(c['hidden_size'], c['intermediate_size'], len(cached + positions)) > slots:
        raise ValueError('Vectors must fit the slot capacity')
    if type(chunk) is not int or not 0 < chunk <= slots:
        raise ValueError('logit chunk size must fit slots')
    kv = c['kv_heads'] * c['head_dim']
    inputs = [dict(name=f'embedding.position_{p}', prefix_size=c['hidden_size']) for p in positions]
    for layer in range(c['num_layers']):
        for kind in ('key', 'value'):
            inputs += [dict(name=f'layer_{layer}.{kind}.position_{p}', prefix_size=kv) for p in cached]
    logits = [dict(name=f'logits.{start}:{min(start + chunk, c["vocab_size"])}',
                   prefix_size=min(chunk, c['vocab_size'] - start))
              for start in range(0, c['vocab_size'], chunk)]
    outputs = logits.copy()
    for layer in range(c['num_layers']):
        for kind in ('key', 'value'):
            outputs += [dict(name=f'layer_{layer}.{kind}.position_{p}', prefix_size=kv)
                        for p in cached + positions]
    for rows in (inputs, outputs):
        for i, row in enumerate(rows):
            row['ciphertext'] = i
    return dict(format='qwen25-24layer-dsl-v1', config=c, mode=mode, function='_hecate_qwen25_24layer',
                new_positions=list(positions), cached_positions=list(cached),
                slots=slots, logit_chunk_size=chunk, logit_ciphertexts=len(logits),
                inputs=inputs, outputs=outputs,
                output_tail='zero; verify after actual encrypted execution',
                embedding='client plaintext lookup, then encrypt; not encrypted token-ID lookup',
                output='last-new-token raw logits plus all 24 layers of encrypted KV',
                bootstrap='backend must schedule real refresh; not emitted explicitly by this DSL',
                native_compiled=False, full_model_ckks_validated=False, gpu_validated=False)
