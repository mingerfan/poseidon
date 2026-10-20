"""Public packed 24-block interface; no Hecate imports."""
import copy
from qwen24 import spec

def signature(config, mode, positions, cached=(), slots=32768):
    old = spec.signature(config, mode, positions, cached, slots, slots)
    kv = config['kv_heads'] * config['head_dim']
    all_positions = list(cached) + list(positions)
    if len(all_positions) * 2 * kv > slots:
        raise ValueError('Prototype cache must fit one ciphertext per layer')
    sig = copy.deepcopy(old)
    sig['format'] = 'qwen25-24layer-r071-packed-v1'
    sig['inputs'] = copy.deepcopy(old['inputs'][:len(positions)])
    if cached:
        sig['inputs'] += [dict(name=f'layer_{i}.kv_cache', prefix_size=len(cached)*2*kv,
            layout='K0,V0,K1,V1,...', positions=list(cached)) for i in range(config['num_layers'])]
    sig['outputs'] = copy.deepcopy(old['outputs'][:old['logit_ciphertexts']])
    sig['outputs'] += [dict(name=f'layer_{i}.kv_cache', prefix_size=len(all_positions)*2*kv,
        layout='K0,V0,K1,V1,...', positions=all_positions) for i in range(config['num_layers'])]
    for rows in (sig['inputs'], sig['outputs']):
        for i, row in enumerate(rows):
            row['ciphertext'] = i
    sig['optimization'] = dict(attention='batched heads and visible keys',
        logits='32768-slot chunks', kv='one ciphertext per layer',
        encoded_plaintexts='official eager compile; optional validated CPU V3 bounded Encode post-pass')
    return sig
