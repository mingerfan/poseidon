"""Round-one ABI: batched GQA, full-slot logits and one KV CT per layer.

Experimental adapter over the frozen model. Explicitly versioned interface;
old clients must not interpret these outputs using the prefix-v1 signature.
"""
import copy
from dataclasses import asdict
from qwen24 import spec
from qwen24 import runtime as base
from qwen24.kernels import qwen_layers as ql
import packed_attention as packed


def signature(config, mode, positions, cached=(), slots=32768):
    old = spec.signature(config, mode, positions, cached, slots, slots)
    kv = config['kv_heads'] * config['head_dim']
    all_positions = list(cached) + list(positions)
    if len(all_positions) * 2 * kv > slots:
        raise ValueError('Prototype cache must fit one ciphertext per layer')
    sig = copy.deepcopy(old)
    sig['format'] = 'qwen25-2layer-dsl-packed-v2-experimental'
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


def load_model(*args, **kwargs):
    # Caller must explicitly select the new ABI. The baseline package is intact.
    mlp_pair = kwargs.pop('mlp_pair', False)
    pair_batch = kwargs.pop('pair_batch', 'none')
    if mlp_pair and pair_batch != 'both_except_last':
        raise ValueError('Pair MLP requires the admitted known-input RMS path')
    early_input_mask = kwargs.pop('early_input_mask', False)
    clean_kv_output = kwargs.pop('clean_kv_output', False)
    if pair_batch not in ('none','both_except_last'):
        raise ValueError('Unknown known-input pair batch mode')
    batch_strategy = kwargs.pop('rms_batch_strategy', 'tiled16')
    if batch_strategy not in ('tiled16','tiled32','tiled64','tiled128'):
        raise ValueError('Unknown RMS batch strategy')
    if pair_batch != 'none':
        kwargs['rms_batch_mode'] = pair_batch
        kwargs['rms_strategy'] = batch_strategy
    global packed
    if kwargs.pop('layout_fusion', False):
        import packed_attention_fused
        from qwen24.kernels import prefix_fusion
        packed = packed_attention_fused
        prefix_fusion.install()
    slots = kwargs.pop('slots', 32768)
    strategy=kwargs.pop('rms_degree_strategy','baseline')
    strategy,marker,silu=strategy.partition('_silu_')
    from qwen24.kernels import silu_parity_candidate
    silu_parity_candidate.install(silu if marker else 'none')
    rms,separator,pair=strategy.partition('_pair')
    packed.PAIR_DEGREE=int(pair) if separator else 13
    if packed.PAIR_DEGREE not in (7,9,11,13):raise ValueError('Invalid pair degree')
    kwargs['rms_degree_strategy']=rms
    model = base.load_model(*args, slots=slots, chunk=slots, **kwargs)
    model.mlp_pair = mlp_pair
    model.coalesce_known_pair = pair_batch != 'none'
    model.early_input_mask = early_input_mask
    model.clean_kv_output = clean_kv_output
    ql.gqa_attention = packed.gqa_attention
    return model


def pack_kv_output(keys, values, *, clean=False):
    value = packed.pack_kv(keys, values)
    if clean:
        import numpy as np
        from qwen24.kernels import ops
        from qwen24.kernels.expr import Plain
        # Numerical boundary mask: do not elide it using an algebraic tail proof.
        mask = np.zeros(value.slots)
        mask[:value.size] = 1.
        value = ql.PackedVector(ops.mul(value.expr, Plain(mask)), value.size, value.slots)
    return value


def evaluate(model, sig, expressions):
    c = asdict(model.config)
    positions, cached = sig['new_positions'], sig['cached_positions']
    expected = signature(c, sig['mode'], positions, cached, sig['slots'])
    for field in ('format', 'inputs', 'outputs', 'logit_chunk_size'):
        if sig[field] != expected[field]:
            raise ValueError('Signature mismatch: ' + field)
    if len(expressions) != len(sig['inputs']) or model.logit_chunk_size != sig['slots']:
        raise ValueError('Input count or logit chunk mismatch')
    old = spec.signature(c, sig['mode'], positions, cached, sig['slots'], sig['slots'])
    ql.gqa_attention = packed.gqa_attention
    count, kv = len(positions), c['kv_heads'] * c['head_dim']
    expanded = list(expressions[:count])
    if cached:
        for expr in expressions[count:]:
            cache = ql.PackedVector(expr, len(cached)*2*kv, sig['slots'])
            keys, values = packed.unpack_kv(cache, tokens=len(cached), kv_size=kv)
            expanded.extend(x.expr for x in keys + values)
    output = base.evaluate(model, old, expanded)
    result = list(output[:old['logit_ciphertexts']])
    cursor, tokens = old['logit_ciphertexts'], len(cached) + count
    for _ in range(c['num_layers']):
        keys = tuple(ql.PackedVector(e, kv, sig['slots']) for e in output[cursor:cursor+tokens])
        cursor += tokens
        values = tuple(ql.PackedVector(e, kv, sig['slots']) for e in output[cursor:cursor+tokens])
        cursor += tokens
        result.append(pack_kv_output(keys, values, clean=getattr(model,'clean_kv_output',False)).expr)
    if cursor != len(output) or len(result) != len(sig['outputs']):
        raise AssertionError('Output accounting mismatch')
    return result


def register(hc, model, sig):
    @hc.func(','.join('c' for _ in sig['inputs']))
    def qwen25_2layer(*expressions):
        return evaluate(model, sig, expressions)
    return qwen25_2layer
