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


from interface import signature


def load_model(*args, **kwargs):
    # Caller must explicitly select the new ABI. The baseline package is intact.
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
    ql.gqa_attention = packed.gqa_attention
    return model


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
        result.append(packed.pack_kv(keys, values).expr)
    if cursor != len(output) or len(result) != len(sig['outputs']):
        raise AssertionError('Output accounting mismatch')
    return result


def register(hc, model, sig):
    @hc.func(','.join('c' for _ in sig['inputs']))
    def qwen25_24layer(*expressions):
        return evaluate(model, sig, expressions)
    return qwen25_24layer
