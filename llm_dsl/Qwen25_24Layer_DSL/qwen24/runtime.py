"""Hecate entrypoints using the frozen Qwen model and encrypted per-layer KV."""
import json
from pathlib import Path
from . import spec
from .weights import load
from .kernels import qwen_layers as ql
from .kernels import qwen_model as qm


def load_model(weights_path, config_path=spec.CONFIG, profiles_path=spec.PROFILES,
               slots=32768, chunk=1024):
    c = spec.config(config_path)
    cfg = qm.QwenConfig(**c)
    if qm.weight_shapes(cfg) != spec.weight_shapes(c):
        raise AssertionError('Public and kernel weight schemas disagree')
    profiles = qm.ModelApproximations.from_dict(json.loads(Path(profiles_path).read_text(encoding='utf-8')))
    return qm.Qwen25Model(load(weights_path, c), profiles, config=cfg,
                         slots=slots, logit_chunk_size=chunk)


def evaluate(model, sig, expressions):
    if len(expressions) != len(sig['inputs']):
        raise ValueError('Ciphertext input count does not match signature')
    positions, cached = tuple(sig['new_positions']), tuple(sig['cached_positions'])
    count, slots, cfg = len(positions), model.slots, model.config
    if slots != sig['slots'] or cfg.num_layers != 24:
        raise ValueError('Model/signature mismatch')
    for profile in model.approximations.blocks:
        if profile.softmax is not None and len(positions + cached) > profile.softmax.max_length:
            raise ValueError('Sequence exceeds the selected Softmax profile')
    embeddings = tuple(ql.PackedVector(e, cfg.hidden_size, slots) for e in expressions[:count])
    if sig['mode'] == 'prefill':
        output = model.prefill(embeddings, positions=positions)
    elif sig['mode'] == 'decode-pair':
        # Build the first token's complete 24-layer KV state. Its logits are not
        # requested, so avoid an unused full-vocabulary projection here.
        hidden, cache = (embeddings[0],), []
        for layer in range(cfg.num_layers):
            hidden, entry = model._decoder_block(hidden, positions[:1], qm.LayerKVCache((), (), ()), layer)
            cache.append(entry)
        output = model.decode(embeddings[1], cache, position=positions[1])
    else:
        cache, offset = [], count
        kv_size = cfg.kv_heads * cfg.head_dim
        for _ in range(cfg.num_layers):
            keys = tuple(ql.PackedVector(e, kv_size, slots) for e in expressions[offset:offset + len(cached)])
            offset += len(cached)
            values = tuple(ql.PackedVector(e, kv_size, slots) for e in expressions[offset:offset + len(cached)])
            offset += len(cached)
            cache.append(qm.LayerKVCache(cached, keys, values))
        output = model.decode(embeddings[0], cache, position=positions[0])
    result = list(output.logits)
    for entry in output.cache:
        if entry.positions != cached + positions:
            raise AssertionError('Cache positions changed')
        result.extend(entry.keys)
        result.extend(entry.values)
    if len(result) != len(sig['outputs']) or any(v.size != row['prefix_size'] for v, row in zip(result, sig['outputs'])):
        raise AssertionError('Output shape does not match signature')
    return [v.expr for v in result]


def register(hc, model, sig):
    @hc.func(','.join('c' for _ in sig['inputs']))
    def qwen25_24layer(*expressions):
        return evaluate(model, sig, expressions)
    return qwen25_24layer
