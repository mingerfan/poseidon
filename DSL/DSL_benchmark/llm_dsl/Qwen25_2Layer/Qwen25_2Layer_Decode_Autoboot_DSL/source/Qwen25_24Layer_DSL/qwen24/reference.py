"""Independent dense original-function reference; no packed kernels or Hecate."""
import math
import numpy as np


def forward(weights, embeddings, positions, c):
    hidden = np.array(embeddings, dtype=np.float64, copy=True)
    seq, h = hidden.shape
    if h != c['hidden_size'] or seq != len(positions):
        raise ValueError('Reference input shape mismatch')
    dim, qh, kh = c['head_dim'], c['query_heads'], c['kv_heads']
    freqs = 1 / c['rope_theta'] ** (np.arange(0, dim, 2) / dim)
    angles = np.outer(positions, freqs)
    angles = np.concatenate((angles, angles), axis=-1)[:, None, :]
    keys, values, bounds = [], [], []

    def rope(v):
        return v * np.cos(angles) + np.concatenate((-v[..., dim//2:], v[..., :dim//2]), axis=-1) * np.sin(angles)

    def norm(x, gain):
        var = np.mean(x*x, axis=-1, keepdims=True) + c['rms_norm_eps']
        return x / np.sqrt(var) * gain, (float(var.min()), float(var.max()))

    for layer in range(c['num_layers']):
        def w(name):
            return weights[f'model_list.{layer}.' + name]
        n, pre_bounds = norm(hidden, w('pre_Normal.weight'))
        q = rope((n @ w('attention.q_weight.weight').T + w('attention.q_weight.bias')).reshape(seq, qh, dim))
        k = rope((n @ w('attention.k_weight.weight').T + w('attention.k_weight.bias')).reshape(seq, kh, dim))
        v = (n @ w('attention.v_weight.weight').T + w('attention.v_weight.bias')).reshape(seq, kh, dim)
        keys.append(k.reshape(seq, -1).copy())
        values.append(v.reshape(seq, -1).copy())
        attended = np.empty((seq, qh, dim))
        centered_max = 0.
        for token in range(seq):
            for head in range(qh):
                group = head // (qh // kh)
                scores = k[:token+1, group] @ q[token, head] / math.sqrt(dim)
                centered_max = max(centered_max, float(np.abs(scores - scores.mean()).max()))
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                attended[token, head] = probability @ v[:token+1, group]
        hidden += attended.reshape(seq, h) @ w('attention.o_weight.weight').T
        n, post_bounds = norm(hidden, w('post_Normal.weight'))
        gate = n @ w('mlplayer.gate_proj.weight').T
        up = n @ w('mlplayer.up_proj.weight').T
        hidden += (gate / (1 + np.exp(-gate)) * up) @ w('mlplayer.down_proj.weight').T
        bounds.append(dict(layer=layer, pre_norm_variance=pre_bounds, post_norm_variance=post_bounds,
                           gate_abs_max=float(np.abs(gate).max()), centered_score_abs_max=centered_max))
    final, final_bounds = norm(hidden[-1], weights['final_normal.weight'])
    # Chunk the public LM matrix to bound conversion/matmul temporaries.
    lm = weights['lm_head.weight']
    logits = np.concatenate([lm[start:start+1024] @ final for start in range(0, len(lm), 1024)])
    return dict(logits=logits, keys=np.array(keys), values=np.array(values)), dict(
        layers=bounds, final_norm_variance=final_bounds)


def assert_provisional_bounds(bounds):
    for row in bounds['layers']:
        for key in ('pre_norm_variance', 'post_norm_variance'):
            lo, hi = row[key]
            if not .5 <= lo <= hi <= 2:
                raise ValueError('Synthetic fixture left provisional norm domain')
        if row['gate_abs_max'] > 2 or row['centered_score_abs_max'] > .5:
            raise ValueError('Synthetic fixture left provisional activation domain')
    lo, hi = bounds['final_norm_variance']
    if not .5 <= lo <= hi <= 2:
        raise ValueError('Synthetic fixture left final norm domain')
