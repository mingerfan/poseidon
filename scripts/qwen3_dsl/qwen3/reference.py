"""Independent dense original-function reference; no packed kernels or Hecate."""
import math
import numpy as np


def forward(weights, embeddings, positions, c, profiles=None):
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

    def approximate(x, profile):
        lo,hi=profile['domain']
        if np.min(x)<lo or np.max(x)>hi:raise ValueError(f"approximation domain exceeded: {profile['function']} [{np.min(x)},{np.max(x)}] vs {lo,hi}")
        if profile['method']=='chebyshev':
            return np.polynomial.chebyshev.chebval((2*x-lo-hi)/(hi-lo),profile['coefficients'])
        if profile['method']=='newton_rsqrt' and profile['evaluation']=='residual':
            scaled=x/hi;residual=1-scaled;y=1+.5*residual
            for _ in range(profile['iterations']-1):
                residual=residual**2*(.75+.25*residual)
                y=y*(1+.5*residual)
            for _ in range(profile['refinement_steps']):y=y*(1.5-.5*scaled*y*y)
            return y/math.sqrt(hi)
        if profile['method']=='normalize_square_silu':
            z=x/2**(profile['squarings']+1)
            positive=sum(z**j/math.factorial(j) for j in range(profile['exp_degree']+1))
            negative=sum((-z)**j/math.factorial(j) for j in range(profile['exp_degree']+1))
            def reciprocal(v,upper,it):
                residual=1-v/upper;y=(1+residual)/upper
                for _ in range(it-1):residual=residual*residual;y=y*(1+residual)
                return y
            inverse=reciprocal(positive+negative,4,profile['intermediate_iterations'] if profile['squarings'] else profile['reciprocal_iterations'])
            positive=positive*inverse;negative=negative*inverse
            for step in range(profile['squarings']):
                positive=positive*positive;negative=negative*negative
                inverse=reciprocal(positive+negative,1.01,profile['reciprocal_iterations'] if step==profile['squarings']-1 else profile['intermediate_iterations'])
                positive=positive*inverse;negative=negative*inverse
            return x*positive
        raise ValueError(f"unsupported reference approximation: {profile['method']}")

    def norm(x, gain, profile=None):
        var = np.mean(x*x, axis=-1, keepdims=True) + c['rms_norm_eps']
        inverse=1/np.sqrt(var) if profile is None else approximate(var,profile)
        return x * inverse * gain, (float(var.min()), float(var.max()))

    for layer in range(c['num_layers']):
        def w(name):
            return weights[f'model_list.{layer}.' + name]
        profile=profiles['blocks'][layer] if profiles else None
        n, pre_bounds = norm(hidden, w('pre_Normal.weight'),profile['pre_norm'] if profile else None)
        q, q_bounds = norm((n @ w('attention.q_weight.weight').T).reshape(seq, qh, dim), w('attention.q_norm.weight'))
        k, k_bounds = norm((n @ w('attention.k_weight.weight').T).reshape(seq, kh, dim), w('attention.k_norm.weight'))
        q_variance = np.mean((n @ w('attention.q_weight.weight').T).reshape(seq, qh, dim)**2,axis=-1)+c['rms_norm_eps']
        k_variance = np.mean((n @ w('attention.k_weight.weight').T).reshape(seq, kh, dim)**2,axis=-1)+c['rms_norm_eps']
        if profile:
            for head in range(qh):q[:,head,:]=(n @ w('attention.q_weight.weight').T).reshape(seq,qh,dim)[:,head,:]*approximate(q_variance[:,head,None],profile['q_norm'][head])*w('attention.q_norm.weight')
            for head in range(kh):k[:,head,:]=(n @ w('attention.k_weight.weight').T).reshape(seq,kh,dim)[:,head,:]*approximate(k_variance[:,head,None],profile['k_norm'][head])*w('attention.k_norm.weight')
        q, k = rope(q), rope(k)
        v = (n @ w('attention.v_weight.weight').T).reshape(seq, kh, dim)
        keys.append(k.reshape(seq, -1).copy())
        values.append(v.reshape(seq, -1).copy())
        attended = np.empty((seq, qh, dim))
        centered_max = 0.
        for token in range(seq):
            for head in range(qh):
                group = head // (qh // kh)
                scores = k[:token+1, group] @ q[token, head] / math.sqrt(dim)
                centered_max = max(centered_max, float(np.abs(scores - scores.mean()).max()))
                if profile:
                    soft=profile['softmax'];center=scores-scores.mean()
                    if np.max(np.abs(center))>soft['centered_bound']:raise ValueError('softmax calibration domain exceeded')
                    numerator=sum((center/2**soft['squarings'])**j/math.factorial(j) for j in range(soft['exp_degree']+1))
                    def reciprocal(v,upper,it):
                        residual=1-v/upper;y=(1+residual)/upper
                        for _ in range(it-1):residual=residual*residual;y=y*(1+residual)
                        return y
                    it=soft['intermediate_iterations'] if soft['squarings'] else soft['reciprocal_iterations']
                    probability=numerator*reciprocal(numerator.sum(),2**math.ceil(math.log2(math.exp(soft['exp_bound'])*1.01))*len(scores),it)
                    for step in range(soft['squarings']):
                        numerator=probability**2;it=soft['reciprocal_iterations'] if step==soft['squarings']-1 else soft['intermediate_iterations']
                        probability=numerator*reciprocal(numerator.sum(),1.01,it)
                else:
                    probability = np.exp(scores - scores.max())
                    probability /= probability.sum()
                attended[token, head] = probability @ v[:token+1, group]
        hidden += attended.reshape(seq, qh*dim) @ w('attention.o_weight.weight').T
        n, post_bounds = norm(hidden, w('post_Normal.weight'),profile['post_norm'] if profile else None)
        gate = n @ w('mlplayer.gate_proj.weight').T
        up = n @ w('mlplayer.up_proj.weight').T
        activation=gate/(1+np.exp(-gate)) if profile is None else approximate(gate,profile['silu'])
        hidden += (activation * up) @ w('mlplayer.down_proj.weight').T
        bounds.append(dict(layer=layer, pre_norm_variance=pre_bounds, post_norm_variance=post_bounds,
                           q_norm_variance=[(float(q_variance[:,h].min()),float(q_variance[:,h].max())) for h in range(qh)],
                           k_norm_variance=[(float(k_variance[:,h].min()),float(k_variance[:,h].max())) for h in range(kh)],
                           gate_abs_max=float(np.abs(gate).max()), centered_score_abs_max=centered_max))
    final, final_bounds = norm(hidden[-1], weights['final_normal.weight'],profiles['final_norm'] if profiles else None)
    # Chunk the public LM matrix to bound conversion/matmul temporaries.
    lm = weights['lm_head.weight']
    logits = np.concatenate([lm[start:start+1024] @ final for start in range(0, len(lm), 1024)])
    return dict(logits=logits, keys=np.array(keys), values=np.array(values)), dict(
        layers=bounds, final_norm_variance=final_bounds)
