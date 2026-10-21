#!/usr/bin/env python3
"""Trace real first-token Qwen3 pre-norm/K/V; compare to the frozen full model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dsl', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['kv', 'projection'], default='kv')
    a = p.parse_args()
    sys.path.insert(0, str(a.dsl))
    import hecate as hc
    from qwen3.runtime import load_model
    from qwen3.kernels import qwen_layers as ql
    model = load_model(a.fixture/'weights', a.fixture/'model.json',
                       a.fixture/'approximations.json', 8192, 1024)
    cfg, profile = model.config, model.approximations.blocks[0]
    weight = lambda name: model.weights['model_list.0.'+name]

    @hc.func('c')
    def qwen3_prefix(image):
        x = ql.PackedVector(image, cfg.hidden_size, 8192)
        if a.stage == 'projection':
            return (ql.linear(x, weight('attention.k_weight.weight')).expr,
                    ql.linear(x, weight('attention.v_weight.weight')).expr)
        n = ql.rms_norm(x, weight('pre_Normal.weight'), rsqrt=profile.pre_norm,
                        eps=cfg.rms_norm_eps)
        k = ql.linear(n, weight('attention.k_weight.weight'))
        v = ql.linear(n, weight('attention.v_weight.weight'))
        k = ql.grouped_head_rms_norm(k, weight('attention.k_norm.weight'),
                                    profiles=profile.k_norm, heads=cfg.kv_heads,
                                    head_dim=cfg.head_dim, eps=cfg.rms_norm_eps)
        k = ql.rope(k, 0, head_dim=cfg.head_dim, theta=cfg.rope_theta)
        return k.expr, v.expr

    a.output.mkdir(parents=True, exist_ok=False)
    print(hc.save(str(a.output), str(a.output), bytecode=True))
    embeddings = np.load(a.fixture/'embeddings.npy')
    reference = np.load(a.fixture/'reference-approximate.npz')
    expected = [reference[field][0, 0] for field in ['keys', 'values']]
    if a.stage == 'projection':
        expected = [np.asarray(weight(f'attention.{field}_weight.weight'), dtype=np.float64) @ embeddings[0]
                    for field in ['k', 'v']]
    fixture = dict(format_version=1, model=dict(
        name='Real Qwen3-0.6B first-token layer0 K/V prefix',
        complete_qwen=False, full_model_ckks_validated=False,
        reference='Frozen full 28-layer approximate FP64 forward; layer0/token0'),
        context=dict(poly_degree=16384, q_moduli_bits=[30]*50,
                     p_moduli_bits=[30]*8, default_scale_log2=40),
        tolerances=dict(atol=0.001, rtol=0.0001),
        inputs=[dict(name='embedding.position_0', values=embeddings[0].tolist())],
        expected_outputs=[dict(name=name, values=values.tolist())
                          for name, values in zip(['layer0.K.position_0', 'layer0.V.position_0'], expected)])
    fixture['model']['stage'] = a.stage
    (a.output/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((a.fixture/'reference-approximate.npz').read_bytes()).hexdigest(),
        weights_manifest_sha256=hashlib.sha256((a.fixture/'weights/weights.json').read_bytes()).hexdigest(),
        config=json.loads((a.fixture/'model.json').read_text()),
        scope='Prefix through first layer K/V, not complete inference'), indent=2)+'\n')


if __name__ == '__main__':
    main()
