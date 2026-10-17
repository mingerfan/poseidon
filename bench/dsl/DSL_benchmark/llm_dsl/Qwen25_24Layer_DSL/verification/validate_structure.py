"""Full default dimensions, shape-only traversal; deliberately not numeric/CKKS."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Expr:
    def __add__(self, other):
        return Expr()

    def __mul__(self, other):
        return Expr()

    def rotate(self, k):
        return Expr()


class Plain(Expr):
    def __init__(self, value):
        pass


def main():
    module = types.ModuleType('qwen24.kernels.expr')
    module.Expr, module.Plain = Expr, Plain
    sys.modules[module.__name__] = module
    from qwen24 import spec
    from qwen24.kernels import qwen_layers as ql, qwen_model as qm
    from qwen24.runtime import evaluate
    c = spec.config()
    cfg = qm.QwenConfig(**c)
    assert qm.weight_shapes(cfg) == spec.weight_shapes(c)
    weights = {name: np.broadcast_to(np.float32(.01), shape) for name, shape in spec.weight_shapes(c).items()}
    profiles = qm.ModelApproximations.from_dict(json.loads(spec.PROFILES.read_text()))
    model = qm.Qwen25Model(weights, profiles, config=cfg, slots=32768, logit_chunk_size=1024)
    calls, matrices = [], []
    original = qm.Qwen25Model._decoder_block
    def observe(instance, hidden, positions, cache, index):
        calls.append(index)
        return original(instance, hidden, positions, cache, index)
    def shape_linear(x, weight, bias=None):
        assert weight.ndim == 2 and weight.shape[1] == x.size and weight.shape[0] <= x.slots
        if bias is not None:
            assert bias.shape == (weight.shape[0],)
        matrices.append(tuple(weight.shape))
        return ql.PackedVector(Expr(), weight.shape[0], x.slots)
    rows = []
    for mode, positions, cached in [('prefill', (0, 1), ()), ('decode-pair', (0, 1), ()),
                                    ('prefill', (0,), ()), ('decode', (1,), (0,))]:
        calls.clear()
        matrices.clear()
        sig = spec.signature(c, mode, positions, cached)
        with patch.object(ql, 'linear', shape_linear), patch.object(qm.Qwen25Model, '_decoder_block', observe):
            result = evaluate(model, sig, [Expr() for _ in sig['inputs']])
        assert calls == list(range(24)) * (2 if mode == 'decode-pair' else 1)
        assert sig['logit_ciphertexts'] == 149
        assert sig['outputs'][148]['prefix_size'] == 384
        expected_tokens = 1 if len(positions) == 1 else 2
        assert len(matrices) == 24 * 7 * expected_tokens + 149
        assert matrices[-149:] == [(1024, 896)] * 148 + [(384, 896)]
        assert len(result) == 149 + 24 * 2 * (len(positions) + len(cached))
        rows.append(dict(mode=mode, positions=positions, cached_positions=cached,
                         block_calls=len(calls), layers=sorted(set(calls)),
                         matrix_calls=len(matrices), input_ciphertexts=len(sig['inputs']),
                         output_ciphertexts=len(result), logit_ciphertexts=149))
    print(json.dumps(dict(status='passed', config=asdict(cfg), checks=rows,
                          scope='Full-size shape traversal with mocked linear arithmetic and symbolic expressions',
                          numeric_arithmetic=False, native_hecate=False, ckks=False, gpu=False), indent=2))


if __name__ == '__main__':
    main()
