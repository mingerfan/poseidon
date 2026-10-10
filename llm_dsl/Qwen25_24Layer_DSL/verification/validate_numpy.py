"""Run actual CLI entrypoints with 24 small blocks and independent dense math."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys
import time
import types
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen24 import spec
from qwen24.weights import load
from qwen24.reference import forward, assert_provisional_bounds
from unpack_outputs import unpack
from check_outputs import check
from prepare_fixture import prepare


class Expr:
    slots = 32
    operations = Counter()

    def __init__(self, values):
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 1 or len(a) > self.slots:
            raise ValueError('Invalid slot vector')
        self.values = np.pad(a, (0, self.slots - len(a))) if len(a) < self.slots else a

    def __add__(self, other):
        if not isinstance(other, Expr):
            raise TypeError('Explicit Plain required')
        Expr.operations['add'] += 1
        return Expr(self.values + other.values)

    def __mul__(self, other):
        if not isinstance(other, Expr):
            raise TypeError('Explicit Plain required')
        Expr.operations['multiply'] += 1
        return Expr(self.values * other.values)

    def rotate(self, k):
        Expr.operations['rotate'] += 1
        return Expr(np.roll(self.values, -k))


class Plain(Expr):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error('Use a new output directory')
    a.output_dir.mkdir(parents=True)
    c = spec.config(ROOT / 'verification/smoke_config.json')
    fixture = a.output_dir / 'fixture'
    prepare(fixture, c, slots=32)
    weights = load(fixture / 'weights', c)
    embeddings = np.load(fixture / 'input_embeddings.npy')
    reference, bounds = forward(weights, embeddings, (0, 1), c)
    assert_provisional_bounds(bounds)
    module = types.ModuleType('qwen24.kernels.expr')
    module.Expr, module.Plain = Expr, Plain
    sys.modules[module.__name__] = module
    from qwen24.kernels import qwen_model as qm
    assert qm.weight_shapes(qm.QwenConfig(**c)) == spec.weight_shapes(c)
    assert len(spec.weight_shapes(spec.config())) == 291
    inputs = []
    for row in embeddings:
        dirty = np.full(32, .3)
        dirty[:c['hidden_size']] = row
        inputs.append(Expr(dirty))
    summaries = []

    def run(mode, positions, values, cached=(), tag=None):
        sig = spec.signature(c, mode, positions, cached, 32, 5)
        functions, calls, captured = [], [], []
        fake = types.ModuleType('hecate')
        def func(typespec):
            assert typespec == ','.join('c' for _ in sig['inputs'])
            def decorate(fn):
                functions.append(fn)
                return fn
            return decorate
        def save(*paths):
            assert len(functions) == 1
            captured.extend(functions[0](*values))
            return 'NumPy-only mock; no native Hecate tracing or CKKS'
        fake.func, fake.save = func, save
        original = qm.Qwen25Model._decoder_block
        def observe(model, hidden, pos, cache, index):
            calls.append((index, tuple(pos), tuple(cache.positions)))
            return original(model, hidden, pos, cache, index)
        target = a.output_dir / (tag or mode)
        argv = [str(ROOT / 'trace_qwen24.py'), '--mode', mode,
                '--config', str(ROOT / 'verification/smoke_config.json'),
                '--weights', str(fixture / 'weights'), '--slots', '32', '--logit-chunk-size', '5',
                '--output-dir', str(target), '--positions', *map(str, positions)]
        if cached:
            argv += ['--cached-positions', *map(str, cached)]
        Expr.operations.clear()
        start = time.monotonic()
        with patch.dict(sys.modules, {'hecate': fake}), patch.object(sys, 'argv', argv), \
             patch.object(qm.Qwen25Model, '_decoder_block', observe):
            runpy.run_path(argv[0], run_name='__main__')
        expected_indices = list(range(24)) * (2 if mode == 'decode-pair' else 1)
        assert [row[0] for row in calls] == expected_indices, calls
        if mode == 'decode-pair':
            assert all(pos == (0,) and old == () for _, pos, old in calls[:24])
            assert all(pos == (1,) and old == (0,) for _, pos, old in calls[24:])
        output = np.stack([v.values for v in captured])
        unpacked, tail = unpack(output, sig, 1e-10)
        expected, observed = forward(weights, embeddings[:len(cached)+len(positions)],
                                     tuple(cached)+tuple(positions), c)
        result = check(unpacked, expected, atol=1e-5)
        assert result['passed'], result
        # Mark mocked entrypoint artifacts explicitly; no fake native-success record.
        trace_sig = json.loads((target / 'signature.json').read_text())
        trace_sig['native_tracing_completed'] = False
        trace_sig['verification_backend'] = 'NumPy mock of hc.func/save'
        (target / 'signature.json').write_text(json.dumps(trace_sig, indent=2), encoding='utf-8')
        summaries.append(dict(mode=tag or mode, blocks_executed=len(calls),
                              unique_block_indices=sorted(set(row[0] for row in calls)),
                              input_ciphertexts=len(values), output_ciphertexts=len(captured),
                              max_abs_tail=tail, comparison=result, operations=dict(Expr.operations),
                              elapsed_seconds=time.monotonic()-start))
        return captured, unpacked

    _, full = run('prefill', (0, 1), inputs)
    _, paired = run('decode-pair', (0, 1), inputs)
    first, _ = run('prefill', (0,), inputs[:1], tag='prefill-one')
    # KV remains the same Expr objects between independent registered calls.
    first_cache = first[4:]
    snapshots = [item.values.copy() for item in first_cache]
    _, decoded = run('decode', (1,), [inputs[1]] + first_cache, cached=(0,))
    for old, snapshot in zip(first_cache, snapshots):
        np.testing.assert_array_equal(old.values, snapshot)
    differences = {}
    for label, candidate in [('decode-pair', paired), ('external-decode', decoded)]:
        differences[label] = {}
        for name in ('logits', 'keys', 'values'):
            delta = float(np.max(np.abs(candidate[name] - full[name])))
            assert delta < 1e-11, (label, name, delta)
            differences[label][name] = delta
    assert not np.allclose(full['keys'][0], full['keys'][-1])
    # Acceptance rejects broken outputs and invalid interfaces/profiles.
    corrupted = dict(full, logits=full['logits'] + .1)
    assert not check(corrupted, reference)['passed']
    rejected = []
    for label, mode, positions, cached in [('empty-cache', 'decode', (1,), ()),
                                          ('overlap', 'decode', (0,), (0,)),
                                          ('bad-pair', 'decode-pair', (0,), ())]:
        try:
            spec.signature(c, mode, positions, cached, 32, 5)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    with (a.output_dir / 'numpy_prefill.npz').open('xb') as f:
        np.savez(f, **full)
    report = dict(status='passed', scope='24 layers, reduced width, all real slot arithmetic and CLI wiring',
                  config=c, slots=32, logit_chunk_size=5, synthetic=True,
                  real_qwen_dimensions_numerically_tested=False, native_hecate=False, ckks=False, gpu=False,
                  checks=summaries, path_differences=differences, rejected_cases=rejected,
                  cache_preserved_without_mutation=True, reference_bounds=bounds)
    with (a.output_dir / 'validation.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'reference_bounds'}, indent=2))


if __name__ == '__main__':
    main()
