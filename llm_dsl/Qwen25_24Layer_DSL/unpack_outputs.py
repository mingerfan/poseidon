"""Unpack decrypted full-slot outputs using the generated ciphertext signature."""
import argparse
import json
from pathlib import Path
import numpy as np


def unpack(values, sig, tail_atol=1e-3):
    a = np.asarray(values)
    if a.shape != (len(sig['outputs']), sig['slots']) or a.dtype.kind not in 'fci':
        raise ValueError('Expected one full decrypted slot row per signature output')
    if not np.isfinite(a).all() or not np.isfinite(tail_atol) or tail_atol < 0:
        raise ValueError('Expected finite output and nonnegative tail tolerance')
    tails = [float(np.abs(row[spec['prefix_size']:]).max(initial=0)) for row, spec in zip(a, sig['outputs'])]
    if max(tails) > tail_atol:
        raise ValueError('Output tail exceeds tolerance: ' + str(max(tails)))
    n = sig['logit_ciphertexts']
    logits = np.concatenate([a[i, :sig['outputs'][i]['prefix_size']] for i in range(n)])
    count = len(sig['new_positions']) + len(sig['cached_positions'])
    kv = sig['config']['kv_heads'] * sig['config']['head_dim']
    cache = a[n:, :kv].reshape(sig['config']['num_layers'], 2, count, kv)
    return dict(logits=logits, keys=cache[:, 0], values=cache[:, 1]), max(tails)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--signature', type=Path, required=True)
    p.add_argument('--slots', type=Path, required=True, help='Decrypted NPY [output ciphertexts, slots]')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tail-atol', type=float, default=1e-3)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output path')
    sig = json.loads(a.signature.read_text(encoding='utf-8'))
    result, tail = unpack(np.load(a.slots, allow_pickle=False), sig, a.tail_atol)
    with a.output.open('xb') as f:
        np.savez(f, **result)
    print(json.dumps(dict(max_abs_tail=tail, tail_checked=True,
                         imaginary_checked=bool(np.iscomplexobj(result['logits'])))))


if __name__ == '__main__':
    main()
