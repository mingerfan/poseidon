"""Check supplied decrypted prefixes against an independent reference NPZ."""
import argparse
import json
import numpy as np


def check(actual, reference, atol=1e-3, rtol=0.):
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    rows = {}
    for name in ('logits', 'keys', 'values'):
        a, b = np.asarray(actual[name]), np.asarray(reference[name])
        if a.shape != b.shape or a.size == 0 or a.dtype.kind not in 'fci' or b.dtype.kind not in 'fci':
            raise ValueError('Shape/type mismatch: ' + name)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite output: ' + name)
        error = np.abs(a - b)
        rows[name] = dict(max_abs_error=float(error.max()),
                          passed=bool(np.all(error <= atol + rtol*np.abs(b))),
                          imaginary_checked=bool(np.iscomplexobj(a)))
    return dict(passed=all(row['passed'] for row in rows.values()), arrays=rows,
                atol=atol, rtol=rtol, tail_checked=False,
                execution_provenance='caller-supplied arrays; does not attest GPU or CKKS execution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--rtol', type=float, default=0.)
    a = p.parse_args()
    with np.load(a.actual, allow_pickle=False) as actual, np.load(a.reference, allow_pickle=False) as reference:
        result = check(actual, reference, a.atol, a.rtol)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
