"""Compare decoded outputs with CKKS error bounds and class agreement."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    def load(path):
        if str(path).endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}
        return {"output": np.load(path, allow_pickle=False)}

    actual, reference = load(args.actual), load(args.reference)
    records = {}
    passed = set(actual) == set(reference)
    for name in reference.keys() & actual.keys():
        x, ref = actual[name], reference[name]
        same_shape = x.shape == ref.shape
        finite = bool(np.isfinite(x).all() and np.isfinite(ref).all())
        ok = bool(same_shape and finite and np.all(np.abs(x-ref) <= 1e-5+1e-4*np.abs(ref)))
        records[name] = dict(ok=ok, exact_equal=bool(np.array_equal(x, ref)),
                             max_abs_diff=float(np.max(np.abs(x-ref))) if same_shape and finite else None,
                             argmax_agreement=float(np.mean(x.argmax(axis=-1) == ref.argmax(axis=-1)))
                             if same_shape and finite and x.ndim else None)
        passed = passed and ok
    print(json.dumps(dict(ok=passed, outputs=records), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
