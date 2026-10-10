"""Compare decrypted GPU output slots with the fixed original-function reference."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def check(path,prefix_only=False):
    value=np.load(path,allow_pickle=False)
    if value.dtype.kind not in 'fci' or not np.isfinite(value).all():
        raise ValueError('Expected finite numeric NPY output')
    a=value.reshape(-1)
    if len(a)!=(1792 if prefix_only else 32768):
        raise ValueError('Expected 32768 decrypted slots (or 1792 with --prefix-only)')
    reference=np.load(ROOT/'data/reference_hidden.npy',allow_pickle=False).reshape(-1)
    error=float(np.max(np.abs(a[:1792]-reference)))
    tail=None if prefix_only else float(np.max(np.abs(a[1792:])))
    return a,dict(file=str(path),max_abs_error=error,max_abs_tail=tail,
        imaginary_components_checked=bool(np.iscomplexobj(a)),
        tail_checked=not prefix_only,threshold=1e-3,
        compared_components_passed=error<=1e-3 and (tail is None or tail<=1e-3),
        note='Real-only output cannot certify imaginary error; prefix-only cannot certify tail.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefill',type=Path,required=True)
    parser.add_argument('--decode',type=Path,required=True)
    parser.add_argument('--prefix-only',action='store_true',help='Check 1792 values only; no tail claim')
    args=parser.parse_args()
    a,pa=check(args.prefill,args.prefix_only); b,pb=check(args.decode,args.prefix_only)
    difference=float(np.max(np.abs(a[:1792]-b[:1792])))
    passed=pa['compared_components_passed'] and pb['compared_components_passed'] and difference<=1e-3
    print(json.dumps(dict(prefill=pa,decode=pb,prefill_decode_max_abs_difference=difference,
        compared_components_passed=passed,source='caller-supplied decrypted outputs; tool does not attest GPU execution'),indent=2))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
