"""Actual Q*P / balanced sparse-secret estimate. Never treats missing attacks as pass.

Run with a Sage-enabled Python. No keys or secret coefficients are read/written.
The default noise is the application's actual CBD(21), not Gaussian 3.19.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chain', type=Path, default=Path(__file__).with_name('q50.json'))
    ap.add_argument('--estimator', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dnum', type=int, default=2)
    ap.add_argument('--hamming', type=int, default=192)
    ap.add_argument('--model', choices=['ADPS16', 'BDGL16', 'MATZOV'], default='ADPS16')
    ap.add_argument('--attacks', default='usvp')
    args = ap.parse_args()
    if not 1 <= args.dnum <= 50 or not 0 < args.hamming <= 65536 or args.hamming % 2:
        ap.error('invalid dnum or balanced sparse-secret weight')
    sys.path.insert(0, str(args.estimator))
    from sage.all import ZZ, oo, RealField
    from estimator import LWE, ND, RC
    chain = json.loads(args.chain.read_text())
    q = chain['q_bottom_first']
    count = math.ceil(len(q) / args.dnum)
    p = chain['p'][:count]
    if len(p) != count:
        ap.error('witness does not contain enough P primes')
    modulus = ZZ(math.prod(q + p))
    params = LWE.Parameters(n=65536, q=modulus,
        Xs=ND.SparseTernary(args.hamming // 2, args.hamming // 2, 65536),
        Xe=ND.CenteredBinomial(21), m=oo, tag='ResNet20-Q50-public-parameters')
    model = getattr(RC, args.model)
    methods = {
        'usvp': lambda: LWE.primal_usvp(params, red_cost_model=model, red_shape_model='gsa'),
        'bdd': lambda: LWE.primal_bdd(params, red_cost_model=model, red_shape_model='gsa'),
        'dual': lambda: LWE.dual(params, red_cost_model=model),
        'dual_hybrid': lambda: LWE.dual_hybrid(params, red_cost_model=model),
        'bdd_hybrid': lambda: LWE.primal_hybrid(params, red_cost_model=model,
            red_shape_model='gsa', mitm=False, babai=False),
        'bdd_mitm_hybrid': lambda: LWE.primal_hybrid(params, red_cost_model=model,
            red_shape_model='gsa', mitm=True, babai=True),
    }
    report = dict(n=65536, q_count=len(q), p_count=len(p), dnum=args.dnum,
        hamming=args.hamming, positive=args.hamming//2, negative=args.hamming//2,
        error='CenteredBinomial(21)', error_stddev=math.sqrt(10.5),
        samples='unbounded', model=args.model, shape='gsa',
        log2_Q=float(RealField(256)(math.prod(q)).log2()),
        log2_QP=float(RealField(256)(modulus).log2()),
        q=q, p=p, threshold=128, status='NOT_VALIDATED', attacks={},
        estimator_files_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((args.estimator/'estimator').glob('*.py'))})
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    for name in args.attacks.split(','):
        if name not in methods:
            ap.error('unknown attack '+name)
        started = time.monotonic()
        print('START', name, 'logQP', report['log2_QP'], 'h', args.hamming, flush=True)
        try:
            cost = methods[name]()
            bits = float(RealField(256)(cost['rop']).log2()) if cost.get('rop', oo) != oo else None
            if bits is not None and not math.isfinite(bits):
                bits = None
            result = dict(bits=bits, cost=str(cost), seconds=time.monotonic()-started)
            if bits is not None and bits < report['threshold']:
                report['status'] = 'REJECT_BELOW_128'
        except Exception as ex:
            result = dict(bits=None, error=repr(ex), seconds=time.monotonic()-started)
        report['attacks'][name] = result
        save()
        print(name, json.dumps(result), flush=True)
    print('STATUS', report['status'], flush=True)
    # A screening tool can reject on one attack. It must not certify security
    # from only uSVP or from skipping failed/expensive sparse hybrid attacks.
    return 2 if report['status'] == 'REJECT_BELOW_128' else 0


if __name__ == '__main__':
    raise SystemExit(main())
