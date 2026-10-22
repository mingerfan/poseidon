"""Actual CPU goldens for definition-time defaults and argument binding; zero API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS=[('construction-linear','linear',False),
       ('arithmetic-alias-chain','snapshot',False),
       ('arithmetic-alias-chain','mutable_default',False),
       ('arithmetic-alias-chain','forwarding',False),
       ('construction-linear','wrong_linear',True),
       ('arithmetic-alias-chain','wrong_snapshot',True),
       ('arithmetic-alias-chain','wrong_mutable_default',True),
       ('arithmetic-alias-chain','wrong_forwarding',True)]

if __name__=='__main__':
    import argparse
    import unittest
    from test_construction_calls import CallGoldenPlainTests
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only',action='store_true',help='Rerun only the snapshot pair; preserve prior evidence')
    args=parser.parse_args()
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(CallGoldenPlainTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    plans=[p for p in PLANS if p[1] in ('snapshot','wrong_snapshot')] if args.snapshot_only else PLANS
    raise SystemExit(main(plans=plans,golden_dir=ROOT/'scripts/baseline/golden_cases/call_binding',
        prefix='call-binding-goldens-',title='Call binding',input_counts=(1,),extra_options=('--call-binding',)))
