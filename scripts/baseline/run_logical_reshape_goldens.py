"""Logical reshape and Conv/BatchNorm composition: manual real FHE tests."""
from hecate_python_env import ROOT
from run_schema3_goldens import main
NAMES=('reshape-bn','reshape-conv1-bn','reshape-conv2-bn')
PLANS=[(n,n,False) for n in NAMES]+[('reshape-bn','wrong-order',True),('reshape-conv1-bn','wrong-conv-channels',True)]
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--conv2-only',action='store_true')
    args=parser.parse_args()
    raise SystemExit(main(plans=[PLANS[2]] if args.conv2_only else PLANS,
        golden_dir=ROOT/'scripts/baseline/golden_cases/logical_reshape',
        prefix='logical-reshape-conv2-' if args.conv2_only else 'logical-reshape-goldens-',
        title='Logical reshape',input_counts=(1,),extra_options=('--object-unary',)))
