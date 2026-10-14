"""Manual native array inplace/view programs, real compiler/SEAL, never API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES=('direct_alias','slice_transition','transpose','copy_flatten','reshape_view','reshape_copy',
       'overlap','zero_dim','empty','callee_fresh','loop','wrong_alias','wrong_overlap')
PLANS=[('arithmetic-alias-chain',n,n.startswith('wrong_')) for n in CASES]
LAYOUT_CASES=('constructor_layout','ufunc_layout')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-only',action='store_true',help='Two independent order-K boundary programs; preserve original 13-case batch')
    args=parser.parse_args()
    plans=[('arithmetic-alias-chain',n,False) for n in LAYOUT_CASES] if args.layout_only else PLANS
    raise SystemExit(main(plans=plans,golden_dir=ROOT/'scripts/baseline/golden_cases/native_array_mutation',
        prefix='native-array-layout-goldens-' if args.layout_only else 'native-array-mutation-goldens-',title='Native array inplace goldens',input_counts=(1,),
        extra_options=('--native-array-mutation','--compiler-configuration','seal-cpu-eva-w45-v1')))
