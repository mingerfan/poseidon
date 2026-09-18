"""Manual positional-unpacking semantic tests, no model API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES=('list','tuple','array','reverse','matrix_flatten','nested','multiple','empty','wrong_order')

if __name__=='__main__':
    raise SystemExit(main(
        plans=[('arithmetic-alias-chain',name,name=='wrong_order') for name in CASES],
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_starred',
        prefix='native-starred-goldens-',title='Native starred argument goldens',input_counts=(1,),
        extra_options=('--native-starred',)))
