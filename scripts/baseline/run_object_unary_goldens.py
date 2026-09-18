"""Manual object unary goldens through real CPU encryption; no paid API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

if __name__=='__main__':
    names=('neg','zero','view','positive','negative_call','positive_call','plain','mixed','wrong_neg','wrong_copy')
    raise SystemExit(main(
        plans=[('arithmetic-alias-chain',name,name.startswith('wrong_')) for name in names],
        golden_dir=ROOT/'scripts/baseline/golden_cases/object_unary',
        prefix='object-unary-goldens-',title='Object unary',input_counts=(1,),
        extra_options=('--object-unary',)))
