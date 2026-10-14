"""Real encrypted object-array arithmetic goldens; no model API calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES=('vector','broadcast','zero_dim','negate','mixed','nested','square','empty','wrong_broadcast')
PLANS=[('explicit-power-2' if name=='square' else 'arithmetic-alias-chain',name,
        name=='wrong_broadcast') for name in CASES]

if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_array_arithmetic',
        prefix='native-array-arithmetic-goldens-',title='Native array arithmetic goldens',
        input_counts=(1,),extra_options=('--native-array-arithmetic',)))
