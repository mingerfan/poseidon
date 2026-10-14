"""Manual native-loop programs, real compiler/SEAL CPU, never a model API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES=('sum4','descending','nested','index_binding','helper_repeated','array_loop','zero_trip','public_weight','wrong_bound')
PLANS=[('construction-sumslots4' if n=='sum4' else
        'construction-sumslots3' if n in ('descending','wrong_bound') else
        'arithmetic-alias-chain',n,n=='wrong_bound') for n in CASES]

if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/native_loops',
        prefix='native-loop-goldens-',title='Native public loop goldens',input_counts=(1,),
        extra_options=('--native-public-loops',)))
