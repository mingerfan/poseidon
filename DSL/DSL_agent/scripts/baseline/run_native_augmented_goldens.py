"""Manual native augmented scalar dispatch through real FHE; never a paid API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

CASES=('add','subtract','multiply_alias','array_alias','plain_left','nested_loop','sum4',
       'wrong_alias','wrong_order')
PLANS=[('construction-sumslots4' if n=='sum4' else 'arithmetic-alias-chain',n,n.startswith('wrong_'))
       for n in CASES]

if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/native_augmented',
        prefix='native-augmented-goldens-',title='Native augmented scalar goldens',input_counts=(1,),
        extra_options=('--native-scalar-augmented','--compiler-configuration','seal-cpu-eva-w45-v1')))
