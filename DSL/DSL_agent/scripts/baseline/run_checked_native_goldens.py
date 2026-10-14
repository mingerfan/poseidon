"""Manual native-function candidates through the real Agent sandbox pipeline; no API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

if __name__=='__main__':
    raise SystemExit(main(plans=[('arithmetic-alias-chain','affine',False),
        ('arithmetic-alias-chain','pair',False),('arithmetic-alias-chain','forward',False),
        ('custom-dual-subtract','dual',False),('custom-dual-subtract','wrong_dual',True)],
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_functions',
        prefix='checked-native-goldens-',title='Checked native candidates',input_counts=(1,2),
        extra_options=('--native-functions',)))
