"""Public construction goldens and counterexamples through real CPU FHE; no API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('construction-linear', 'linear', False),
         ('arithmetic-alias-chain', 'list_alias', False),
         ('custom-dual-subtract', 'dual_subtract', False),
         ('construction-linear', 'wrong_linear', True),
         ('arithmetic-alias-chain', 'wrong_list_alias', True)]

if __name__ == '__main__':
    raise SystemExit(main(plans=PLANS, golden_dir=ROOT/'scripts/baseline/golden_cases/public_construction',
        prefix='public-construction-goldens-', title='Public construction', input_counts=(1, 2),
        extra_options=('--public-construction',)))
