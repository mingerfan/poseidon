"""Manual reverse/augmented/alias goldens, actual Dacapo/SEAL, zero API calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('broadcast-add-one', 'reverse_add', False),
         ('broadcast-mul-one', 'reverse_multiply', False),
         ('broadcast-sub-one', 'reverse_subtract', False),
         ('broadcast-sub-four', 'reverse_subtract', False),
         ('arithmetic-alias-chain', 'alias_chain', False),
         ('broadcast-sub-four', 'wrong_reverse_subtract', True),
         ('arithmetic-alias-chain', 'wrong_alias_chain', True)]

if __name__ == '__main__':
    raise SystemExit(main(plans=PLANS,
        golden_dir=ROOT/'scripts/baseline/golden_cases/extended_arithmetic',
        prefix='extended-arithmetic-goldens-', title='Extended arithmetic',
        input_counts=(1,), extra_options=('--extended-arithmetic',)))
