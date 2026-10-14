"""Explicit graph powers 2/4 and wrong-exponent counterexamples; no model API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('explicit-power-2','power2',False), ('explicit-power-4','power4',False),
         ('explicit-power-2','wrong_power2',True), ('explicit-power-4','wrong_power4',True)]

if __name__ == '__main__':
    raise SystemExit(main(plans=PLANS, golden_dir=ROOT/'scripts/baseline/golden_cases/explicit_power',
                          prefix='explicit-power-goldens-', title='Explicit power', input_counts=(1,)))
