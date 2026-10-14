"""Actual Dacapo/SEAL CPU validation of public arrays and derived constants."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS=[('construction-linear','linear',False),
       ('arithmetic-alias-chain','scalar',False),
       ('arithmetic-alias-chain','broadcast',False),
       ('construction-linear','wrong_linear',True),
       ('arithmetic-alias-chain','wrong_scalar',True),
       ('arithmetic-alias-chain','wrong_broadcast',True)]

if __name__=='__main__':
    import unittest
    from test_public_numbers import NumericGoldenPlainTests
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericGoldenPlainTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/public_numbers',
        prefix='public-number-goldens-',title='Public numbers',input_counts=(1,),extra_options=('--public-numbers',)))
