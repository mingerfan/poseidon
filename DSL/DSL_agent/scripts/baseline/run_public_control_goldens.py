"""Actual Dacapo/SEAL CPU evidence for bounded public control; no API calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('construction-chebyshev3','chebyshev',False),
         ('construction-linear','linear',False),
         ('arithmetic-alias-chain','affine',False),
         ('construction-chebyshev3','wrong_chebyshev',True),
         ('construction-linear','wrong_linear',True),
         ('arithmetic-alias-chain','wrong_affine',True)]

if __name__ == '__main__':
    import unittest
    from test_public_control import ControlGoldenPlainTests
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ControlGoldenPlainTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/public_control',
        prefix='public-control-goldens-',title='Public control',input_counts=(1,),extra_options=('--public-control',)))
