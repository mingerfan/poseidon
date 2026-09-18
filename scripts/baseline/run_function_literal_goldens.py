"""Real SEAL CPU validation for lambda capture and public stable sorting."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS=[('construction-linear','linear',False),
       ('arithmetic-alias-chain','defaults',False),
       ('arithmetic-alias-chain','stable',False),
       ('construction-linear','wrong_linear',True),
       ('arithmetic-alias-chain','wrong_defaults',True),
       ('arithmetic-alias-chain','wrong_stable',True)]

if __name__=='__main__':
    import unittest
    from test_function_literals import LiteralGoldenPlainTests
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(LiteralGoldenPlainTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/function_literals',
        prefix='function-literal-goldens-',title='Function literals',input_counts=(1,),extra_options=('--function-literals',)))
