"""Scoped comprehensions and lazy public iterators through actual SEAL CPU."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS=[('construction-linear','linear',False),
       ('arithmetic-alias-chain','lazy_zip',False),
       ('arithmetic-alias-chain','dict_values',False),
       ('construction-linear','wrong_linear',True),
       ('arithmetic-alias-chain','wrong_lazy_zip',True),
       ('arithmetic-alias-chain','wrong_dict_values',True)]

if __name__=='__main__':
    import unittest
    from test_public_iteration import IterationGoldenPlainTests
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(IterationGoldenPlainTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/public_iteration',
        prefix='public-iteration-goldens-',title='Public iteration',input_counts=(1,),extra_options=('--public-iteration',)))
