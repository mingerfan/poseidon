"""Actual Dacapo/SEAL CPU validation for public sequence and alias semantics."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS=[('construction-linear','linear',False),
       ('arithmetic-alias-chain','repeat',False),
       ('arithmetic-alias-chain','write',False),
       ('construction-linear','wrong_linear',True),
       ('arithmetic-alias-chain','wrong_repeat',True),
       ('arithmetic-alias-chain','wrong_write',True)]

if __name__=='__main__':
    import unittest
    from test_public_sequences import SequenceGoldenPlainTests
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(SequenceGoldenPlainTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/public_sequences',
        prefix='public-sequence-goldens-',title='Public sequences',input_counts=(1,),extra_options=('--public-sequences',)))
