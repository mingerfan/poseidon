"""Nested closure goldens through existing Dacapo/SEAL CPU; no paid API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('construction-linear', 'linear', False),
         ('arithmetic-alias-chain', 'state', False),
         ('arithmetic-alias-chain', 'late_binding', False),
         ('construction-linear', 'wrong_linear', True),
         ('arithmetic-alias-chain', 'wrong_state', True),
         ('arithmetic-alias-chain', 'wrong_late_binding', True)]

if __name__ == '__main__':
    import unittest
    from test_closure_construction import ClosureGoldenPlainTests
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ClosureGoldenPlainTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(main(plans=PLANS, golden_dir=ROOT/'scripts/baseline/golden_cases/closures',
        prefix='closure-goldens-', title='Closures', input_counts=(1,), extra_options=('--closures',)))
