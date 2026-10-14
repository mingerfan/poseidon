"""Lexical construction function goldens through real Dacapo/SEAL; zero API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [('construction-linear', 'linear', False),
         ('explicit-power-4', 'quartic', False),
         ('arithmetic-alias-chain', 'list_alias', False),
         ('construction-linear', 'wrong_linear', True),
         ('explicit-power-4', 'wrong_quartic', True),
         ('arithmetic-alias-chain', 'wrong_list_alias', True)]

if __name__ == '__main__':
    import argparse
    import unittest
    from test_function_construction import FunctionGoldenPlainTests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quartic-only', action='store_true', help='Rerun only the corrected quartic pair; keep earlier evidence')
    args = parser.parse_args()
    preflight = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(FunctionGoldenPlainTests))
    if not preflight.wasSuccessful():
        raise SystemExit(1)
    plans = [p for p in PLANS if p[0] == 'explicit-power-4'] if args.quartic_only else PLANS
    raise SystemExit(main(plans=plans, golden_dir=ROOT/'scripts/baseline/golden_cases/function_composition',
        prefix='function-composition-goldens-', title='Function composition', input_counts=(1,),
        extra_options=('--function-composition',)))
