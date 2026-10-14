"""Explicit encrypted-zero ABI goldens/counterexamples; no model API calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

PLANS = [
    ('zero-linear-pure', 'pure', False),
    ('zero-linear-mixed', 'mixed', False),
    ('zero-mlp-hidden', 'hidden', False),
    ('zero-conv', 'conv', False),
    ('zero-quad', 'quad', False),
    ('zero-multiply', 'multiply', False),
    ('zero-linear-pure', 'wrong_pure', True),
    ('zero-linear-mixed', 'wrong_mixed', True),
    ('zero-quad', 'wrong_quad', True),
]

if __name__ == '__main__':
    raise SystemExit(main(plans=PLANS, golden_dir=ROOT/'scripts/baseline/golden_cases/encrypted_zero',
                          prefix='zero-golden-batch-', title='Encrypted-zero', input_counts=(2, 5)))
