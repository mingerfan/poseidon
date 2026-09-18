"""Manual typed-public-operand fixtures; no inference or native execution here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / 'scripts/baseline/golden_cases/broadcast'
# Registry expected from the fixed graph translation, NOT modified after tracing.
# Subtraction lowering exports the negative public value; -(-x-c) equals x+c.
FIXTURES = {
    'add_one': dict(case='broadcast-add-one', constants={'c0': [.375]},
                    features=('add.length1', 'statement.alias'), outputs=1),
    'mul_one': dict(case='broadcast-mul-one', constants={'c0': [-.5]},
                    features=('multiply.length1',), outputs=1),
    'sub_one': dict(case='broadcast-sub-one', constants={'c0': [-.375]},
                    features=('subtract.length1',), outputs=1),
    'sub_four': dict(case='broadcast-sub-four', constants={'c0': [-.125, .25, -.5, -.75]},
                     features=('subtract.length4',), outputs=1),
    'sub_scalar': dict(case='broadcast-sub-scalar',
                       constants={'c0': [.5, -.25, .125, .75], 'c1': -.375},
                       features=('subtract.scalar',), outputs=1),
}
PLANS = [(row['case'], name, False) for name, row in FIXTURES.items()] + [
    (row['case'], 'wrong_' + name, True) for name, row in FIXTURES.items()]
