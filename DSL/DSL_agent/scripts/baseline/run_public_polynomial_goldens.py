"""Exact upstream GenPoly through existing Dacapo/SEAL CPU, no API calls."""
import shlex
from hecate_python_env import ROOT,VENV,enter_nix
from run_schema3_goldens import main

PLANS = [('construction-genpoly3','leaf',False),
         ('construction-genpoly3','split',False),
         ('construction-genpoly7','seventh',False),
         ('construction-genpoly3','wrong_leaf',True),
         ('construction-genpoly3','wrong_split',True),
         ('construction-genpoly7','wrong_seventh',True),
         ('construction-genpoly-even','even_unsupported',True),
         ('construction-genpoly5','missing_giant',False)]

if __name__ == '__main__':
    # System Python has no NumPy. Precheck in the existing isolated dependency
    # environment; release its lock before per-case native compiler runs.
    command = ('LD_LIBRARY_PATH=$HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '
               'PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/'bin/python'),
               '-m','unittest','test_public_polynomial.PolynomialGoldenPlainTests','-q']))
    code = enter_nix(command,seconds=120)
    if code: raise SystemExit(code)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/public_polynomial',
        prefix='public-polynomial-goldens-',title='Upstream GenPoly',input_counts=(1,),
        extra_options=('--public-polynomial',)))
