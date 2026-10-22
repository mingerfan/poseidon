"""Object array / upstream SumSlots through existing Dacapo/SEAL CPU; no API."""
import shlex
from hecate_python_env import ROOT,VENV,enter_nix
from run_schema3_goldens import main

PLANS = [('construction-sumslots4','sumslots4',False),
         ('construction-sumslots3','sumslots3',False),
         ('construction-linear','linear',False),
         ('arithmetic-alias-chain','empty_views',False),
         ('construction-sumslots3','wrong_stride',True),
         ('arithmetic-alias-chain','wrong_empty',True)]

if __name__ == '__main__':
    command = ('LD_LIBRARY_PATH=$HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '
               'PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/'bin/python'),
               '-m','unittest','test_object_arrays.ObjectGoldenPlainTests','-q']))
    code = enter_nix(command,seconds=120)
    if code: raise SystemExit(code)
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/object_arrays',
        prefix='object-array-goldens-',title='Object arrays / upstream SumSlots',input_counts=(1,),
        extra_options=('--object-arrays',)))
