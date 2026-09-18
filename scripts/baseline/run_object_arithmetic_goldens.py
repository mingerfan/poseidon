"""Real CPU encrypted checks of object arithmetic; manual, no paid API."""
from hecate_python_env import ROOT
from run_schema3_goldens import main

if __name__=='__main__':
    names=('broadcast','alias','overlap','empty','wrong_alias','wrong_empty')
    raise SystemExit(main(
        plans=[('arithmetic-alias-chain',name,name.startswith('wrong_')) for name in names],
        golden_dir=ROOT/'scripts/baseline/golden_cases/object_arithmetic',
        prefix='object-arithmetic-goldens-',title='Object arithmetic',
        input_counts=(1,),extra_options=('--object-arithmetic',)))

