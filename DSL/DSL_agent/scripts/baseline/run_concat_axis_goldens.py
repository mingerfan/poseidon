"""Discriminating axis0 positive and wrong-axis real FHE counterexample."""
from hecate_python_env import ROOT
from run_schema3_goldens import main
PLANS=[('concat-axis0-sensitive','concat-axis0-sensitive',False),
       ('concat-axis0-sensitive','wrong-sensitive-axis',True)]
if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/concat',
        prefix='concat-axis-goldens-',title='Concat axis discriminator',input_counts=(2,),extra_options=('--object-unary',)))
