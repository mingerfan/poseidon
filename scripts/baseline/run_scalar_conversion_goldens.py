"""Manual v21 baseline; real CPU HEVM execution, zero API calls."""
from hecate_python_env import ROOT
from scalar_conversion_exercises import EXERCISES
from run_schema3_goldens import main

if __name__=='__main__':
    plans=[('exercise-'+name,name,False) for name in EXERCISES]
    plans += [('exercise-sc-int-negative','wrong_int',True),('exercise-sc-item-flat','wrong_item',True)]
    raise SystemExit(main(plans=plans,golden_dir=ROOT/'scripts/baseline/golden_cases/scalar_conversion',
        prefix='scalar-conversion-goldens-',title='Scalar conversion',input_counts=(1,),
        extra_options=('--scalar-conversion',)))
