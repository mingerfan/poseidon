"""Manual typed-array cohort with actual storage observation; never calls APIs."""
from hecate_python_env import ROOT
from native_array_exercises import EXERCISES
from run_schema3_goldens import main

if __name__ == '__main__':
    raise SystemExit(main(
        plans=[('exercise-'+name,spec[0],False) for name,spec in EXERCISES.items()]
              + [('exercise-na-transpose','wrong_transpose',True)],
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_arrays',
        prefix='native-array-exercise-goldens-',title='Native array exercise goldens',input_counts=(1,),
        extra_options=('--native-arrays',),
        case_options={'exercise-'+name:('--construction-exercise',name) for name in EXERCISES}))
