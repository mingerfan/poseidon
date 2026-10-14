"""Manual eleven-case targeted native-call cohort, no model API."""
from hecate_python_env import ROOT
from native_function_exercises import EXERCISES
from run_schema3_goldens import main

if __name__ == '__main__':
    raise SystemExit(main(
        plans=[('exercise-'+name,name.removeprefix('nf-').replace('-','_'),False) for name in EXERCISES],
        golden_dir=ROOT/'scripts/baseline/golden_cases/native_exercises',
        prefix='native-exercise-goldens-',title='Native function exercise goldens',input_counts=(1,2),
        extra_options=('--native-functions',),
        case_options={'exercise-'+name: ('--construction-exercise',name) for name in EXERCISES}))
