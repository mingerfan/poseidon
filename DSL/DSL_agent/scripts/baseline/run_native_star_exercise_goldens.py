"""Eight frozen star exercises + semantic counterexample; zero API calls."""
from hecate_python_env import ROOT
from native_star_exercises import EXERCISES
from run_schema3_goldens import main

if __name__=='__main__':
    plans=[('exercise-'+name,fixture,False) for name,(fixture,_,_) in EXERCISES.items()]
    plans.append(('exercise-ns-reverse','../native_starred_exercises/wrong_reverse',True))
    raise SystemExit(main(plans=plans,golden_dir=ROOT/'scripts/baseline/golden_cases/native_starred',
        prefix='native-star-exercise-goldens-',title='Native starred exercise goldens',input_counts=(1,),
        extra_options=('--native-starred','--compiler-configuration','seal-cpu-eva-w45-v1'),
        case_options={'exercise-'+n:('--construction-exercise',n) for n in EXERCISES}))
