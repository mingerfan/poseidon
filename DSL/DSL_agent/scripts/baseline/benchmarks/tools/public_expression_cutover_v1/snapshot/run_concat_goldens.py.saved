"""Static concat: manual real FHE positives and discriminating counterexamples."""
from hecate_python_env import ROOT
from run_schema3_goldens import main
NAMES=('concat-vector','concat-packed','concat-axis0','concat-axisneg','concat-three','concat-spatial')
PLANS=[(n,n,False) for n in NAMES]+[('concat-vector','wrong-order',True),('concat-axisneg','wrong-axis',True)]
if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/concat',
        prefix='concat-goldens-',title='Concat',input_counts=(1,2),extra_options=('--object-unary',)))
