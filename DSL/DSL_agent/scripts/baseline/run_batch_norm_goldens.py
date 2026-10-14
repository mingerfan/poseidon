"""Fixed inference BN manual goldens and channel/epsilon counterexamples."""
from hecate_python_env import ROOT
from run_schema3_goldens import main
NAMES=('bn-batched','bn-spatial','bn-no-affine','bn-zero-gamma','bn-polynomial','bn-linear')
PLANS=[(n,n,False) for n in NAMES]+[('bn-batched','wrong-channel',True),('bn-batched','wrong-epsilon',True)]
if __name__=='__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/batch_norm',
        prefix='batch-norm-goldens-',title='Fixed BatchNorm',input_counts=(1,2),
        extra_options=('--object-unary',)))
