"""New explicit45 requests through real tracing/compiler/SEAL; no paid API.

Same nine frozen manual loop programs including wrong-bound negative control.
Keep the earlier failed legacy40 batch; do not rewrite its status or artifacts.
"""
from hecate_python_env import ROOT
from run_native_loop_goldens import PLANS
from run_schema3_goldens import main

if __name__ == '__main__':
    raise SystemExit(main(plans=PLANS,golden_dir=ROOT/'scripts/baseline/golden_cases/native_loops',
        prefix='native-loop-config45-goldens-',title='Native loops explicit compiler45',input_counts=(1,),
        extra_options=('--native-public-loops','--compiler-configuration','seal-cpu-eva-w45-v1')))
