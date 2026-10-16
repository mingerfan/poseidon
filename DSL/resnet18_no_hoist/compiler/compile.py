"""Compile the frozen CKKS MLIR through real Hecate and compare every artifact byte."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_assets import prepare,sha
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('hecate_opt',type=Path)
    p.add_argument('--work',type=Path,default=ROOT/'work');a=p.parse_args()
    assets=prepare(a.work.resolve());output=a.work.resolve()/'compiler-recheck';output.mkdir(exist_ok=False)
    prefix=output/'resnet18'
    command=[str(a.hecate_opt.resolve()),str(assets/'resnet18.mlir'),
        '--ckks-config='+str(ROOT/'compiler/profiled_HEAAN_GPU_blb5.json'),
        '--pass-pipeline=builtin.module(func.func(emit-hevm{prefix='+str(prefix)+'}))',
        '-o',str(output/'compiled.mlir')]
    with (output/'compile.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    for suffix in ('','.native_scales','.native_primitives'):
        actual=Path(str(prefix)+'.native_resnet18.hevm'+suffix)
        expected=ROOT/('artifacts/resnet18.hevm'+suffix)
        assert actual.read_bytes()==expected.read_bytes(),actual
    print('REAL_COMPILER_RECHECK PASS',sha(ROOT/'artifacts/resnet18.hevm'))
