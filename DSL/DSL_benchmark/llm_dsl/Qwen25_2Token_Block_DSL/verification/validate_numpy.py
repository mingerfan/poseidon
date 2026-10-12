"""Exercise both actual tracing entrypoints with NumPy slots; not CKKS/GPU."""
import argparse
import collections
import json
from pathlib import Path
import runpy
import sys
import time
import types
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class Expr:
    operations=collections.Counter()
    def __init__(self,values):
        a=np.asarray(values,dtype=np.float64)
        if a.ndim!=1 or len(a)>32768:
            raise ValueError('Expected slot vector fitting 32768 slots')
        self.values=a if len(a)==32768 else np.pad(a,(0,32768-len(a)))
    def __add__(self,other):
        if not isinstance(other,Expr):raise TypeError('Explicit Plain required')
        Expr.operations['add_plain' if isinstance(other,Plain) else 'add']+=1
        return Expr(self.values+other.values)
    def __mul__(self,other):
        if not isinstance(other,Expr):raise TypeError('Explicit Plain required')
        Expr.operations['multiply_plain' if isinstance(other,Plain) else 'multiply']+=1
        return Expr(self.values*other.values)
    def rotate(self,steps):
        Expr.operations['rotate']+=1
        return Expr(np.roll(self.values,-steps))

class Plain(Expr):
    pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Use a new verification output directory')
    args.output_dir.mkdir(parents=True)
    inputs=np.load(args.fixture_dir/'input_slots.npy',allow_pickle=False)
    expected=np.load(ROOT/'data/reference_hidden.npy',allow_pickle=False).reshape(-1)
    fake_expr=types.ModuleType('qwen2token.kernels.expr')
    fake_expr.Expr,fake_expr.Plain=Expr,Plain
    sys.modules[fake_expr.__name__]=fake_expr
    outputs={}; checks=[]
    for mode in ('prefill','decode-pair'):
        print('Checking actual entrypoint:',mode,flush=True)
        start=time.monotonic(); Expr.operations.clear()
        fake=types.ModuleType('hecate'); functions=[]; captured=[]
        def func(signature):
            assert signature=='c,c'
            def decorate(fn):functions.append(fn);return fn
            return decorate
        def save(*paths):
            assert len(functions)==1
            result=functions[0](*(Expr(x) for x in inputs))
            assert isinstance(result,list) and len(result)==1
            captured.append(result[0].values)
            return 'NumPy verification only; native save not executed'
        fake.func,fake.save=func,save
        command=[str(ROOT/'trace_qwen_block.py'),'--mode',mode,'--weights',str(args.fixture_dir/'weights.npz'),
                 '--output-dir',str(args.output_dir/mode)]
        with patch.dict(sys.modules,{'hecate':fake}),patch.object(sys,'argv',command):
            runpy.run_path(command[0],run_name='__main__')
        value=captured[0]
        if not np.isfinite(value).all():raise AssertionError('Nonfinite block output')
        error=float(np.max(np.abs(value[:1792]-expected)))
        tail=float(np.max(np.abs(value[1792:])))
        assert error<1e-5 and tail<1e-10,(mode,error,tail)
        with (args.output_dir/(mode+'.npy')).open('xb') as stream:np.save(stream,value)
        outputs[mode]=value
        row=dict(mode=mode,original_function_max_abs_error=error,max_abs_tail=tail,
                 elapsed_seconds=time.monotonic()-start,operations=dict(Expr.operations))
        checks.append(row); print(json.dumps(row),flush=True)
    difference=float(np.max(np.abs(outputs['prefill']-outputs['decode-pair'])))
    assert difference<1e-12,difference
    report=dict(status='passed',scope='full real-dimension DSL arithmetic and entrypoint wiring with NumPy',
                native_hecate=False,ckks=False,gpu=False,slots=32768,positions=[7,8],
                checks=checks,prefill_decode_max_abs_difference=difference)
    with (args.output_dir/'validation.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
