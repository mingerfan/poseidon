"""Generate two distinct real-width blocks and full vocabulary; verify actual DSL numerically."""
import argparse,collections,json,sys,time,types
from pathlib import Path
import numpy as np
from model_entry import ROOT,configuration,model,evaluate,signature

class Expr:
    operations=collections.Counter()
    max_depth=0
    def __init__(self,values,depth=0):
        a=np.asarray(values,dtype=np.float64)
        if a.ndim!=1 or len(a)>32768:raise ValueError('Invalid slot vector')
        self.values=a if len(a)==32768 else np.pad(a,(0,32768-len(a)))
        self.depth=depth;Expr.max_depth=max(Expr.max_depth,depth)
    def __add__(self,other):
        if not isinstance(other,Expr):raise TypeError('Explicit Plain required')
        Expr.operations['add_plain' if isinstance(other,Plain) else 'add_cc']+=1
        return Expr(self.values+other.values,max(self.depth,other.depth))
    def __mul__(self,other):
        if not isinstance(other,Expr):raise TypeError('Explicit Plain required')
        Expr.operations['mul_plain' if isinstance(other,Plain) else 'mul_cc']+=1
        return Expr(self.values*other.values,max(self.depth,other.depth)+1)
    def rotate(self,steps):
        Expr.operations['rotate']+=1
        return Expr(np.roll(self.values,-steps),self.depth)
class Plain(Expr):pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();assert not a.output_dir.exists()
    a.output_dir.mkdir(parents=True)
    # The generic generator takes a config dict; the frozen 24-layer CLI is untouched.
    import importlib.util
    location=ROOT.parent/'Qwen25_24Layer_DSL/prepare_fixture.py'
    spec=importlib.util.spec_from_file_location('fixture_generator',location)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    c=configuration();fixture=a.output_dir/'fixture'
    info=generator.prepare(fixture,c)
    (a.output_dir/'fixture-info.json').write_text(json.dumps(info,indent=2))
    fake=types.ModuleType('qwen24.kernels.expr');fake.Expr,fake.Plain=Expr,Plain
    sys.modules[fake.__name__]=fake
    m=model(fixture/'weights')
    inputs=np.load(fixture/'input_slots.npy',allow_pickle=False)
    # Nonzero padding probes ABI correctness throughout both blocks.
    inputs[:,896:898]=[.3,-.2]
    start=time.monotonic();actual=evaluate(m,tuple(Expr(row) for row in inputs),'full')
    reference=np.load(fixture/'reference.npz',allow_pickle=False)
    expected=[]
    for begin in range(0,c['vocab_size'],1024):expected.append(reference['logits'][begin:begin+1024])
    for layer in range(2):
        for kind in ('keys','values'):
            for position in range(2):expected.append(reference[kind][layer,position])
    rows=[]
    for descriptor,value,target in zip(signature('full')['outputs'],actual,expected):
        assert np.isfinite(value.values).all()
        n=len(target)
        error=float(np.max(np.abs(value.values[:n]-target)))
        tail=float(np.max(np.abs(value.values[n:])))
        rows.append(dict(name=descriptor['name'],max_abs_error=error,max_abs_tail=tail,
                         conservative_multiplication_depth=value.depth))
    assert len(actual)==len(expected)==157
    report=dict(status='passed' if all(r['max_abs_error']<1e-5 and r['max_abs_tail']<1e-10 for r in rows) else 'failed',
        ckks=False,native_compiled=False,gpu=False,scope='2 distinct real-width blocks, two-token decode, final norm, full LM head, all KV',
        rows=rows,max_abs_error=max(r['max_abs_error'] for r in rows),
        conservative_multiplication_depth=Expr.max_depth,operations=dict(Expr.operations),elapsed_seconds=time.monotonic()-start)
    (a.output_dir/'numpy-validation.json').write_text(json.dumps(report,indent=2))
    (a.output_dir/'runtime-fixture.json').write_text(json.dumps(dict(inputs=inputs.tolist(),
        expected=[x.tolist() for x in expected],output_names=[r['name'] for r in rows],tolerance=1e-3,
        synthetic=True,positions=[0,1],num_layers=2,reference='independent original sqrt/SiLU/Softmax')))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if report['status']!='passed':raise RuntimeError('Packed DSL differs from original-function reference')

if __name__=='__main__':main()
