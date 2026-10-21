#!/usr/bin/env python3
"""Compile unequal-scale and equal-nominal-scale residuals for numeric validation."""
import json, subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
source=base/'depth9-gpu1'
t='tensor<1x!earth.ci<0 * 0>>'; pt='tensor<1x!earth.pl<0 * 0>>'
for name in ['unequal-scale-add','equal-nominal-add']:
 out=base/name;out.mkdir(exist_ok=True)
 for file in ['operator-spec.json','compiler-profile.json']:(out/file).write_bytes((source/file).read_bytes())
 lines=[f'module {{ func.func @chains(%x: {t}, %y: {t}) -> ({t}, {t}) {{',f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64}} : () -> {pt}']
 def mul(dst,a,b='%p',bt=pt):lines.append(f'%{dst} = "earth.mul"({a}, {b}) : ({t}, {bt}) -> {t}')
 def add(dst,a,b):lines.append(f'%{dst} = "earth.add"({a}, {b}) : ({t}, {t}) -> {t}')
 mul('a','%x')
 if name=='unequal-scale-add':
  mul('b','%y');mul('c','%b');add('d','%a','%c');mul('e','%d');expected=[.25*1.01-.125*1.01**2,(.25*1.01-.125*1.01**2)*1.01]
 else:
  mul('b','%a');mul('c','%b');add('d','%c','%x');mul('e','%d');expected=[.25*1.01**3+.25,(.25*1.01**3+.25)*1.01]
 lines += [f'return %d, %e : {t}, {t}','}}'];(out/'input.mlir').write_text('\n'.join(lines)+'\n')
 cmd=[arg.replace(str(source),str(out)) for arg in json.loads((source/'command.json').read_text())]
 with (out/'compile.log').open('w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 fixture=json.loads((source/'fixture.json').read_text());fixture['model']['name']=name
 for row,value in zip(fixture['expected_outputs'],expected):row['values']=[value]
 (out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n');(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 print(out)
