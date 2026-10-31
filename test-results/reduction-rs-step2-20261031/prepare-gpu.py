from pathlib import Path
import hashlib, json, runpy, subprocess
root=Path('/home/xuming/poseidon-reduction-rs-step2-20261031')
runtime=root/'runtime';compiler=runtime/'third_party/dacapo';out=root/'gpu';out.mkdir(exist_ok=True)
generate=runpy.run_path(str(runtime/'integrations/dacapo/generate_resnet20_manual_profiles.py'))['generate']
_,spec=generate(out)
source=runpy.run_path(str(compiler/'test/runtime-plan/check-native-planning.py'))['source']
lines=source(18,residual=True).splitlines()
ct,pt='tensor<1x!earth.ci<0 * 0>>','tensor<1x!earth.pl<0 * 0>>'
seed=[]
for i in range(8):
    coefficient=(i+1)/64
    seed += [f'%p{i} = "earth.constant"() {{value = dense<{coefficient}> : tensor<32768xf64>, rms_var = 1.0 : f64}} : () -> {pt}',
             f'%m{i} = "earth.mul"(%x, %p{i}) : ({ct}, {pt}) -> {ct}']
previous='%m0'
for i in range(1,8):
    name='%seed' if i==7 else f'%sum{i}'
    seed.append(f'{name} = "earth.add"({previous}, %m{i}) : ({ct}, {ct}) -> {ct}');previous=name
(out/'reduction.mlir').write_text('\n'.join(lines[:1]+seed+[line.replace('%x','%seed') for line in lines[1:]]))
rows=[]
for enabled in (False,True):
    name='enabled' if enabled else 'disabled';dest=out/(name+'.mlir')
    command=[str(root/'bin/hecate-opt'),str(out/'reduction.mlir'),'--dacapo',
        f'--ckks-config={out}/compiler.json','--waterline=40','--bootstrap-scale=40',
        '--boot-placement=depth-dp','--polynomial-optimizations=none',
        '--runtime-plan-format=json',f'--runtime-plan-coalesce-rescales={str(enabled).lower()}',
        f'--runtime-plan-physical-level-operator-spec-path={out}/operator-spec.json',
        '--runtime-plan-levels-per-logical-level=1',
        '--runtime-plan-boot-profile=poseidon-resnet20-manual-q50-scale40',
        '--runtime-plan-boot-implementation=native',
        f'--runtime-plan-operator-spec-path={out}/operator-spec.json',
        '--runtime-plan-device-counts=1','--runtime-plan-device-count=1',
        '--runtime-plan-target-id=poseidon-ckks-gpu',
        '--runtime-plan-context-id=poseidon-resnet20-manual-q50',
        '--runtime-plan-operator-spec-id=poseidon-resnet20-manual-q50',
        '--runtime-plan-operator-spec-version=1','--runtime-plan-inline-payload-max-bytes=1048576',
        '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256((out/'operator-spec.json').read_bytes()).hexdigest(),
        '-o',str(dest)]
    with (out/(name+'.compile.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,(out/(name+'.compile.log')).read_text()[-5000:]
    plan=json.loads((out/(name+'.squares.runtime-plan.json')).read_text());values={v['id']:v for v in plan['values']}
    boots=[s for s in plan['execution'] if s.get('op')=='boot']
    row=dict(enabled=enabled,rs=sum(s.get('op')=='rescale' for s in plan['execution']),boots=len(boots),
             final=[(values[v]['level'],values[v]['scale_log2']) for v in plan['final_outputs']],
             boot_inputs=[(values[s['inputs'][0]]['level'],values[s['inputs'][0]]['scale_log2']) for s in boots])
    assert boots and all(level==5 and abs(scale-40)<1e-8 for level,scale in row['boot_inputs']),row
    rows.append(row)
assert rows[0]['rs']-rows[1]['rs']==7,rows
assert rows[0]['final']==rows[1]['final'] and rows[0]['boots']==rows[1]['boots'],rows
(out/'plan-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
