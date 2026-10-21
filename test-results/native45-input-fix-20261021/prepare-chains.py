import hashlib
import json
from pathlib import Path
import subprocess

root=Path.cwd(); out_root=root/'test-results/native45-input-fix-20261021'
out_root.mkdir(exist_ok=True)
spec=json.loads((root/'scripts/qwen3_dsl/profiles/operator-spec.json').read_text())
boot=spec['boot_profiles'][0]
assert len(spec['boot_profiles'])==1 and boot['output_level']==22 and boot['output_scale_log2']==45
ct='tensor<1x!earth.ci<0 * 0>>';pt='tensor<1x!earth.pl<0 * 0>>'
manifest=[]
for strategy,depth in [('greedy',50),('depth-dp',50)]:
 out=out_root/f'{strategy}-depth{depth}';out.mkdir(exist_ok=True)
 config=json.loads((root/'scripts/qwen3_dsl/profiles/compiler-profile.json').read_text())
 budget=boot['output_level']-spec['levels']['lower_bound']
 config.update(rescalingFactor=30,bootstrapLevelUpperBound=budget+1,bootstrapLevelLowerBound=1,levelUpperBound=budget+1,levelLowerBound=1)
 (out/'operator-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 (out/'compiler-profile.json').write_text(json.dumps(config,indent=2)+'\n')
 lines=[f'module {{ func.func @chains(%x: {ct}, %y: {ct}) -> ({ct}, {ct}) {{',
        f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64}} : () -> {pt}']
 for arg in ['x','y']:
  prev='%'+arg
  for i in range(depth):
   dst=f'%{arg}{i}';lines.append(f'{dst} = "earth.mul"({prev}, %p) : ({ct}, {pt}) -> {ct}');prev=dst
 lines += [f'return %x{depth-1}, %y{depth-1} : {ct}, {ct}','}}']
 (out/'input.mlir').write_text('\n'.join(lines)+'\n')
 fixture=dict(format_version=1,model=dict(name=f'native45 {strategy} depth{depth}; two scalar chains'),
  context=dict(poly_degree=16384,q_moduli_bits=[30]*50,p_moduli_bits=[30]*8,default_scale_log2=40),
  tolerances=dict(atol=.001,rtol=.0001),inputs=[dict(name='x',values=[.25]),dict(name='y',values=[-.125])],
  expected_outputs=[dict(name='out_x',values=[.25*1.01**depth]),dict(name='out_y',values=[-.125*1.01**depth])])
 (out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
 template=root/'test-results/threshold-rescale-20261021/depth9-gpu1'
 cmd=[a.replace(str(template),str(out)) for a in json.loads((template/'command.json').read_text())]
 cmd=[f'--boot-placement={strategy}' if a.startswith('--boot-placement=') else '--bootstrap-scale=45' if a.startswith('--bootstrap-scale=') else
      '--runtime-plan-boot-profile='+boot['profile_id'] if a.startswith('--runtime-plan-boot-profile=') else
      '--runtime-plan-format=json' if a.startswith('--runtime-plan-format=') else
      '--runtime-plan-gpu-budget-bytes=15032385536' if a.startswith('--runtime-plan-gpu-budget-bytes=') else
      '--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256((out/'operator-spec.json').read_bytes()).hexdigest() if a.startswith('--runtime-plan-operator-spec-sha256=') else a for a in cmd]
 with (out/'compile.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 plan=json.loads((out/'result.chains.runtime-plan.json').read_text()); values={v['id']:v for v in plan['values']}
 boots=[step for step in plan['execution'] if step.get('op')=='boot']
 assert all(values[b['output']]['level']==22 and values[b['output']]['scale_log2']==45 for b in boots)
 row=dict(name=out.name,boot_count=len(boots),boot_inputs=[{k:values[b['inputs'][0]][k] for k in ['level','scale_log2']} for b in boots],
          output_states=[{k:values[v][k] for k in ['level','scale_log2']} for v in plan['final_outputs']])
 manifest.append(row);print(row,flush=True)
(out_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
