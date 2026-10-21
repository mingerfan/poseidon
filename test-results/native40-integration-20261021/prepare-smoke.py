#!/usr/bin/env python3
"""Compile a synthetic two-chain graph through the production Qwen compiler/runner path."""
import argparse, hashlib, json, subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--devices',type=int,default=1);p.add_argument('--capacity-gib',type=float,default=7.5);p.add_argument('--reserve-gib',type=float,default=6);p.add_argument('--poly-degree',type=int,choices=[16384,65536],default=16384)
a=p.parse_args();out=Path(__file__).resolve().parent/(f'gpu{a.devices}' if a.poly_degree==16384 else f'gpu{a.devices}-n{a.poly_degree}');out.mkdir(exist_ok=True)
spec_path=root/'scripts/qwen3_dsl/profiles/operator-spec.json';spec=json.loads(spec_path.read_text());spec['context']['poly_degree']=a.poly_degree
(out/'operator-spec.json').write_text(json.dumps(spec,indent=2)+'\n');spec_path=out/'operator-spec.json'
config=json.loads((root/'scripts/qwen3_dsl/profiles/compiler-profile.json').read_text());config.update(polynomialDegree=a.poly_degree,levelUpperBound=5,bootstrapLevelUpperBound=5,bootstrapLevelLowerBound=1)
(out/'compiler-profile.json').write_text(json.dumps(config,indent=2)+'\n')
t='tensor<1x!earth.ci<0 * 0>>';pt='tensor<1x!earth.pl<0 * 0>>'
lines=[f'module {{ func.func @chains(%x: {t}, %y: {t}) -> ({t}, {t}) {{',f'%p = "earth.constant"() {{value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64}} : () -> {pt}']
for name in ('x','y'):
    prev='%'+name
    for i in range(9):
        nxt=f'%{name}{i}';lines.append(f'{nxt} = "earth.mul"({prev}, %p) : ({t}, {pt}) -> {t}');prev=nxt
lines += [f'return %x8, %y8 : {t}, {t}','}}']
(out/'input.mlir').write_text('\n'.join(lines)+'\n')
cmd=[str(root/'third_party/ckks-runtime/third_party/dacapo/build/nix/bin/hecate-opt'),'--dacapo','--boot-placement=depth-dp','--bootstrap-scale=40','--waterline=40',f'--ckks-config={out}/compiler-profile.json',
'--runtime-plan-format=binary','--runtime-plan-bundle-format=pack','--runtime-plan-boot-profile=poseidon-gpu-degree58-da2-scale40-experimental','--runtime-plan-boot-implementation=native',f'--runtime-plan-physical-level-operator-spec-path={spec_path}','--runtime-plan-levels-per-logical-level=5',
f'--runtime-plan-device-counts={a.devices}',f'--runtime-plan-device-count={a.devices}',f'--runtime-plan-operator-spec-path={spec_path}',f'--runtime-plan-communication-profile-path={root}/scripts/qwen3_dsl/profiles/communication-profile.json',
f'--runtime-plan-target-id={spec["target_id"]}',f'--runtime-plan-context-id={spec["context"]["context_id"]}',f'--runtime-plan-operator-spec-id={spec["spec_id"]}','--runtime-plan-operator-spec-version=1','--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256(spec_path.read_bytes()).hexdigest(),
'--runtime-plan-memory=true','--runtime-plan-memory-reuse=true','--runtime-plan-plaintext-schedule=waterline','--runtime-plan-prepare-backend=auto',f'--runtime-plan-gpu-budget-bytes={int(a.capacity_gib*2**30)}',f'--runtime-plan-reserve-bytes={int(a.reserve_gib*2**30)}','--runtime-plan-workspace-bytes-per-op=1073741824',
'--runtime-plan-host-budget-bytes=8589934592','--runtime-plan-pinned-budget-bytes=1073741824','--runtime-plan-raw-budget-bytes=67108864','--runtime-plan-waterline-fraction=0.5','--runtime-plan-lookahead-groups=128','--runtime-plan-ntt=true',str(out/'input.mlir'),'-o',str(out/'result.mlir')]
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'compile.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
fixture=dict(format_version=1,model={'name':'synthetic two-chain native40 engineering smoke; not Qwen inference'},context=dict(poly_degree=a.poly_degree,q_moduli_bits=[30]*50,p_moduli_bits=[30]*8,default_scale_log2=40),tolerances=dict(atol=.001,rtol=.0001),inputs=[dict(name='x',values=[.25]),dict(name='y',values=[-.125])],expected_outputs=[dict(name='out_x',values=[.25*1.01**9]),dict(name='out_y',values=[-.125*1.01**9])])
(out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
print(out)
