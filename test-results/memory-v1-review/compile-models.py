import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();compiler=root/'third_party/ckks-runtime/third_party/dacapo';binary=compiler/'build/nix/bin/hecate-opt'
spec=root/'test-results/e2e-new-profile-e63ec18/profiles/operator-spec.json';d=json.loads(spec.read_text());digest='sha256:'+hashlib.sha256(spec.read_bytes()).hexdigest()
for topology in ['1gpu','4gpu','2x2']:
 out=root/'test-results/memory-v1-review'/topology
 for variant in ['baseline','release','reuse']:
  prefix=out/f'mlp.{variant}'
  passes=[f'estimate-runtime-memory{{prefix={prefix} operator-spec={spec} report-tag=before}}']
  if variant!='baseline':passes.append('plan-runtime-memory'+('{enable-reuse=false}' if variant=='release' else ''))
  passes += [f'estimate-runtime-memory{{prefix={prefix} operator-spec={spec}}}',f'emit-runtime-plan{{prefix={prefix} plan-id=990 target-id={d["target_id"]} operator-spec-id={d["spec_id"]} operator-spec-version={d["version"]} operator-spec-sha256={digest} context-id={d["context"]["context_id"]} boot-profile={d["boot_profiles"][0]["profile_id"]} boot-implementation=decrypt_reencrypt}}']
  with (out/f'{variant}.compile.log').open('w') as log:
   subprocess.run([str(binary),str(out/'mlp.input.mlir'),'-p=builtin.module(func.func('+','.join(passes)+'))','-o',str(prefix)+'.mlir'],cwd=compiler,stdout=log,stderr=log,check=True)
  report=json.loads(Path(str(prefix)+'._hecate_MLP.memory.json').read_text());plan=json.loads(Path(str(prefix)+'._hecate_MLP.runtime-plan.json').read_text())
  print(topology,variant,'peaks MiB',[r['peak_bytes']/2**20 for r in report['places']],'reuse',sum('reuse_input' in op for op in plan['execution']),'release',sum(op['kind']=='release' for phase in ['initialization','execution'] for op in plan[phase]),flush=True)
