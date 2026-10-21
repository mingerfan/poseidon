#!/usr/bin/env python3
"""Compile pre-change policies with the preserved server toolchain (no GPU use)."""
import hashlib,json,subprocess
from pathlib import Path
base=Path('/home/xuming/poseidon-threshold-rescale-20261021')
old=Path('/home/xuming/poseidon-native40-integration-20261021')
for depth,host in [(9,False),(40,False),(40,True)]:
 name=f'baseline-depth{depth}{"-host" if host else ""}'
 out=base/name;out.mkdir(exist_ok=True)
 source=base/f'depth{depth}{"-host" if host else ""}-gpu1'
 cfg=json.loads((old/'gpu2/compiler-profile.json').read_text());cfg.update(rescalingFactor=120,levelUpperBound=10 if host else 5,bootstrapLevelUpperBound=10 if host else 5,bootstrapLevelLowerBound=1)
 (out/'compiler-profile.json').write_text(json.dumps(cfg,indent=2)+'\n')
 spec=json.loads((old/'gpu2/operator-spec.json').read_text());spec['rescale_mode']='lazy';spec['context'].pop('rns_moduli',None)
 spec_text=json.dumps(spec,indent=2)+'\n';(out/'operator-spec.json').write_text(spec_text)
 (out/'input.mlir').write_bytes((source/'input.mlir').read_bytes())
 command=json.loads((source/'command.json').read_text()) if (source/'command.json').exists() else json.loads((base/'depth9-gpu1/command.json').read_text())
 tool=old/'toolchain';args=[]
 for arg in command[1:]:
  arg=arg.replace(str(source),str(out))
  if '/mnt/zhitai/' in arg:
   if arg.startswith('--ckks-config='):arg=f'--ckks-config={out}/compiler-profile.json'
   elif 'physical-level-operator-spec-path=' in arg:arg=f'--runtime-plan-physical-level-operator-spec-path={out}/operator-spec.json'
   elif arg.startswith('--runtime-plan-operator-spec-path='):arg=f'--runtime-plan-operator-spec-path={out}/operator-spec.json'
   elif 'communication-profile-path=' in arg:arg=f'--runtime-plan-communication-profile-path={old}/profiles/communication-profile.json'
   elif arg.endswith('/input.mlir'):arg=str(out/'input.mlir')
   elif arg.endswith('/result.mlir'):arg=str(out/'result.mlir')
  if arg.startswith('--runtime-plan-levels-per-logical-level='):arg='--runtime-plan-levels-per-logical-level=5'
  if arg.startswith('--runtime-plan-format='):arg='--runtime-plan-format=json'
  if arg.startswith('--runtime-plan-operator-spec-sha256='):arg='--runtime-plan-operator-spec-sha256=sha256:'+hashlib.sha256(spec_text.encode()).hexdigest()
  args.append(arg)
 cmd=[str(tool/'lib/ld-linux-x86-64.so.2'),'--library-path',f'{tool}/lib:/opt/conda/lib:/lib/x86_64-linux-gnu',str(tool/'bin/hecate-opt'),*args]
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'compile.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(out)
