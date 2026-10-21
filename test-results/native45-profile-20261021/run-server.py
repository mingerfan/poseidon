#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
runner=Path('/home/xuming/poseidon-threshold-rescale-20261021/build/bin/poseidon_runtime_gpu_qwen_e2e')
uuid='GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='12288')
rows=[]
for case in json.loads((base/'manifest.json').read_text()):
 out=base/case['name']
 cmd=[str(runner),str(out/'result.chains.runtime-plan.json'),str(out/'operator-spec.json'),str(out/'fixture.json'),str(out/'report.json'),str(out/'decoded.json'),'--bundle',str(out/'result.chains.bundle'),'--backend','gpu']
 (out/'run-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'run.log').open('w') as log:
  run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 row=dict(case,returncode=run.returncode,gpu_uuid=uuid)
 if (out/'report.json').exists():
  report=json.loads((out/'report.json').read_text())
  row.update(comparison=report.get('comparison'),error=report.get('error'),
   context_q0_level=report.get('context_q0_level'),native_profiles=report.get('native_boot_profiles'))
 rows.append(row);(base/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
