#!/usr/bin/env python3
"""Run the final local binary and retain exact commands, hashes and exit codes."""
import hashlib,json,os,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
root=base.parents[1]
runner=root/'build-runtime-gpu-api-release/bin/poseidon_runtime_gpu_qwen_e2e'
case=base/'gpu1'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
for name,extra in [('normal',[]),('limit',['--native-boot-workspace-bytes','1'])]:
    report='report.json' if name=='normal' else 'limit-report.json'
    decoded='decoded.json' if name=='normal' else 'limit-decoded.json'
    log='run.log' if name=='normal' else 'limit-run.log'
    command=[str(runner),str(case/'result.chains.runtime-plan.bin'),str(case/'operator-spec.json'),str(case/'fixture.json'),str(case/report),str(case/decoded)]+extra
    metadata=dict(command=command,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']},binary_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    with (case/log).open('w') as output:
        metadata['returncode']=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
    (case/f'{name}-command.json').write_text(json.dumps(metadata,indent=2)+'\n')
    result=json.loads((case/report).read_text())
    if name=='normal':
        assert result['stage']=='complete' and result['runtime_timing']['boot_calls']==4
        assert result['host_boot_instructions']==0 and result['native_boot_instructions']==4
        assert result['comparison']['slots_checked']==16384
        # Numerical correctness remains reported independently; no tolerance changes.
        assert metadata['returncode']==(0 if result['passed'] else 1)
    else:
        assert metadata['returncode']!=0 and 'exceeds configured hard limit' in result['error']
    print(name,result['stage'],metadata['returncode'],flush=True)
