"""One capacity-corrected full GPU run; preserve the 26 GiB pool failure."""
import json
import subprocess
from pathlib import Path
from run_validation import ROOT,FRAME,ENV,run,sha

PREFIX = 'gpu-full-pool31'
STATUS = ROOT/'validation-pool31.status'


def state(value):
    STATUS.write_text(value+'\n')
    print(value,flush=True)


if __name__=='__main__':
    if STATUS.exists():
        raise FileExistsError(STATUS)
    try:
        row = subprocess.check_output(['nvidia-smi','-i','1',
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader'],text=True)
        assert row[0].strip()=='GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5'
        assert row[0].strip() not in processes and int(row[1])>=32256,'GPU1 unavailable'
        old = json.loads((ROOT/'gpu-full.provenance.json').read_text())
        plan = next((ROOT/'compiled-jit').glob('*.runtime-plan.json'))
        fixture = ROOT/'fixture-final.json'
        binary = ROOT/'runtime-build-final/poseidon_runtime_gpu_mlp_e2e'
        profiles = FRAME/'profiles-native-v4'
        for key,path in (('binary_sha256',binary),('plan_sha256',plan),('fixture_sha256',fixture),
                         ('spec_sha256',profiles/'operator-spec.json'),
                         ('native_config_sha256',profiles/'native-config.json')):
            assert sha(path)==old[key],key
        assert not (ROOT/(PREFIX+'.json')).exists()
        env = dict(ENV,CUDA_VISIBLE_DEVICES='1',POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='31744',
                   POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary',
                   POSEIDON_RUNTIME_TRACE=str(ROOT/(PREFIX+'.trace.csv')))
        command = [binary,plan,profiles/'operator-spec.json',
                   Path(str(plan).removesuffix('.runtime-plan.json')+'.bundle'),fixture,
                   ROOT/'clear.json',ROOT/(PREFIX+'.json'),profiles/'native-config.json']
        provenance = dict(old,command=list(map(str,command)),
                          environment={k:v for k,v in env.items() if k.startswith(('POSEIDON_','CUDA_'))},
                          change='Pool cap only: 26 GiB to 31 GiB; same binary/plan/fixtures/profiles/gates',
                          previous_failure='gpu-full.log: pool cap exceeded with 5836767232 driver bytes free')
        with (ROOT/(PREFIX+'.provenance.json')).open('x') as stream:
            json.dump(provenance,stream,indent=2)
        state('GPU1_FULL_POOL31_RUNNING')
        run(['/usr/bin/time','-v','stdbuf','-oL',*command],PREFIX+'.log',env)
        state('INDEPENDENT_FINAL_AUDIT')
        run(['python3','audit_result.py','--gpu-prefix',PREFIX,'--report','final-audit-pool31.json'],
            'final-audit-pool31.log')
        state('FULL_MODEL_VALIDATION_PASS; no commit or push')
    except Exception:
        state('FAILED; first failing gate preserved; no automatic retry')
        raise
