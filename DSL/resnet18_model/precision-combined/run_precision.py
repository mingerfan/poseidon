"""Fail-stop precision correction: component proof, compiler, clear/prefix/full GPU, audit."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'resnet18-model-v2'
FRAME = ROOT.parent/'resnet20-framework-v1'
PROFILES = ROOT/'profiles-combined'
PYTHON = '/opt/conda/envs/apollo/bin/python'
ENV = {k:v for k,v in os.environ.items() if not k.startswith(('POSEIDON_','CUDA_'))}
ENV.update(HECATE=str(FRAME/'dacapo'),HECATE_BUILD=str(FRAME/'build-make'),
           PYTHONPATH=f'{FRAME}/dacapo/python/hecate:{ROOT.parent}/resnet18-model-dev/poly')
UUID = 'GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5'
UUIDS = [UUID,'GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6','GPU-0650e59e-7268-0a1a-8c3e-58f09c3e7fa2']


def state(value):
    (ROOT/'validation.status').write_text(value+'\n')
    print(value,flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def run(command,log,env=None):
    with (ROOT/log).open('x') as output:
        output.write(json.dumps(list(map(str,command)))+'\n')
        output.flush()
        subprocess.run(list(map(str,command)),cwd=ROOT,env=env or ENV,
                       stdout=output,stderr=subprocess.STDOUT,check=True)


def compile_clear(prefix):
    suffix = 'prefix' if prefix else 'full'
    traced = BASE/('traced-prefix/trace_prefix.mlirbc' if prefix else 'traced/ResNet18.mlirbc')
    out = ROOT/('compiled-'+suffix)
    run([PYTHON,'compile_model.py','resnet18','--operator-spec',PROFILES/'operator-spec.json',
         '--compiler-profile',PROFILES/'compiler-profile.json','--traced-mlir',traced,
         '--hecate-opt',BASE/'compiler-build/hecate-opt','--hecate-build',FRAME/'build-make',
         '--boot-placement','depth-dp','--waterline','40','--lazy-rescale-level-factor',str(json.loads((PROFILES/'native-config.json').read_text())['logical_rescale_level_factor']),
         '--device-counts','1','--runtime-plan-memory','--runtime-plan-memory-report',
         '--inline-payload-max-bytes','0','--plan-format','json','--output-dir',out,
         '--defer-plaintext'],'compile-'+suffix+'.log')
    plan, = out.glob('*.runtime-plan.json')
    bundle = Path(str(plan).removesuffix('.runtime-plan.json')+'.bundle')
    fixture = BASE/('fixture-prefix.json' if prefix else 'fixture-final.json')
    clear = ROOT/('clear-'+suffix+'.json')
    if not prefix:
        run(['python3','check_logical_graph.py'],'arithmetic-audit.log')
    run([BASE/'runtime-build-final/dacapo_mlp_vec_e2e',plan,PROFILES/'operator-spec.json',bundle,
         fixture,clear,ROOT/('clear-'+suffix+'.diff.log')],'clear-'+suffix+'.log')
    data = json.loads(clear.read_text())
    assert data['passed'] and data['python_within_tolerance'] and data['require_python_match']
    return plan,bundle,fixture,clear


def gpu(prefix,artifacts):
    suffix = 'prefix' if prefix else 'full'
    plan,bundle,fixture,clear = artifacts
    state('WAITING_IDLE_GPU_FOR_CORRECTED_'+suffix.upper())
    for _ in range(720):
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader'],text=True)
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        free = {row.split(',')[0].strip():int(row.split(',')[1]) for row in rows}
        uuid = next((candidate for candidate in UUIDS if candidate not in processes and free[candidate]>=32256),None)
        if uuid:break
        time.sleep(30)
    else:
        raise TimeoutError('no idle GPU available; no other task was interrupted')
    binary = ROOT/'runtime-build/poseidon_runtime_gpu_mlp_e2e'
    env = dict(ENV,CUDA_VISIBLE_DEVICES=uuid,POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='31744',
               POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary',
               POSEIDON_RUNTIME_TRACE=str(ROOT/('gpu-'+suffix+'.trace.csv')))
    command = [binary,plan,PROFILES/'operator-spec.json',bundle,fixture,clear,
               ROOT/('gpu-'+suffix+'.json'),PROFILES/'native-config.json']
    provenance = dict(command=list(map(str,command)),gpu_uuid=uuid,binary_sha256=sha(binary),
        plan_sha256=sha(plan),fixture_sha256=sha(fixture),clear_sha256=sha(clear),
        spec_sha256=sha(PROFILES/'operator-spec.json'),native_config_sha256=sha(PROFILES/'native-config.json'),
        environment={k:v for k,v in env.items() if k.startswith(('POSEIDON_','CUDA_'))},
        correction='EvalMod internal scale60; same model/N/Q/P/degree/ratio128; compiler plans level and scale; original gates')
    with (ROOT/('gpu-'+suffix+'.provenance.json')).open('x') as output:
        json.dump(provenance,output,indent=2)
    state('GPU_CORRECTED_'+suffix.upper()+'_RUNNING')
    run(['/usr/bin/time','-v','stdbuf','-oL',*command],'gpu-'+suffix+'.log',env)
    result = json.loads((ROOT/('gpu-'+suffix+'.json')).read_text())
    assert result['passed'] and result['native_nohoist'] and len(result['poseidon_output'])==1000


if __name__=='__main__':
    if (ROOT/'validation.status').exists():
        raise FileExistsError('precision validation already started')
    try:
        state('WAITING_COMPONENT_COMPARISON')
        for _ in range(2400):
            if (ROOT/'component.status').read_text().startswith('COMBINED_COMPONENT_PASS'):
                break
            os.kill(int((ROOT/'component.pid').read_text()),0)
            time.sleep(10)
        else:
            raise TimeoutError('component probes did not finish')
        run(['python3','prepare_combined.py'],'prepare-fix.log')
        files = [*ROOT.glob('*.py'),*PROFILES.glob('*.json'),
            ROOT/'runtime-support/poseidon_gpu_mlp_e2e_test.cpp',
            BASE/'ResNet18.py',BASE/'imagenet_layout.py',
            BASE/'runtime-support/include/runtime/runtime.hpp',BASE/'compiler-build/hecate-opt']
        with (ROOT/'source-provenance.json').open('x') as output:
            json.dump({str(p):sha(p) for p in files},output,indent=2)
        state('COMPONENT_CORRECTION_PASS; COMPILING_MODEL_AND_PREFIX')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            full = pool.submit(compile_clear,False)
            prefix = compile_clear(True)
            gpu(True,prefix)
            state('CORRECTED_PREFIX_PASS; WAITING_FULL_CLEAR_AND_ARITHMETIC_AUDIT')
            artifacts = full.result()
        gpu(False,artifacts)
        state('FINAL_INDEPENDENT_AUDIT')
        run(['python3','audit_combined.py'],'final-audit.log')
        state('PRECISION_FIX_FULL_MODEL_PASS; no commit or push')
    except Exception:
        state('FAILED; first failing gate preserved; no automatic retry')
        raise
