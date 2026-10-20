"""Fail-stop server validation through the existing compiler and runtime.

No numerical retries, no publishing, no changes to the native model branches.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAME = ROOT.parent/'resnet20-framework-v1'
PYTHON = '/opt/conda/envs/apollo/bin/python'
GPU = 1
STATUS = ROOT/'validation.status'
ENV = {k:v for k,v in os.environ.items() if not k.startswith('POSEIDON_')}
ENV.update(HECATE=str(FRAME/'dacapo'),HECATE_BUILD=str(FRAME/'build-make'),
           PYTHONPATH=f'{FRAME}/dacapo/python/hecate:{ROOT.parent}/resnet18-model-dev/poly')


def state(value):
    STATUS.write_text(value+'\n')
    print(value,flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, log, env=None):
    with (ROOT/log).open('x') as output:
        output.write(json.dumps(dict(command=list(map(str,command))))+'\n')
        output.flush()
        subprocess.run(list(map(str,command)),cwd=ROOT,env=env or ENV,stdout=output,
                       stderr=subprocess.STDOUT,check=True)


def idle_gpu():
    for _ in range(480):
        row = subprocess.check_output(['nvidia-smi','-i',str(GPU),
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'],text=True)
        if int(row[1]) >= 28672 and row[0].strip() not in processes:
            return row[0].strip()
        time.sleep(30)
    raise RuntimeError('GPU remained occupied; no tasks were interrupted')


def compile_and_clear(prefix):
    suffix = '-prefix' if prefix else ''
    state('DACAPO'+suffix.upper()+'_PLANNING')
    out = ROOT/('compiled'+suffix)
    if out.exists():
        raise FileExistsError(out)
    profiles = FRAME/'profiles-native-v4'
    traced = ROOT/'traced-prefix/trace_prefix.mlirbc' if prefix else ROOT/'traced/ResNet18.mlirbc'
    run([PYTHON,'compile_model.py','resnet18','--operator-spec',profiles/'operator-spec.json',
         '--compiler-profile',profiles/'compiler-profile.json','--traced-mlir',traced,
         '--hecate-opt',FRAME/'build-make/bin/hecate-opt','--hecate-build',FRAME/'build-make',
         '--boot-placement','depth-dp','--waterline','40','--lazy-rescale-level-factor','3',
         '--device-counts','1','--runtime-plan-memory','--runtime-plan-memory-report',
         '--inline-payload-max-bytes','0','--plan-format','json','--output-dir',out],
        'compile'+suffix+'.log')
    plans = list(out.glob('*.runtime-plan.json'))
    assert len(plans) == 1
    plan = plans[0]
    bundle = Path(str(plan).removesuffix('.runtime-plan.json')+'.bundle')
    fixture = ROOT/('fixture-prefix.json' if prefix else 'fixture-final.json')
    clear = ROOT/('clear'+suffix+'.json')
    state('COMPILED'+suffix.upper()+'_CLEAR_RUNTIME')
    run([ROOT/'runtime-build/dacapo_mlp_vec_e2e',plan,profiles/'operator-spec.json',bundle,
         fixture,clear,ROOT/('clear'+suffix+'.diff.log')],'run-clear'+suffix+'.log')
    result = json.loads(clear.read_text())
    assert result['passed'] and result['require_python_match'] and result['python_within_tolerance']
    assert len(result['output']) == 1000
    return plan,bundle,fixture,clear


def gpu_test(prefix, artifacts):
    suffix = '-prefix' if prefix else '-full'
    plan,bundle,fixture,clear = artifacts
    profiles = FRAME/'profiles-native-v4'
    binary = ROOT/'runtime-build/poseidon_runtime_gpu_mlp_e2e'
    state('WAITING_GPU'+suffix.upper())
    uuid = idle_gpu()
    env = dict(ENV,CUDA_VISIBLE_DEVICES=str(GPU),POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='26624',
               POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary',
               POSEIDON_RUNTIME_TRACE=str(ROOT/('gpu'+suffix+'.trace.csv')))
    command = [binary,plan,profiles/'operator-spec.json',bundle,fixture,clear,
               ROOT/('gpu'+suffix+'.json'),profiles/'native-config.json']
    provenance = dict(command=list(map(str,command)),gpu_index=GPU,gpu_uuid=uuid,
                      binary_sha256=sha(binary),plan_sha256=sha(plan),fixture_sha256=sha(fixture),
                      spec_sha256=sha(profiles/'operator-spec.json'),
                      native_config_sha256=sha(profiles/'native-config.json'),
                      environment={k:v for k,v in env.items() if k.startswith(('CUDA_','POSEIDON_'))},
                      scope='compiled model; normal SequentialRuntime; Native NoHoistBsgs')
    (ROOT/('gpu'+suffix+'.provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    state('GPU1'+suffix.upper()+'_RUNNING')
    run(['/usr/bin/time','-v','stdbuf','-oL',*command],'gpu'+suffix+'.log',env)
    result = json.loads((ROOT/('gpu'+suffix+'.json')).read_text())
    assert result['passed'] and result['native_nohoist'] and len(result['poseidon_output']) == 1000
    assert result['runtime_timing']['boot_calls'] > 0
    if not prefix:
        predicted = max(range(1000),key=result['poseidon_output'].__getitem__)
        reference = max(range(1000),key=result['python_output'].__getitem__)
        f = json.loads(fixture.read_text())
        classification = dict(prediction=predicted,reference_prediction=reference,label=f['label'],
                              predictions_match=predicted==reference,label_match=predicted==f['label'],
                              scope='single image0; not dataset accuracy')
        (ROOT/'classification.json').write_text(json.dumps(classification,indent=2)+'\n')
        assert predicted == reference


if __name__ == '__main__':
    if STATUS.exists():
        raise FileExistsError('validation already started; inspect existing state')
    try:
        state('TRACING_STEM_PREFIX')
        run([PYTHON,'-u','trace_prefix.py'],'trace-prefix.log')
        prefix = compile_and_clear(True)
        state('NATIVE_COMPONENT_GATE')
        uuid = idle_gpu()
        env = dict(ENV,CUDA_VISIBLE_DEVICES=str(GPU),POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='26624',
                   POSEIDON_RUNTIME_BOOTSTRAP_DEGREE='65536',
                   POSEIDON_RUNTIME_BOOTSTRAP_LINEAR_MODE='no_hoist',
                   POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE='1',
                   POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE='1',
                   POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE='1',
                   POSEIDON_RUNTIME_BOOTSTRAP_DIAGNOSTICS='1')
        run([FRAME/'build-runtime-v2/poseidon/bin/poseidon_runtime_gpu_bootstrap_e2e'],
            'native-component.log',env)
        gpu_test(True,prefix)
        state('WAITING_FULL_EARTH_EXPORT')
        for _ in range(720):
            status = (ROOT/'frontend.status').read_text()
            if status.startswith('FAILED'):
                raise RuntimeError('frontend export failed')
            if status.startswith('SEMANTICS_AND_EARTH_TRACE_PASS'):
                break
            time.sleep(10)
        else:
            raise RuntimeError('frontend export timeout')
        full = compile_and_clear(False)
        gpu_test(False,full)
        state('FULL_MODEL_VALIDATION_PASS; pause for user review; no push')
    except Exception:
        state('FAILED; first failing gate retained; no numerical retry or push')
        raise
