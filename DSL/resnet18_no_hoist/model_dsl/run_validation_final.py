"""Resume from already compiled plans after bounded communication cleanup."""
import concurrent.futures
import json
import time
from pathlib import Path
from run_validation import ROOT,FRAME,ENV,run,sha,idle_gpu,GPU

STATUS = ROOT/'validation-final.status'
BIN = ROOT/'runtime-build-final'
PROFILES = FRAME/'profiles-native-v4'


def state(value):
    STATUS.write_text(value+'\n')
    print(value,flush=True)


def artifacts(prefix):
    plan = next((ROOT/('compiled-jit-prefix' if prefix else 'compiled-jit')).glob('*.runtime-plan.json'))
    return plan,Path(str(plan).removesuffix('.runtime-plan.json')+'.bundle'),ROOT/('fixture-prefix.json' if prefix else 'fixture-final.json')


def clear_test(prefix):
    plan,bundle,fixture = artifacts(prefix)
    suffix = '-final-prefix' if prefix else '-final'
    clear = ROOT/('clear-final-prefix.json' if prefix else 'clear.json')
    if clear.exists():
        raise FileExistsError(clear)
    run([BIN/'dacapo_mlp_vec_e2e',plan,PROFILES/'operator-spec.json',bundle,fixture,clear,
         ROOT/('clear'+suffix+'.diff.log')],'run-clear'+suffix+'.log')
    result = json.loads(clear.read_text())
    assert result['passed'] and result['require_python_match'] and result['python_within_tolerance']
    return clear


def gpu_test(prefix):
    plan,bundle,fixture = artifacts(prefix)
    clear = ROOT/('clear-final-prefix.json' if prefix else 'clear.json')
    suffix = '-final-prefix' if prefix else '-full'
    uuid = idle_gpu()
    env = dict(ENV,CUDA_VISIBLE_DEVICES=str(GPU),POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='26624',
               POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary',
               POSEIDON_RUNTIME_TRACE=str(ROOT/('gpu'+suffix+'.trace.csv')))
    command = [BIN/'poseidon_runtime_gpu_mlp_e2e',plan,PROFILES/'operator-spec.json',bundle,
               fixture,clear,ROOT/('gpu'+suffix+'.json'),PROFILES/'native-config.json']
    provenance = dict(command=list(map(str,command)),gpu_index=GPU,gpu_uuid=uuid,
                      binary_sha256=sha(command[0]),compiler_sha256=sha(ROOT/'compiler-build/hecate-opt'),
                      plan_sha256=sha(plan),fixture_sha256=sha(fixture),
                      spec_sha256=sha(PROFILES/'operator-spec.json'),
                      native_config_sha256=sha(PROFILES/'native-config.json'),
                      runtime_header_sha256=sha(ROOT/'runtime-support/include/runtime/runtime.hpp'),
                      environment={k:v for k,v in env.items() if k.startswith(('POSEIDON_','CUDA_'))},
                      scope='compiler plan; normal SequentialRuntime with bounded single-rank communication cleanup; Native no-hoist')
    (ROOT/('gpu'+suffix+'.provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    if not prefix:
        state('GPU1_FULL_RUNNING')
    run(['/usr/bin/time','-v','stdbuf','-oL',*command],'gpu'+suffix+'.log',env)
    result = json.loads((ROOT/('gpu'+suffix+'.json')).read_text())
    assert result['passed'] and result['native_nohoist'] and len(result['poseidon_output'])==1000
    if not prefix:
        reference = max(range(1000),key=result['python_output'].__getitem__)
        prediction = max(range(1000),key=result['poseidon_output'].__getitem__)
        label = json.loads(fixture.read_text())['label']
        (ROOT/'classification.json').write_text(json.dumps(dict(prediction=prediction,
            reference_prediction=reference,label=label,predictions_match=prediction==reference,
            label_match=prediction==label,scope='single image0; not dataset accuracy'),indent=2)+'\n')
        assert prediction==reference


if __name__ == '__main__':
    if STATUS.exists():
        raise FileExistsError(STATUS)
    try:
        state('WAITING_FINAL_RUNTIME_BUILD')
        for _ in range(120):
            if (BIN/'provenance.json').exists():
                break
            time.sleep(10)
        else:
            raise RuntimeError('final runtime build timeout')
        state('TRANSFER_RESOURCE_STRESS_GATE')
        run(['python3','check_runtime_transfers.py'],'transfer-stress-launcher.log')
        clear_test(True)
        state('CPU_FULL_AND_GPU_PREFIX_RUNNING')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            prefix = pool.submit(gpu_test,True)
            clear_test(False)
            prefix.result()
        gpu_test(False)
        state('FINAL_INDEPENDENT_AUDIT')
        run(['python3','audit_result.py'],'final-audit.log')
        state('FULL_MODEL_VALIDATION_PASS; pause for user review; no commit or push')
    except Exception:
        state('FAILED; inspect first failing gate; no numerical retry')
        raise
