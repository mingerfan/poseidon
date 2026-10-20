"""Validate deferred encoding using the normal sequential runtime and Native Boot."""
import json
import subprocess
from pathlib import Path
from run_validation import ROOT,FRAME,PYTHON,ENV,run,sha,idle_gpu,GPU

STATUS = ROOT/'validation-jit.status'


def state(value):
    STATUS.write_text(value+'\n')
    print(value,flush=True)


def compile_clear(prefix):
    suffix = '-jit-prefix' if prefix else '-jit'
    out = ROOT/('compiled'+suffix)
    if out.exists():
        raise FileExistsError(out)
    profiles = FRAME/'profiles-native-v4'
    traced = ROOT/'traced-prefix/trace_prefix.mlirbc' if prefix else ROOT/'traced/ResNet18.mlirbc'
    state('COMPILING'+suffix.upper())
    run([PYTHON,'compile_model.py','resnet18','--defer-plaintext',
         '--operator-spec',profiles/'operator-spec.json',
         '--compiler-profile',profiles/'compiler-profile.json','--traced-mlir',traced,
         '--hecate-opt',ROOT/'compiler-build/hecate-opt','--hecate-build',FRAME/'build-make',
         '--boot-placement','depth-dp','--waterline','40','--lazy-rescale-level-factor','3',
         '--device-counts','1','--runtime-plan-memory','--runtime-plan-memory-report',
         '--inline-payload-max-bytes','0','--plan-format','json','--output-dir',out],
        'compile'+suffix+'.log')
    plan = next(out.glob('*.runtime-plan.json'))
    memory = json.loads(next(out.glob('*.memory.json')).read_text())
    device_peak = max(p['peak_bytes'] for p in memory['places'] if p['place']['kind']=='device')
    if device_peak > 20*1024**3:
        raise RuntimeError(f'object-only memory {device_peak} leaves insufficient room for Native resources')
    bundle = Path(str(plan).removesuffix('.runtime-plan.json')+'.bundle')
    fixture = ROOT/('fixture-prefix.json' if prefix else 'fixture-final.json')
    clear = ROOT/('clear-jit-prefix.json' if prefix else 'clear.json')
    state('CLEAR_RUNTIME'+suffix.upper())
    run([ROOT/'runtime-build/dacapo_mlp_vec_e2e',plan,profiles/'operator-spec.json',bundle,
         fixture,clear,ROOT/('clear'+suffix+'.diff.log')],'run-clear'+suffix+'.log')
    result = json.loads(clear.read_text())
    assert result['passed'] and result['require_python_match'] and result['python_within_tolerance']
    assert len(result['output']) == 1000
    return plan,bundle,fixture,clear


def gpu_test(prefix,artifacts):
    suffix = '-jit-prefix' if prefix else '-full'
    plan,bundle,fixture,clear = artifacts
    profiles = FRAME/'profiles-native-v4'
    binary = ROOT/'runtime-build/poseidon_runtime_gpu_mlp_e2e'
    uuid = idle_gpu()
    command = [binary,plan,profiles/'operator-spec.json',bundle,fixture,clear,
               ROOT/('gpu'+suffix+'.json'),profiles/'native-config.json']
    env = dict(ENV,CUDA_VISIBLE_DEVICES=str(GPU),POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='26624',
               POSEIDON_RUNTIME_ROTATION_KEY_POLICY='binary',
               POSEIDON_RUNTIME_TRACE=str(ROOT/('gpu'+suffix+'.trace.csv')))
    provenance = dict(command=list(map(str,command)),gpu_index=GPU,gpu_uuid=uuid,
                      binary_sha256=sha(binary),compiler_sha256=sha(ROOT/'compiler-build/hecate-opt'),
                      plan_sha256=sha(plan),fixture_sha256=sha(fixture),
                      spec_sha256=sha(profiles/'operator-spec.json'),
                      native_config_sha256=sha(profiles/'native-config.json'),
                      environment={k:v for k,v in env.items() if k.startswith(('CUDA_','POSEIDON_'))},
                      scope='deferred Encode/upload; normal SequentialRuntime; Native NoHoistBsgs')
    (ROOT/('gpu'+suffix+'.provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    state('GPU1'+suffix.upper()+'_RUNNING')
    run(['/usr/bin/time','-v','stdbuf','-oL',*command],'gpu'+suffix+'.log',env)
    result = json.loads((ROOT/('gpu'+suffix+'.json')).read_text())
    assert result['passed'] and result['native_nohoist'] and len(result['poseidon_output'])==1000
    assert result['runtime_timing']['boot_calls'] > 0
    if not prefix:
        cpu = max(range(1000),key=result['python_output'].__getitem__)
        gpu = max(range(1000),key=result['poseidon_output'].__getitem__)
        f = json.loads(fixture.read_text())
        (ROOT/'classification.json').write_text(json.dumps(dict(prediction=gpu,reference_prediction=cpu,
            label=f['label'],predictions_match=cpu==gpu,label_match=gpu==f['label'],
            scope='single image0; not dataset accuracy'),indent=2)+'\n')
        assert cpu == gpu


if __name__ == '__main__':
    if STATUS.exists():
        raise FileExistsError('JIT validation already started')
    try:
        semantic = json.loads((ROOT/'semantics-final.json').read_text())
        assert semantic['frontend_sha256']==sha(ROOT/'ResNet18.py')
        assert semantic['layout_sha256']==sha(ROOT/'imagenet_layout.py')
        assert '[PASS] RuntimePlan native GPU bootstrap' in (ROOT/'native-component.log').read_text()
        assert json.loads((ROOT/'gpu-prefix.json').read_text())['passed']
        gpu_test(True,compile_clear(True))
        gpu_test(False,compile_clear(False))
        state('FINAL_INDEPENDENT_AUDIT')
        run(['python3','audit_result.py'],'final-audit.log')
        state('FULL_MODEL_VALIDATION_PASS; pause for user review; no commit or push')
    except Exception:
        state('FAILED; inspect first failing gate; no numerical retry')
        raise
