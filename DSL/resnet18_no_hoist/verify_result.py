"""Independently audit the complete shipped ResNet18 model-level DSL package."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT/name).read_text())

if __name__ == '__main__':
    manifest = read('manifest.sha256.json')
    for name,digest in manifest.items():
        assert sha(ROOT/name) == digest,name
    for name,digest in read('patches/validated-overrides.json')['sources'].items():
        assert sha(ROOT/name) == digest,name
    packaging = read('evidence/packaging-validation.json')
    assert packaging['status']=='PASS' and packaging['prefix_plan_byte_identical']
    assert packaging['profiles_regenerated_identically'] and packaging['new_full_clear_semantics_matches_original_fixture']
    assert read('evidence/portable-prefix-clear.json')['passed']
    for record in read('evidence/portable-entry-build.json'):
        name = {'dacapo_mlp_vec_e2e':'dacapo_mlp_vec_e2e.cpp',
                'poseidon_runtime_gpu_mlp_e2e':'poseidon_gpu_mlp_e2e_test.cpp',
                'poseidon_runtime_gpu_bootstrap_e2e':'component-probe.cpp'}[record['target']]
        assert sha(ROOT/'build-support'/name)==record['source_sha256']
    assert sha(ROOT/'build-support/optimizer.cpp')==read('evidence/portable-optimizer-build.json')['source_sha256']
    semantic = read('evidence/frontend-semantics.json')
    fixture = read('fixtures/image0.json')
    gpu,clear = read('evidence/gpu-full.json'),read('evidence/clear-full.json')
    final = read('evidence/final-audit.json')
    config,spec = read('profiles/native-config.json'),read('profiles/operator-spec.json')
    assert sha(ROOT/'ResNet18.py') == semantic['frontend_sha256']
    assert sha(ROOT/'imagenet_layout.py') == semantic['layout_sha256']
    assert sha(ROOT/'evidence/frontend-semantics.json') == fixture['semantic_report_sha256']
    assert gpu['model_sha256'] == clear['model_sha256'] == fixture['model_sha256'] == semantic['model_sha256']
    assert semantic['status'] == 'PASS' and semantic['weights'] == 102
    assert clear['passed'] and clear['python_within_tolerance'] and clear['require_python_match']
    assert clear['output'] == fixture['python_output']
    assert len(fixture['packed_inputs']) == 8 and all(len(x)==32768 for x in fixture['packed_inputs'])
    provenance = read('evidence/gpu-full.provenance.json')
    assert provenance['fixture_sha256'] == sha(ROOT/'fixtures/image0.json')
    assert provenance['clear_sha256'] == sha(ROOT/'evidence/clear-full.json')
    assert provenance['spec_sha256'] == sha(ROOT/'profiles/operator-spec.json')
    assert provenance['native_config_sha256'] == sha(ROOT/'profiles/native-config.json')
    digest = hashlib.sha256()
    with gzip.open(ROOT/'evidence/runtime-plan.json.gz','rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            digest.update(block)
    assert 'sha256:'+digest.hexdigest() == gpu['plan_sha256'] == clear['plan_sha256']
    assert digest.hexdigest() == provenance['plan_sha256'] == final['plan_sha256']
    with gzip.open(ROOT/'evidence/runtime-plan.json.gz','rt') as f:
        plan = json.load(f)
    assert len(plan['external_inputs']) == 8 and len(plan['final_outputs']) == 1
    computes = [i for phase in ('initialization','execution','finalization') for i in plan[phase] if i['kind']=='compute']
    boots = [i for i in computes if i['op']=='boot']
    assert len(computes) == gpu['runtime_timing']['compute_calls'] == 423312
    assert len(boots) == gpu['runtime_timing']['boot_calls'] == final['boot_count'] == 4705
    assert all(i['attrs']['implementation']=='native' and i['attrs']['operator_profile']==config['profile_id'] for i in boots)
    assert all(not p['host_requirements']['needs_secret_key'] and not p['host_requirements']['needs_host_compute'] for p in spec['boot_profiles'])
    assert config['linear_mode']==4 and config['project_real'] and config['bootstrap_ratio']==128 and config['evalmod_log_scale']==60
    assert config['native_probe_sha256'] == 'sha256:'+sha(ROOT/'evidence/native-probe.log')
    assert read('evidence/arithmetic-audit.json')['logical_model_unchanged']
    assert read('evidence/arithmetic-audit.json')['status']=='PASS'
    assert gpu['passed'] and gpu['native_nohoist'] and gpu['require_python_match']
    expected,actual = fixture['python_output'],gpu['poseidon_output']
    assert len(expected)==len(actual)==1000 and all(math.isfinite(x) for x in expected+actual)
    assert gpu['tolerances']==dict(absolute=0.1,python_absolute=0.1,relative=0.006,imaginary=0.01)
    errors = [abs(a-b) for a,b in zip(actual,expected)]
    assert all(e<=0.1+0.006*abs(v) for e,v in zip(errors,expected))
    maximum,rms = max(errors),math.sqrt(sum(e*e for e in errors)/1000)
    assert math.isclose(maximum,gpu['python_comparison']['max_abs'],abs_tol=1e-12)
    assert math.isclose(maximum,final['max_logit_error'],abs_tol=1e-12)
    assert math.isclose(rms,final['rms_logit_error'],abs_tol=1e-12)
    assert gpu['max_imaginary']<=0.01
    prediction = max(range(1000),key=actual.__getitem__)
    assert prediction==max(range(1000),key=expected.__getitem__)==fixture['label']==65
    assert final['status']=='PASS' and final['model_unchanged'] and final['original_gates_unchanged']
    print(json.dumps(dict(status='PASS',files=len(manifest),compute_calls=len(computes),native_boots=len(boots),
        prediction=prediction,max_error=maximum,rms_error=rms,scope='image0 only; compiled model-level DSL; not dataset accuracy'),indent=2))
