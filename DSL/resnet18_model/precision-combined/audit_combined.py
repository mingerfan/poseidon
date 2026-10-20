"""Audit the corrected encrypted model against the original unchanged accuracy gates."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'resnet18-model-v2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


if __name__=='__main__':
    plan_path, = (ROOT/'compiled-full').glob('*.runtime-plan.json')
    plan = json.loads(plan_path.read_text())
    full = json.loads((ROOT/'gpu-full.json').read_text())
    clear = json.loads((ROOT/'clear-full.json').read_text())
    fixture = json.loads((BASE/'fixture-final.json').read_text())
    semantic = json.loads((BASE/'semantics-final.json').read_text())
    old = json.loads((BASE/'gpu-full-pool31.json').read_text())
    provenance = json.loads((ROOT/'gpu-full.provenance.json').read_text())
    manifest = json.loads((ROOT/'source-provenance.json').read_text())
    for path,digest in manifest.items():
        assert sha(path)==digest,path
    assert full['passed'] and full['native_nohoist'] and full['require_python_match']
    assert clear['passed'] and clear['python_within_tolerance'] and clear['require_python_match']
    assert full['model_sha256']==clear['model_sha256']==fixture['model_sha256']==semantic['model_sha256']
    assert full['plan_sha256']==clear['plan_sha256']=='sha256:'+sha(plan_path)
    for key,path in (('binary_sha256',ROOT/'runtime-build/poseidon_runtime_gpu_mlp_e2e'),
                     ('plan_sha256',plan_path),('fixture_sha256',BASE/'fixture-final.json'),
                     ('clear_sha256',ROOT/'clear-full.json'),('spec_sha256',ROOT/'profiles-combined/operator-spec.json'),
                     ('native_config_sha256',ROOT/'profiles-combined/native-config.json')):
        assert provenance[key]==sha(path),key
    assert sha(BASE/'ResNet18.py')==semantic['frontend_sha256']
    assert sha(BASE/'imagenet_layout.py')==semantic['layout_sha256']
    graph = json.loads((ROOT/'arithmetic-audit.json').read_text())
    assert graph['status']=='PASS' and graph['logical_model_unchanged']
    boot_id = json.loads((ROOT/'profiles-combined/native-config.json').read_text())['profile_id']
    boots = [i for i in plan['execution'] if i.get('op')=='boot']
    assert boots and all(i['attrs']['implementation']=='native' and
        i['attrs']['operator_profile']==boot_id for i in boots)
    assert len(boots)==full['runtime_timing']['boot_calls']
    expected,actual = fixture['python_output'],full['poseidon_output']
    assert len(expected)==len(actual)==len(clear['output'])==1000
    assert all(math.isfinite(v) for v in expected+actual+clear['output'])
    assert max(abs(a-b) for a,b in zip(expected,clear['output']))<=1e-9
    errors = [abs(a-b) for a,b in zip(expected,actual)]
    maximum,rms = max(errors),math.sqrt(sum(e*e for e in errors)/1000)
    assert math.isclose(maximum,full['python_comparison']['max_abs'],abs_tol=1e-12)
    assert math.isclose(rms,full['python_comparison']['rms'],abs_tol=1e-12)
    tolerance = full['tolerances']
    assert tolerance['absolute']==tolerance['python_absolute']==0.1
    assert tolerance['relative']==0.006 and tolerance['imaginary']==0.01
    assert all(e<=0.1+0.006*abs(v) for e,v in zip(errors,expected))
    assert full['max_imaginary']<=0.01
    prediction = max(range(1000),key=actual.__getitem__)
    assert prediction==max(range(1000),key=expected.__getitem__)==fixture['label']
    report = dict(status='PASS',scope='single image0; corrected native bootstrap; compiled model DSL',
        evalmod_log_scale=60,bootstrap_ratio=128,model_unchanged=True,original_gates_unchanged=True,
        prediction=prediction,label=fixture['label'],max_logit_error=maximum,rms_logit_error=rms,
        baseline_max_error=old['python_comparison']['max_abs'],
        max_error_improvement=old['python_comparison']['max_abs']/maximum,
        max_imaginary=full['max_imaginary'],boot_count=len(boots),
        online_seconds=full['runtime_seconds'],plan_sha256=sha(plan_path))
    with (ROOT/'final-audit.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))
