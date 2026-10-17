"""Independent audit of the model-level framework ResNet20 real Native Boot run."""
import gzip,hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parent/'evidence'
package=root.parent
manifest=json.loads((package/'manifest.sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected, name
semantic=json.loads((root/'frontend-semantics.json').read_text())
fixture=json.loads((package/'fixtures/image0.json').read_text())
assert hashlib.sha256((package/'ResNet20.py').read_bytes()).hexdigest()==semantic['frontend_sha256']
assert hashlib.sha256((root/'frontend-semantics.json').read_bytes()).hexdigest()==fixture['semantic_report_sha256']
assert semantic['model_sha256']==fixture['model_sha256'] and semantic['weights']==116
read=lambda p:json.loads(((root.parent/p) if p.startswith('profiles/') else (root/p)).read_text())
digest=lambda p:'sha256:'+hashlib.sha256(((root.parent/p) if p.startswith('profiles/') else (root/p)).read_bytes()).hexdigest()
r=read('framework-gpu-full-v3.json');p=json.loads(gzip.decompress((root/'runtime-plan.json.gz').read_bytes()))
s=read('profiles/operator-spec.json');c=read('profiles/native-config.json');clear=read('compiled-clear-v4.json')
assert r['plan_sha256']==('sha256:'+hashlib.sha256(gzip.decompress((root/'runtime-plan.json.gz').read_bytes())).hexdigest())==clear['plan_sha256']
assert r['operator_spec_sha256']==digest('profiles/operator-spec.json')==p['target']['operator_spec']['source_sha256']
assert r['model_sha256']==clear['model_sha256'] and clear['passed'] and clear['require_python_match']
assert clear['python_max_abs_diff']==0 and clear['output']==r['python_output']
ops=[i for phase in ('initialization','execution','finalization') for i in p[phase]]
computes=[i for i in ops if i['kind']=='compute'];boots=[i for i in computes if i['op']=='boot']
assert len(computes)==r['runtime_timing']['compute_calls']==16222
assert len(boots)==r['runtime_timing']['boot_calls']==466
assert all(i['attrs']['implementation']=='native' and i['attrs']['operator_profile']==c['profile_id'] for i in boots)
assert c['project_real'] is True and c['linear_mode']==4 and r['native_nohoist'] and r['require_python_match']
assert all(b['implementation']=='native' and not b['host_requirements']['needs_secret_key'] and not b['host_requirements']['needs_host_compute'] for b in s['boot_profiles'])
assert all(math.isfinite(x) for x in r['poseidon_output'])
errors=[x-y for x,y in zip(r['poseidon_output'],r['python_output'])]
maximum=max(map(abs,errors));rms=math.sqrt(sum(e*e for e in errors)/len(errors))
assert abs(maximum-r['python_comparison']['max_abs'])<1e-12 and abs(rms-r['python_comparison']['rms'])<1e-12
prediction=max(range(10),key=r['poseidon_output'].__getitem__)
reference=max(range(10),key=r['python_output'].__getitem__)
assert prediction==reference==3 and r['passed']
assert r['max_imaginary']<=r['tolerances']['imaginary']
assert 'FRAMEWORK_GPU_FULL_EXIT=0' in (root/'framework-gpu-full-v3.log').read_text()
probe=(root/'native-nohoist-real-profile-v2.log').read_text()
assert '[PASS] RuntimePlan native GPU bootstrap' in probe and 'project_real=1' in probe and 'linear_mode=4' in probe
assert c['native_probe_sha256']==digest('native-nohoist-real-profile-v2.log')
result=dict(status='PASS',scope='model-level Python -> full Dacapo -> RuntimePlan -> SequentialRuntime -> Poseidon Native NoHoistBsgs',
    image=0,label=3,prediction=prediction,reference_prediction=reference,classification_correct=True,
    clear_plan_max_error=0,native_boot_calls=len(boots),compute_calls=len(computes),
    max_logit_error=maximum,rms_logit_error=rms,max_imaginary=r['max_imaginary'],
    online_seconds=r['runtime_seconds'],plan_sha256=r['plan_sha256'],operator_spec_sha256=r['operator_spec_sha256'],
    report_sha256=digest('framework-gpu-full-v3.json'),native_real_projection=True,
    limitations=['single image, not dataset accuracy','no same-cipher native-vs-DSL comparison',
        'reported CKKS errors are nonzero','E2E internal limits are not user-specified acceptance targets'])
assert result == read('framework-resnet20-real-full-v3-acceptance.json')
print(json.dumps(result,indent=2))
