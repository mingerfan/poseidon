"""Independently recompute final metrics and check plan/report/source provenance."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-prefix', default='gpu-full')
    parser.add_argument('--report', default='final-audit.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan_path = next((root/'compiled-jit').glob('*.runtime-plan.json'))
    plan = json.loads(plan_path.read_text())
    full = json.loads((root/(args.gpu_prefix+'.json')).read_text())
    clear = json.loads((root/'clear.json').read_text())
    fixture = json.loads((root/'fixture-final.json').read_text())
    semantic = json.loads((root/'semantics-final.json').read_text())
    provenance = json.loads((root/(args.gpu_prefix+'.provenance.json')).read_text())
    assert full['passed'] and full['native_nohoist'] and clear['passed']
    assert clear['require_python_match'] and clear['python_within_tolerance']
    assert len(plan['external_inputs']) == len(fixture['packed_inputs']) == 8
    assert len(plan['final_outputs']) == 1
    assert full['model_sha256'] == clear['model_sha256'] == fixture['model_sha256'] == semantic['model_sha256']
    assert full['plan_sha256'] == clear['plan_sha256'] == 'sha256:'+sha(plan_path)
    assert provenance['plan_sha256'] == sha(plan_path)
    assert semantic['frontend_sha256'] == sha(root/'ResNet18.py')
    assert semantic['layout_sha256'] == sha(root/'imagenet_layout.py')
    assert provenance['binary_sha256'] == sha(root/'runtime-build-final/poseidon_runtime_gpu_mlp_e2e')
    counts = Counter(i.get('op') for i in plan['execution'] if i['kind'] == 'compute')
    boots = [i for i in plan['execution'] if i.get('op') == 'boot']
    assert boots and all(i['attrs']['implementation'] == 'native' for i in boots)
    assert full['runtime_timing']['boot_calls'] == len(boots)
    expected, actual = fixture['python_output'],full['poseidon_output']
    assert len(expected) == len(actual) == len(clear['output']) == 1000
    assert all(math.isfinite(v) for v in expected+actual)
    errors = [abs(a-b) for a,b in zip(expected,actual)]
    max_error = max(errors)
    rms = math.sqrt(sum(e*e for e in errors)/1000)
    assert math.isclose(max_error,full['python_comparison']['max_abs'],abs_tol=1e-12)
    assert math.isclose(rms,full['python_comparison']['rms'],abs_tol=1e-12)
    tolerance = full['tolerances']
    assert tolerance['absolute'] == tolerance['python_absolute'] == 0.1
    assert tolerance['relative'] == 0.006 and tolerance['imaginary'] == 0.01
    assert all(e <= 0.1+0.006*abs(v) for e,v in zip(errors,expected))
    assert full['max_imaginary'] <= 0.01
    predicted = max(range(1000),key=actual.__getitem__)
    reference = max(range(1000),key=expected.__getitem__)
    assert predicted == reference
    report = dict(status='PASS',scope='single image0; compiled model-level DSL; not dataset accuracy',
                  prediction=predicted,reference_prediction=reference,label=fixture['label'],
                  label_match=predicted==fixture['label'],compute_count=sum(counts.values()),
                  boot_count=len(boots),max_logit_error=max_error,rms_logit_error=rms,
                  max_imaginary=full['max_imaginary'],online_seconds=full['runtime_seconds'],
                  semantic_dense_max_error=semantic['dense_max_error'],
                  plan_sha256=sha(plan_path),operations=dict(counts))
    with (root/args.report).open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report,indent=2))
