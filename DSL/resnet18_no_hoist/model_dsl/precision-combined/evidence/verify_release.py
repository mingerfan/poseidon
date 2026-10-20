"""Verify the published sources and unchanged single-image acceptance gates."""
import hashlib
import json
import math
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
CORRECTED = EVIDENCE.parent
MODEL = CORRECTED.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return json.loads((EVIDENCE/name).read_text())

if __name__ == '__main__':
    checked = 0
    external = []
    for source, digest in read('source-provenance.json').items():
        if '/resnet18-precision-combined-v1/' in source:
            path = CORRECTED/source.split('/resnet18-precision-combined-v1/',1)[1]
        elif '/resnet18-model-v2/' in source:
            relative = source.split('/resnet18-model-v2/',1)[1]
            if relative == 'compiler-build/hecate-opt':
                external.append(source)
                continue
            path = MODEL/relative
        else:
            raise AssertionError(source)
        assert sha(path) == digest, str(path)
        checked += 1
    assert checked == 14 and len(external) == 1
    semantic = json.loads((MODEL/'evidence/semantics-final.json').read_text())
    assert semantic['status'] == 'PASS'
    assert sha(MODEL/'ResNet18.py') == semantic['frontend_sha256']
    assert sha(MODEL/'imagenet_layout.py') == semantic['layout_sha256']
    full, clear, audit = read('gpu-full.json'), read('clear-full.json'), read('final-audit.json')
    assert full['passed'] and full['native_nohoist'] and full['require_python_match']
    assert clear['passed'] and clear['python_within_tolerance'] and clear['require_python_match']
    assert full['model_sha256'] == clear['model_sha256'] == semantic['model_sha256']
    assert full['plan_sha256'] == clear['plan_sha256'] == 'sha256:'+audit['plan_sha256']
    assert read('arithmetic-audit.json')['logical_model_unchanged']
    assert read('arithmetic-audit.json')['status'] == 'PASS'
    provenance = read('gpu-full.provenance.json')
    assert provenance['clear_sha256'] == sha(EVIDENCE/'clear-full.json')
    assert provenance['spec_sha256'] == sha(CORRECTED/'profiles-combined/operator-spec.json')
    assert provenance['native_config_sha256'] == sha(CORRECTED/'profiles-combined/native-config.json')
    expected, actual = clear['output'], full['poseidon_output']
    assert len(expected) == len(actual) == 1000
    assert all(math.isfinite(v) for v in expected+actual)
    tolerance = full['tolerances']
    assert tolerance == dict(absolute=0.1,python_absolute=0.1,relative=0.006,imaginary=0.01)
    errors = [abs(a-b) for a,b in zip(actual,expected)]
    assert all(error <= 0.1+0.006*abs(value) for error,value in zip(errors,expected))
    assert full['max_imaginary'] <= 0.01
    assert max(range(1000),key=actual.__getitem__) == semantic['label'] == audit['prediction'] == 65
    assert audit['status'] == 'PASS' and audit['original_gates_unchanged'] and audit['model_unchanged']
    assert full['runtime_timing']['boot_calls'] == audit['boot_count'] == 4705
    assert math.isclose(max(errors),audit['max_logit_error'],abs_tol=1e-12)
    assert math.isclose(math.sqrt(sum(e*e for e in errors)/1000),audit['rms_logit_error'],abs_tol=1e-12)
    print('PASS: 14 shipped source/config hashes; 1000 logits; original gates; final audit')
    print('External compiler binary, plan/bundle, FHE keys and dataset are retained on the server.')
