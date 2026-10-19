"""Verify package bytes, frozen-source bindings and archived CPU acceptance."""
import ast
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest = json.loads((ROOT/'manifest.json').read_text('utf-8'))
    for name, sha in manifest['files'].items():
        p = (ROOT/name).resolve()
        assert p.is_relative_to(ROOT.resolve()) and digest(p) == sha, name
        if p.suffix == '.py': ast.parse(p.read_text('utf-8-sig'), filename=name)
    for name, entry in json.loads((ROOT/'source-binding.json').read_text('utf-8'))['files'].items():
        assert digest(ROOT/name) == entry['sha256'], name
    check = json.loads((ROOT/'evidence/completion-verification.json').read_text('utf-8'))
    assert check['passed'] and check['boot_count'] == 154 and check['outputs_checked'] == 7
    assert digest(ROOT/'evidence/result.json') == check['result_sha256']
    data = json.loads((ROOT/'evidence/result.json').read_text('utf-8'))
    assert data['passed'] and len(data['outputs']) == 7 and len(data['bootstraps']) == 154
    for key in ('max_abs_error','max_imaginary','max_abs_tail'):
        assert check[key] < check['threshold']
    print(json.dumps(dict(passed=True,files=len(manifest['files']),frozen_hashes_verified=True,
        historical_cpu_ckks_passed=True,new_ckks_executed=False,gpu_validated=False)))
if __name__ == '__main__': main()
