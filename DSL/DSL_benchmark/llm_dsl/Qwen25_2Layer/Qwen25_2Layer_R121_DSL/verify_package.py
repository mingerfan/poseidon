"""Check package hashes, syntax, provenance and historical full acceptance."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=read(ROOT/'manifest.json')
expected=set(m['files'])
# Nested manifests are also covered; only the top-level manifest is excluded.
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'manifest.json' and '__pycache__' not in p.parts}
assert actual==expected,(actual-expected,expected-actual)
for name,entry in m['files'].items():
    p=ROOT/name
    assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
    if p.suffix=='.py':ast.parse(p.read_bytes(),name)
for name,binding in read(ROOT/'source-binding.json')['files'].items():
    assert sha(ROOT/name)==binding['sha256'],name
p=read(ROOT/'evidence/completion-verification.json')
r=read(ROOT/'evidence/result.json')
assert p['passed'] and p['model_processes_exited'] and p['outputs_checked']==7
assert p['result_sha256']==sha(ROOT/'evidence/result.json')
assert r['passed'] and r['complete_decoder_blocks']==2 and r['diagnostic_ciphertext_copies']==0
assert read(ROOT/'frozen/model/package/config/model.json')['num_layers']==2
print(json.dumps(dict(passed=True,files=len(expected),historical_full_ckks=True,current_ckks_executed=False)))
