"""Verify the distributed files; generated fixtures may live alongside them."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
for name, record in manifest['files'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError('Missing or invalid path: ' + name)
    content = path.read_bytes()
    if len(content) != record['bytes'] or hashlib.sha256(content).hexdigest() != record['sha256']:
        raise RuntimeError('File changed: ' + name)
print('Verified', len(manifest['files']), 'files; full 24-layer CKKS/GPU remains unvalidated.')
