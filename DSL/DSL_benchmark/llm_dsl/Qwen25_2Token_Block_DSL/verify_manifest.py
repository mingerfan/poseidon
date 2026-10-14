"""Verify exact handoff contents against the included SHA-256 inventory."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name,record in manifest['files'].items():
    path=root/name
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise RuntimeError('Missing or invalid path: '+name)
    content=path.read_bytes()
    if len(content)!=record['bytes'] or hashlib.sha256(content).hexdigest()!=record['sha256']:
        raise RuntimeError('File changed: '+name)
print('Verified',len(manifest['files']),'files. Scope:',manifest['scope'])
