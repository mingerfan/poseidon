"""Verify package bytes without importing Hecate or running the model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid file: ' + name)
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Checksum mismatch: ' + name)
    binding = json.loads((ROOT / 'source-binding.json').read_text(encoding='utf-8'))
    for name, expected in binding.items():
        if manifest['files'][name]['sha256'] != expected['sha256']:
            raise ValueError('Frozen server source mismatch: ' + name)
    print(json.dumps(dict(passed=True, files=len(manifest['files']), frozen_server_files=len(binding))))


if __name__ == '__main__':
    main()
