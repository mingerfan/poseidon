#!/usr/bin/env python3
"""Preserve small evidence and compressed plans, excluding model/bundle payloads."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

base = Path('/home/xuming/poseidon-model-regression-20261021')
roots = [p for p in base.iterdir() if p.is_dir() and not p.is_symlink() and
         (p.name == 'profiles' or p.name.startswith(('resnet20-', 'resnet-runtime-profiles', 'qwen-prefix',
                              'qwen-projection', 'qwen-full-', 'optimized-before')))
         and p.name != 'qwen-full-before-missing-config']
manifest = []
with tarfile.open('/home/xuming/poseidon-model-regression-evidence-20261021.tar.gz', 'w:gz') as archive:
    files = [p for p in base.iterdir() if p.is_file() and p.suffix in ['.json', '.log', '.py', '.cpp', '.sh']]
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file(): continue
            if path.name == 'clear.log': continue  # superseded harness setup attempt
            if path.suffix in ['.json', '.log', '.mlirattr'] or path.name.endswith('.runtime-plan.bin'):
                files.append(path)
            elif path.name == 'logical.mlir' and root.name.startswith('resnet20-'):
                files.append(path)
    for path in sorted(set(files)):
        relative = path.relative_to(base)
        data = path.read_bytes()
        row = dict(path=str(relative), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        compress = path.suffix in ['.bin', '.mlir'] or path.name == 'gpu-decoded.json'
        if compress:
            data = gzip.compress(data, mtime=0)
            destination = str(relative)+'.gz'
        else: destination = str(relative)
        row['archived_as'] = destination
        manifest.append(row)
        info = tarfile.TarInfo(destination)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    data = (json.dumps(manifest, indent=2)+'\n').encode()
    info = tarfile.TarInfo('evidence-manifest.json')
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
print('Archived', len(manifest), 'files')
