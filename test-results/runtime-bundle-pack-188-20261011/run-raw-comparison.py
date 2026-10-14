#!/usr/bin/env python3
"""Run one raw I/O pair on existing bundles, with no blob hashes."""
import json
import struct
import subprocess
import tempfile
from pathlib import Path

root = Path('/home/xuming/poseidon-runtime-io-20261010')
loose = root / 'artifacts/gpu4/qwen24._hecate_qwen25_24layer.bundle'
packed = root / 'artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.bundle'
source = root / 'raw-io-comparison.cpp'
binary = root / 'raw-io-comparison'
subprocess.run(['g++', '-std=c++17', '-O2', '-DNDEBUG', str(source), '-o', str(binary)], check=True)
# Only metadata is loaded here. Keep the original manifest order used by pack.
manifest = json.loads((loose / 'manifest.json').read_bytes())
with tempfile.TemporaryDirectory(prefix='bundle-raw-io-', dir=root) as temporary:
    index = Path(temporary) / 'index.bin'
    with index.open('wb') as output:
        for entry in manifest['blobs']:
            output.write(struct.pack('<Q64s', entry['byte_length'], entry['content'][7:].encode('ascii')))
    del manifest
    for mode, directory in [('files', loose), ('pack', packed)]:
        with (root / f'results/raw-read-{mode}.json').open('w') as output:
            subprocess.run(['nice', '-n', '10', str(binary), mode, str(directory), str(index)],
                           stdout=output, check=True)
        print((root / f'results/raw-read-{mode}.json').read_text(), flush=True)
