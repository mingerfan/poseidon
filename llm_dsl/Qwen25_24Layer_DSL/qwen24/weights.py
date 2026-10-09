"""Load NPY directories or NPZ archives with the explicit my_qwen2 weight names."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .spec import weight_shapes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, c):
    path = Path(path).resolve()
    shapes = weight_shapes(c)
    if path.is_dir():
        manifest = json.loads((path / 'weights.json').read_text(encoding='utf-8'))
        if manifest['config'] != c or set(manifest['arrays']) != set(shapes):
            raise ValueError('Weight manifest does not match model config/names')
        arrays = {}
        for name, record in manifest['arrays'].items():
            source = (path / record['file']).resolve()
            if not source.is_relative_to(path) or not source.is_file():
                raise ValueError('Invalid weight path: ' + name)
            if file_hash(source) != record['sha256']:
                raise ValueError('Weight checksum mismatch: ' + name)
            arrays[name] = np.load(source, mmap_mode='r', allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(shapes):
                raise ValueError('NPZ must contain the exact 291 named weights')
            arrays = {name: archive[name] for name in shapes}
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype.kind not in 'fiu':
            raise ValueError('Invalid weight shape/type: ' + name)
        # Chunked finite checks avoid a whole-embedding-table temporary.
        for start in range(0, len(value), 256):
            if not np.isfinite(value[start:start + 256]).all():
                raise ValueError('Nonfinite weight: ' + name)
    return arrays
