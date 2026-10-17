"""Generate 24 distinct synthetic layers; never download pretrained weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from qwen24.spec import CONFIG, config, weight_shapes
from qwen24.weights import file_hash, load
from qwen24.reference import forward, assert_provisional_bounds


def prepare(output, c, slots=32768, seed=20261009):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new fixture directory')
    if slots < c['hidden_size']:
        raise ValueError('Embedding does not fit slots')
    directory = output / 'weights'
    directory.mkdir(parents=True)
    arrays = {}
    for name, shape in weight_shapes(c).items():
        # Per-name stable seeds keep all 24 layers distinct and order-independent.
        key = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(key)
        path = directory / (name + '.npy')
        value = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        for start in range(0, shape[0], 256):
            piece = (min(256, shape[0] - start),) + shape[1:]
            if name == 'emb_weight.weight':
                sample = rng.normal(size=piece)
                sample /= np.sqrt(np.mean(sample * sample, axis=1, keepdims=True))
            elif 'Normal.weight' in name or name == 'final_normal.weight':
                sample = rng.uniform(.98, 1.02, size=piece)
            else:
                sample = rng.normal(scale=.05/math.sqrt(shape[-1]) if len(shape) == 2 else .002, size=piece)
            value[start:start+len(sample)] = sample
        value.flush()
        del value
        arrays[name] = dict(file=path.name, shape=list(shape), dtype='float32', sha256=file_hash(path))
    record = dict(format='qwen24-npy-weights-v1', synthetic=True, seed=seed, config=c,
                  tied_embedding_lm_head=False, arrays=arrays)
    with (directory / 'weights.json').open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    weights = load(directory, c)
    ids = np.array([1, 3], dtype=np.int64)
    embeddings = np.array(weights['emb_weight.weight'][ids], dtype=np.float64)
    packed = np.zeros((2, slots))
    packed[:, :c['hidden_size']] = embeddings
    target, bounds = forward(weights, embeddings, [0, 1], c)
    assert_provisional_bounds(bounds)
    for name, value in [('token_ids', ids), ('input_embeddings', embeddings), ('input_slots', packed)]:
        with (output / (name + '.npy')).open('xb') as f:
            np.save(f, value)
    with (output / 'reference.npz').open('xb') as f:
        np.savez(f, **target)
    with (output / 'reference_bounds.json').open('x', encoding='utf-8') as f:
        json.dump(bounds, f, indent=2)
    return dict(config=c, synthetic=True, seed=seed, positions=[0, 1], slots=slots,
                weight_files=len(arrays), weight_bytes=sum(math.prod(s)*4 for s in weight_shapes(c).values()),
                reference='exact sqrt, sigmoid/SiLU and softmax; no CKKS',
                warning='Synthetic acceptance does not calibrate pretrained Qwen weights')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--slots', type=int, default=32768)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.output_dir, config(a.config), a.slots)
    with (a.output_dir / 'fixture.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
