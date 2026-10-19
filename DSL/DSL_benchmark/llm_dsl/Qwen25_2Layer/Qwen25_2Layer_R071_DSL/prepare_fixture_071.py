"""Explicit opt-in synthetic fixture; full dimensions produce about 1.2 GB of weights."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen/model/package'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    from prepare_fixture import prepare
    from qwen24.spec import config
    c = config()
    summary = prepare(a.output_dir, c)
    with np.load(a.output_dir/'reference.npz', allow_pickle=False) as ref:
        outputs = [ref['logits'][i:i+32768].tolist() for i in range(0,c['vocab_size'],32768)]
        outputs += [np.stack((ref['keys'][i],ref['values'][i]),axis=1).reshape(-1).tolist() for i in range(2)]
    names = [f'logits.{i}:{min(i+32768,c["vocab_size"])}' for i in range(0,c['vocab_size'],32768)]
    names += ['layer_0.kv_cache', 'layer_1.kv_cache']
    fixture = dict(inputs=np.load(a.output_dir/'input_slots.npy',allow_pickle=False).tolist(),
        expected=outputs, output_names=names, num_layers=2, tokens=2,
        vocab_size=c['vocab_size'], synthetic_weights=True, expected_boots=154)
    (a.output_dir/'runtime-fixture.json').write_text(json.dumps(fixture), encoding='utf-8')
    (a.output_dir/'fixture.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
