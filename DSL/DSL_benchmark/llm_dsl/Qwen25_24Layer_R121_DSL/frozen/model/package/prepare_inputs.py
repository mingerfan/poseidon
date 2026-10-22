"""Client-side embedding lookup and slot packing; encryption stays with the client."""
import argparse
from pathlib import Path
import numpy as np
from qwen24.spec import CONFIG, config
from qwen24.weights import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--token-ids', type=int, nargs='+', required=True)
    p.add_argument('--slots', type=int, default=32768)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = config(a.config)
    if a.output.exists():
        p.error('Use a new output path')
    if a.slots < c['hidden_size'] or any(t < 0 or t >= c['vocab_size'] for t in a.token_ids):
        p.error('Invalid slots or token IDs')
    weights = load(a.weights, c)
    values = np.zeros((len(a.token_ids), a.slots), dtype=np.float64)
    values[:, :c['hidden_size']] = weights['emb_weight.weight'][a.token_ids]
    with a.output.open('xb') as f:
        np.save(f, values)
    print('Saved plaintext slot vectors; encrypt each row before running the DSL.')


if __name__ == '__main__':
    main()
