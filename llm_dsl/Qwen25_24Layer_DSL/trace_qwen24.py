"""Export a full 24-block Hecate graph, or inspect its interface with --plan."""
import argparse
import json
from pathlib import Path
from qwen24 import spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prefill', 'decode', 'decode-pair'), default='prefill')
    p.add_argument('--positions', type=int, nargs='+', default=[0, 1])
    p.add_argument('--cached-positions', type=int, nargs='+', default=[])
    p.add_argument('--config', type=Path, default=spec.CONFIG)
    p.add_argument('--approximations', type=Path, default=spec.PROFILES)
    p.add_argument('--weights', type=Path)
    p.add_argument('--slots', type=int, default=32768)
    p.add_argument('--logit-chunk-size', type=int, default=1024)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--plan', action='store_true', help='No native import, weights or tracing')
    a = p.parse_args()
    sig = spec.signature(spec.config(a.config), a.mode, a.positions, a.cached_positions,
                         a.slots, a.logit_chunk_size)
    profiles = json.loads(a.approximations.read_text(encoding='utf-8'))
    if len(profiles['blocks']) != 24:
        p.error('Exactly 24 explicit block profiles required')
    for profile in profiles['blocks']:
        if 'softmax' in profile and len(a.positions + a.cached_positions) > profile['softmax']['max_length']:
            p.error('Requested length exceeds selected Softmax profile')
    sig['approximations'] = profiles
    if a.plan:
        print(json.dumps(sig, indent=2))
        return
    if a.weights is None or a.output_dir is None:
        p.error('--weights and --output-dir are required unless --plan')
    if a.output_dir.exists():
        p.error('Use a new output directory; existing artifacts are preserved')
    import hecate as hc
    from qwen24.runtime import load_model, register
    from qwen24.weights import file_hash
    model = load_model(a.weights, a.config, a.approximations, a.slots, a.logit_chunk_size)
    register(hc, model, sig)
    a.output_dir.mkdir(parents=True)
    print(hc.save(str(a.output_dir), str(a.output_dir)))
    sig['native_tracing_completed'] = True
    sig['weights_binding'] = file_hash(a.weights / 'weights.json' if a.weights.is_dir() else a.weights)
    with (a.output_dir / 'signature.json').open('x', encoding='utf-8') as f:
        json.dump(sig, f, indent=2)


if __name__ == '__main__':
    main()
