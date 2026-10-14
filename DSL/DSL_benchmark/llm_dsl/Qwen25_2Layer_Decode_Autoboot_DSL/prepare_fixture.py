"""Generate the fixed two-block synthetic weights and exact-function reference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source/Qwen25_2Layer_Decode_Test'))
from model_entry import configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    path = ROOT / 'source/Qwen25_24Layer_DSL/prepare_fixture.py'
    spec = importlib.util.spec_from_file_location('frozen_fixture_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.prepare(args.output_dir, configuration())
    # Match the active CKKS fixture's nonzero unused input slots.
    import numpy as np
    slots = np.load(args.output_dir / 'input_slots.npy', allow_pickle=False)
    slots[:, 896:898] = [.3, -.2]
    np.save(args.output_dir / 'input_slots.npy', slots)
    result['dirty_input_tail'] = {'slots': [896, 897], 'values': [.3, -.2]}
    (args.output_dir / 'fixture.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
