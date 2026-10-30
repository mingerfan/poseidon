"""Trace exactly the tested round-071 decode-pair strategy."""
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
STRATEGY = 'degree10p2_pair13_silu_canonical_fold2'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-B', str(ROOT/'frozen/model/trace_combined_model.py'),
        '--mode', 'decode-pair', '--positions', '0', '1', '--rms-degree-strategy', STRATEGY,
        '--weights', str(a.weights.resolve()), '--output-dir', str(a.output_dir.resolve())], check=True)

if __name__ == '__main__':
    main()
