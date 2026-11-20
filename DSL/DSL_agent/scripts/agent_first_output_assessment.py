#!/usr/bin/env python3
"""Audit retained real-backend evidence against frozen DSL; never repair it.

This evidence import does not start the compiler or a native worker. The B
runner supplies the witness and retains its original source/SDK/native proofs.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'baseline'))
from delivery_component import read_bytes
from first_output_assessment import assess_once


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('delivery', 'binding', 'policy', 'witness', 'output'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args(argv)
    witness = json.loads(read_bytes(args.witness))
    result = assess_once(args.delivery, args.binding, json.loads(read_bytes(args.policy)),
                         args.output, lambda request, source: witness)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
