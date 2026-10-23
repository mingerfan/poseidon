"""Decode decrypted slot arrays using the recorded interface layouts."""

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root/"config/model.json").read_text())
    runtime = importlib.import_module(config["name"]+".runtime")
    signature = json.loads(args.signature.read_text())
    arrays = runtime.load_arrays(args.slots, signature["outputs"])
    runtime.save_arrays(args.output, runtime.unpack_arrays(arrays, signature), signature["outputs"])


if __name__ == "__main__":
    main()
