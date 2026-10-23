"""Pack already normalized model tensors into ciphertext slot arrays."""

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root/"config/model.json").read_text())
    runtime = importlib.import_module(config["name"]+".runtime")
    entries = runtime.read_config()[2]["inputs"]
    runtime.save_arrays(args.output, runtime.pack_arrays(runtime.load_arrays(args.inputs, entries)), entries)


if __name__ == "__main__":
    main()
