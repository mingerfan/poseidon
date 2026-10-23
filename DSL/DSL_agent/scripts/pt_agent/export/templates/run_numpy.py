"""Evaluate structured kernels with float64 slot arithmetic for self-checks."""

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root/"config/model.json").read_text())
    runtime = importlib.import_module(config["name"]+".runtime")
    weights = importlib.import_module(config["name"]+".weights").load(args.weights)
    interface = runtime.read_config()[2]
    arrays = runtime.load_arrays(args.inputs, interface["inputs"])
    runtime.save_arrays(args.output, runtime.run_numpy(weights, arrays), interface["outputs"])


if __name__ == "__main__":
    main()
