"""Export the package graph, or inspect its interface without a native import."""

import argparse
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bytecode", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT/"config/model.json").read_text())
    signature = json.loads((ROOT/"interfaces/default.json").read_text())
    approximations = json.loads((ROOT/"config/approximations.json").read_text())
    if args.plan:
        signature["approximations"] = {name: {key: value for key, value in item.items()
                                            if key != "coefficients"} for name, item in approximations.items()}
        print(json.dumps(signature, indent=2))
        return
    if args.weights is None or args.output_dir is None:
        parser.error("--weights and --output-dir are required unless --plan")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing artifacts are preserved")
    import hecate as hc

    runtime = importlib.import_module(config["name"]+".runtime")
    weights = importlib.import_module(config["name"]+".weights").load(args.weights)
    runtime.register(hc, weights)
    args.output_dir.mkdir(parents=True)
    if args.bytecode:
        print(hc.save(str(args.output_dir), str(args.output_dir), bytecode=True))
    else:
        print(hc.save(str(args.output_dir), str(args.output_dir)))
    signature.update(function="_hecate_"+config["name"], mlir="trace_model.mlirbc" if args.bytecode else "trace_model.mlir",
                     native_export_format="mlir-bytecode" if args.bytecode else "mlir-text",
                     native_tracing_completed=True,
                     weights_binding=dict(file=args.weights.name, bytes=args.weights.stat().st_size))
    with (args.output_dir/"signature.json").open("x") as stream:
        json.dump(signature, stream, indent=2)


if __name__ == "__main__":
    main()
