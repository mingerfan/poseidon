"""Export a PT-DSL file using an existing development model's public weights."""

import argparse
from pathlib import Path

from .package import export_package
from ..dsl.parser import parse
from ..models.zoo import NAMES, get


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = args.program.read_text()
    graph = get(args.model, args.models_dir).import_graph()
    path = export_package(text, parse(text), graph.weights, args.out,
                          run_info=dict(program_path=args.program, run_dir=args.program.parent))
    print(path)


if __name__ == "__main__":
    main()
