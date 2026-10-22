"""Command-line PT-DSL parsing, optional type checking and canonical printing."""

import argparse
from pathlib import Path

from .errors import PTDSLError
from .parser import parse
from .printer import print_program
from .typecheck import check, load_weight_shapes
from ..examples.shapes import mlp_weight_shapes, resnet20_weight_shapes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path)
    cli.add_argument("--weights")
    cli.add_argument("--print", action="store_true", dest="print_source")
    args = cli.parse_args()
    try:
        program = parse(args.file.read_text())
    except PTDSLError as exc:
        print(f"ERROR {exc.code} line {exc.lineno}: {exc.msg}")
        return 2
    status = 0
    if args.weights is not None:
        examples = {"mlp": mlp_weight_shapes, "resnet20": resnet20_weight_shapes}
        weights = examples[args.weights]() if args.weights in examples else load_weight_shapes(args.weights)
        result = check(program, weights)
        for diag in result.diags:
            print(diag)
        status = 0 if result.ok else 1
    if args.print_source:
        print(print_program(program), end="")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
