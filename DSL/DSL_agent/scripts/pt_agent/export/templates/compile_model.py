"""Substitute paths into the exported compiler command and execute it once."""

import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    substitutions = dict(hecate_opt=str(args.hecate_opt.resolve()), input=str(args.input.resolve()),
                         package=str(ROOT), output_dir=str(args.output_dir.resolve()))
    argv = [arg.format(**substitutions) for arg in json.loads((ROOT/"config/compile_argv.json").read_text())]
    if args.dry_run:
        print(json.dumps(argv))
        return 0
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing artifacts are preserved")
    args.output_dir.mkdir(parents=True)
    return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
