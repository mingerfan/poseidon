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
    parser.add_argument("--mode", choices=("compiler", "managed"), default="compiler")
    args = parser.parse_args()
    substitutions = dict(hecate_opt=str(args.hecate_opt.resolve()), input=str(args.input.resolve()),
                         package=str(ROOT), output_dir=str(args.output_dir.resolve()))
    if args.mode == 'managed':
        import importlib
        config = json.loads((ROOT/'config/model.json').read_text())
        streams = importlib.import_module(config['name']+'.core.manage.streams')
        template = json.loads((ROOT/'config/managed_compile_argv.json').read_text())
        commands = [[a.format(**substitutions) for a in argv] for argv in template]
        if args.dry_run:
            print(json.dumps(commands))
            return 0
        if args.output_dir.exists():
            parser.error('Use a new output directory; existing artifacts are preserved')
        args.output_dir.mkdir(parents=True)
        for argv in commands:
            result = streams.run(argv,cwd=args.output_dir)
            if result.returncode:
                return result.returncode
        return 0
    argv = [arg.format(**substitutions) for arg in json.loads((ROOT/"config/compile_argv.json").read_text())]
    if args.dry_run:
        print(json.dumps(argv))
        return 0
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing artifacts are preserved")
    args.output_dir.mkdir(parents=True)
    if args.input.suffix=='.gz' or Path(str(args.input)+'.gz').exists():
        import importlib
        config = json.loads((ROOT/'config/model.json').read_text())
        streams = importlib.import_module(config['name']+'.core.manage.streams')
        return streams.run(argv,cwd=args.output_dir).returncode
    return subprocess.run(argv, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
