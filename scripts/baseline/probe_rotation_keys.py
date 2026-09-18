"""Read actual public Galois key files; no encryption, decryption or API calls."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import sys

from hecate_python_env import ROOT, WORK, VENV, enter_nix
from seal_artifact_gate import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--expect-missing", action="store_true")
    parser.add_argument("--report", type=Path, help="New evidence JSON under the native results directory")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Requires source root")
    require(args.keys.resolve().is_relative_to(WORK / "results") and args.keys.is_dir(), "Invalid key directory")
    require(args.report is None or (args.report.resolve().is_relative_to(WORK / "results") and
            args.report.parent.is_dir() and not args.report.exists()), "Report must be a new native result file")
    require(1 <= len(args.steps) <= 6 and len(set(args.steps)) == len(args.steps) and
            all(s in (-3, -2, -1, 1, 2, 3) for s in args.steps), "Invalid steps")
    if not args.inside:
        options = ["--inside", "--keys", str(args.keys), "--steps", *map(str, args.steps)]
        if args.expect_missing:
            options.append("--expect-missing")
        if args.report is not None:
            options += ["--report", str(args.report)]
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
                   shlex.join([str(VENV / "bin/python"), str(Path(__file__).resolve()), *options]))
        return enter_nix(command, seconds=60)
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Requires pinned Nix/venv")
    lib = ctypes.CDLL(str(WORK / "build-dacapo/seal-golden-keys/libseal_golden_metadata.so"))
    lib.verify_galois_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
    lib.verify_galois_file.restype = ctypes.c_int
    steps = (ctypes.c_int64 * len(args.steps))(*args.steps)
    code = lib.verify_galois_file(os.fsencode(args.keys / "parm.seal"), os.fsencode(args.keys / "gal.seal"), steps, len(args.steps))
    expected = 2 if args.expect_missing else 0
    result = dict(actual_key_file_checked=True, required_steps=args.steps, result_code=code,
                  expected_result_code=expected, matched=code == expected, encrypted_execution=False)
    if args.report is not None:
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result))
    return 0 if code == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
