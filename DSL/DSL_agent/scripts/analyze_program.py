#!/usr/bin/env python3
"""Analyze the fixed Runtime's storage for a compiled Poseidon program package."""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[3] if Path(__file__).resolve().parents[1].name == "DSL_agent" and Path(__file__).resolve().parents[2].name == "DSL" else Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "scripts/baseline"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, help="Budget within the existing native sandbox limit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from hecate_python_env import VENV, enter_nix
    if not args.inside:
        command = [str(VENV / "bin/python"), "-B", str(Path(__file__).resolve()), "--inside",
                   "--bundle", str(args.bundle.resolve()), "--output", str(args.output.resolve())]
        if args.memory_mib is not None:
            command += ["--memory-mib", str(args.memory_mib)]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 '
                         'OMP_NUM_THREADS=1 ' + shlex.join(command),
                         seconds=120, environment="poseidon-cpu")
    from application_planning.compiled_resources import analyze_program, canonical
    if os.environ.get("IN_NIX_SHELL") != "pure" or Path(sys.prefix) != VENV:
        parser.error("Pinned resource worker required")
    if args.output.exists() or not args.output.resolve().parent.is_dir():
        parser.error("Fresh output path required")
    record = analyze_program(args.bundle,
        memory_budget_bytes=None if args.memory_mib is None else args.memory_mib * 1024**2)
    with args.output.open("xb") as stream:
        stream.write(canonical(record))
    print(json.dumps(dict(status=record["status"], output=str(args.output.resolve()),
        retained_payload_lower_bytes=record["resources"]["known_retained_payload_lower_bytes"]["value"],
        peak_memory="unknown", compilation_performed=False, encrypted_execution=False, paid_calls=0)))
    return 1 if record["status"] == "blocked" else 0

if __name__ == "__main__":
    raise SystemExit(main())
