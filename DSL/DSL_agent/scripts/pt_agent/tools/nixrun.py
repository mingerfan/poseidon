"""Run the pinned venv Python inside the project's offline pure Nix shell."""

import argparse
import os
from pathlib import Path
import shlex
import sys

from ..paths import WORK_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    if not args or options.seconds <= 0:
        parser.error("Provide Python arguments after -- and a positive --seconds")
    os.environ.setdefault("POSEIDON_PLATFORM", "aarch64-linux")
    os.environ.setdefault("POSEIDON_WORK_ROOT", str(WORK_ROOT))
    scripts_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(scripts_dir / "baseline"))
    from hecate_python_env import enter_nix, ROOT, VENV

    command = ("cd " + shlex.quote(str(scripts_dir)) + " && env LD_LIBRARY_PATH=\"$HECATE_PYTHON_LIBRARY_PATH\" "
               "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
               + shlex.quote(str(VENV / "bin/python")) + " -B " + shlex.join(args))
    os.chdir(ROOT)
    return enter_nix(command, seconds=options.seconds)


if __name__ == "__main__":
    sys.exit(main())
