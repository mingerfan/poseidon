"""Run the pinned venv Python inside the project's offline pure Nix shell."""

import argparse
import os
from pathlib import Path
import shlex
import sys

from ..paths import ENV_VARS, WORK_ROOT


def forwarded_env() -> str:
    """The pure shell drops the caller's environment; forward the pt_agent location overrides explicitly."""
    return "".join(f"{name}={shlex.quote(os.environ[name])} " for name in ENV_VARS if os.environ.get(name, "").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--python", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.args[1:] if options.args[:1] == ["--"] else options.args
    if not args or options.seconds <= 0:
        parser.error("Provide Python arguments after -- and a positive --seconds")
    if options.python is not None:
        options.python = options.python.expanduser().absolute()
        if not options.python.is_file() and not options.python.is_symlink():
            parser.error("--python must name an existing interpreter")
    os.environ.setdefault("POSEIDON_PLATFORM", "aarch64-linux")
    os.environ.setdefault("POSEIDON_WORK_ROOT", str(WORK_ROOT))
    scripts_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(scripts_dir / "baseline"))
    from hecate_python_env import enter_nix, ROOT, VENV

    interpreter = options.python if options.python is not None else VENV / "bin/python"
    forwarded = forwarded_env()
    interpreter_check = ("test -x " + shlex.quote(str(interpreter)) + " && "
                         if options.python is not None else "")
    command = ("cd " + shlex.quote(str(scripts_dir)) + " && " + interpreter_check
               + "env LD_LIBRARY_PATH=\"$HECATE_PYTHON_LIBRARY_PATH\" "
               + forwarded + "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
               + shlex.quote(str(interpreter)) + " -B " + shlex.join(args))
    os.chdir(ROOT)
    return enter_nix(command, seconds=options.seconds)


if __name__ == "__main__":
    sys.exit(main())
