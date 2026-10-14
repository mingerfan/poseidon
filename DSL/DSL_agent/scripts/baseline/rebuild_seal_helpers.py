"""Rebuild local key/metadata helpers only, using already installed SEAL 4.0.0."""
import sys
import shlex
from hecate_python_env import enter_nix, WORK

if __name__ == "__main__":
    sys.exit(enter_nix("cmake --build " + shlex.quote(str(WORK / 'build-dacapo/seal-golden-keys')) + " "
                      "--parallel 2", seconds=180))
