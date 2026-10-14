"""Rebuild only the existing local HecateFrontend target; no installs/downloads."""
import sys
import shlex

from hecate_python_env import enter_nix, WORK


if __name__ == "__main__":
    sys.exit(enter_nix("cmake --build " + shlex.quote(str(WORK / 'build-dacapo/hecate-18.1.2-nix')) + " "
                      "--target HecateFrontend --parallel 2", seconds=300))
