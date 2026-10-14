"""Prepare frozen requests inside the existing pinned Python environment."""
import sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(HERE/"prepare_requests.py")])
raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
