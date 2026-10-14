"""Enter the existing fixed environment once; never install dependencies."""
import argparse,json,os,shlex,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]))
from hecate_python_env import VENV,enter_nix
parser=argparse.ArgumentParser();parser.add_argument("target",choices=("source_controls.py","application_controls.py","prepare_offline.py","verify_offline.py"));args=parser.parse_args()
command='PYTHONPATH=/home/lhohy/Code/Poseidon/scripts/baseline LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(HERE/args.target)])
raise SystemExit(enter_nix(command,seconds=300,keep_env=(),environment="poseidon-cpu"))
