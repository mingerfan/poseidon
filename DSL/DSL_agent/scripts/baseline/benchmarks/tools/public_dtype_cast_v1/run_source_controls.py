import os,sys,shlex,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from hecate_python_env import enter_nix,VENV
p=argparse.ArgumentParser();p.add_argument("--baseline",action="store_true");p.add_argument("--inside",action="store_true");args=p.parse_args()
if args.inside:
 assert os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV
 target=HERE/"source_controls.py"
 os.execv(str(VENV/"bin/python"),[str(VENV/"bin/python"),"-B",str(target)])
else:
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*(["--baseline"]if args.baseline else[])])
 raise SystemExit(enter_nix(command,seconds=240,keep_env=(),environment="poseidon-cpu"))
