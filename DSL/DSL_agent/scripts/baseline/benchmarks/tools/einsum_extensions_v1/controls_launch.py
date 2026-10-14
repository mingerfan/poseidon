import os,pathlib,sys,shlex
ROOT=pathlib.Path.cwd();BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([
 str(VENV/"bin/python"),"-B",str(ROOT/"scripts/baseline/benchmarks/tools/einsum_extensions_v1/controls.py")])
raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
