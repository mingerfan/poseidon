from hecate_python_env import VENV,enter_nix
import shlex
from pathlib import Path
tool=Path("scripts/baseline/benchmarks/tools/structured_input_live_v1").resolve()
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/"bin/python"),"-B","-m","unittest","discover","-s",str(tool),"-p","test_*.py","-v"])
raise SystemExit(enter_nix(command,seconds=90,keep_env=(),environment="poseidon-cpu"))
