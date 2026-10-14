from hecate_python_env import VENV,enter_nix
import shlex
from pathlib import Path
p=Path("/home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/pytorch_input_binding_v1/smoke.py")
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(p)])
raise SystemExit(enter_nix(command,seconds=60,keep_env=(),environment="poseidon-cpu"))
