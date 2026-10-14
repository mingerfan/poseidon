from hecate_python_env import VENV,enter_nix
import shlex
from pathlib import Path
p=Path("scripts/baseline/benchmarks/tools/pytorch_input_binding_v1/capture.py").resolve()
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(p)])
raise SystemExit(enter_nix(command,seconds=300,keep_env=(),environment="poseidon-cpu"))
