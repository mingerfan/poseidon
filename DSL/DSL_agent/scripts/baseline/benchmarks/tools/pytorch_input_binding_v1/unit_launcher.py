from hecate_python_env import VENV,enter_nix
import shlex
command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts/baseline '+shlex.join([str(VENV/"bin/python"),"-B","-m","unittest","test_pytorch_input_binding","-v"])
raise SystemExit(enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu"))
