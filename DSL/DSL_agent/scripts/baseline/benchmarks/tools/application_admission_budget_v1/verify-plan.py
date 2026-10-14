import sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts/baseline').resolve()))
from hecate_python_env import enter_nix
sys.exit(enter_nix('PYTHONPATH="/home/lhohy/Code/Poseidon/scripts/baseline" LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" /home/lhohy/poseidon-work/platforms/aarch64-linux/venvs/hecate-2.0.1-cpu/bin/python -B scripts/baseline/benchmarks/tools/application_admission_budget_v1/offline_cpu.py',seconds=120,environment='poseidon-cpu'))
