import sys
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from hecate_python_env import enter_nix
sys.exit(enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/lhohy/poseidon-work/platforms/aarch64-linux/venvs/hecate-2.0.1-cpu/bin/python -B -m unittest test_staged_application_reuse.StagedContextControls application_execution.tests.test_context test_application_model_source_context test_qualification_session -v',seconds=120,environment="poseidon-cpu"))
