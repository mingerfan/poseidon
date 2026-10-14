import sys, shlex
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from hecate_python_env import VENV, enter_nix
cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B","-m","unittest","test_unified_graph_scheduling","-v"])
raise SystemExit(enter_nix('cd '+shlex.quote(str(ROOT/"scripts/baseline"))+' && '+cmd,seconds=600,environment="poseidon-cpu"))
