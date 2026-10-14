import sys,shlex,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
import torch
for dims in(0,[0],(0,)):
 try:print(repr(dims),torch.tensor([True,False]).flip(dims).tolist())
 except(TypeError,RuntimeError)as e:print(repr(dims),type(e).__name__,str(e))
