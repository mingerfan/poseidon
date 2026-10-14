"""Pinned component checks only; no compiler, encryption or model API."""
import sys,json,hashlib,os,shlex,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*(["--baseline"] if "--baseline" in sys.argv else [])])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
import subprocess
results=[]
for mode in ("baseline","stage"):
    log=HERE/(mode+"-r1401.log")
    args=[str(VENV/"bin/python"),"-B",str(HERE/"observe.py"),"--inside"]
    if mode=="baseline":args.append("--baseline")
    with log.open("x")as f:
        child=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=150)
    results.append(dict(mode=mode,exit=child.returncode,log=str(log)))
    print(json.dumps(results[-1]),flush=True)
print(json.dumps(dict(phases=results,api_calls=0,compilations=0,encrypted_calls=0)))
