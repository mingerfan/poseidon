"""Ordered offline stages; stop before backend if controls/intake fail."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
results=Path(sys.argv[1])
commands=[("run_controls.py",[str(results/"agent-mha-ordered-controls-v3")],180),
          ("run_capture.py",[str(results/"agent-mha-ordered-capture-v2")],500),
          ("backend_acceptance.py",[str(results/"agent-mha-ordered-backend-v1"),str(results/"agent-mha-ordered-capture-v2")],1250)]
for name,args,seconds in commands:
 result=subprocess.run([sys.executable,"-B",str(root/name),*args],timeout=seconds)
 print("WINDOW_STAGE",name,"exit",result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
