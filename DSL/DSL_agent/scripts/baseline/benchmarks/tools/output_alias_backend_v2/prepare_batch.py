from pathlib import Path
import subprocess,sys
T=Path(__file__).resolve().parent
for tool in ("controls.py","controls_scope.py","freeze_qualification.py"):
 code=subprocess.call([sys.executable,"-B",str(T/tool),"--inside"])
 if code:raise SystemExit(code)
