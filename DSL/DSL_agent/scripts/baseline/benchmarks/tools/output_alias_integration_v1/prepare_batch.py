from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
for tool in ("controls_scope.py","prepare_requests.py"):
 code=subprocess.call([sys.executable,"-B",str(HERE/tool),"--inside"])
 if code:raise SystemExit(code)
