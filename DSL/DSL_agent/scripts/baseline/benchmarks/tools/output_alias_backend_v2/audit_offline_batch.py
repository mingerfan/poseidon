from pathlib import Path
import subprocess,sys
T=Path(__file__).resolve().parent
for tool,args in (("audit.py",["--inside","offline"]),("numerical_audit_v3.py",["--inside","offline"]),("package_cli_replay.py",[])):
 code=subprocess.call([sys.executable,"-B",str(T/tool),*args])
 if code:raise SystemExit(code)
