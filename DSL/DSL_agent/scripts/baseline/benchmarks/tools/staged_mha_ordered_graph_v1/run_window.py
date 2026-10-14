import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
cmds=[['run_controls.py',sys.argv[1]],['run_capture.py',sys.argv[2]]]
codes=[]
for name,out in cmds:
 r=subprocess.run([sys.executable,'-B',str(root/name),out],timeout=500)
 codes.append(r.returncode)
 print('WINDOW_STAGE',name,'exit',r.returncode,flush=True)
 if r.returncode:break
raise SystemExit(0 if codes==[0,0] else 1)
