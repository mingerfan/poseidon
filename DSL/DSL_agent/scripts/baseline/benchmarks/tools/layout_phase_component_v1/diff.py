from pathlib import Path
import sys,json,shlex
H=Path(__file__).resolve().parent;B=H.parents[2];sys.path.insert(0,str(B))
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=60,environment="poseidon-cpu"))
from component_contract import prepare_task,request_options
p=B/"benchmarks/tools/provenance_transport_live_v2/prepared/wide_tail_linear/request.json"
old=json.loads(p.read_bytes());opts=request_options(old);opts["generation_guidance"]="explicit-v18";new=prepare_task(old["model"],opts)
def differences(a,b,path):
 if isinstance(a,dict)and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:print("KEY",path+"."+k)
   else:differences(a[k],b[k],path+"."+k)
 elif a!=b:print("DIFF",path,str(a)[:100],str(b)[:100])
differences(old["constant_origins"],new["constant_origins"],"origins")
print("public_constants_equal",old["public_constants"]==new["public_constants"])
