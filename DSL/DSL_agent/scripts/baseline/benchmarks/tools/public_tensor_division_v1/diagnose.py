"""One immutable-graph diagnostic, no runtime or model API."""
import json,os,sys,shlex
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[2]
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(B))
def inside():
 from test_division import module,capture
 from public_tensor_division import normalize
 graph,i,d=capture(module("operator",[.5,2.,3.]),[[2,3]])
 try:normalize(graph,i,d)
 except ValueError as e:
  tb=e.__traceback__;frame=None
  while tb is not None:
   if tb.tb_frame.f_code.co_name=="check":frame=tb.tb_frame
   tb=tb.tb_next
  assert frame is not None
  f=frame.f_locals;expected=[v for n in f["source"]for v in f["entries"].get(n["name"],[n])]
  differences=[dict(index=k,actual=a,expected=b)for k,(a,b)in enumerate(zip(f["normalized"],expected))if a!=b]
  row=dict(error=str(e),actual_length=len(f["normalized"]),expected_length=len(expected),
   first_differences=differences[:2],entries_match=set(f["entries"])==f["actual"])
  p=H/"initial-projection-diagnostic.json"
  with p.open("x")as stream:json.dump(row,stream,indent=2,sort_keys=True);stream.write("\n")
  print(json.dumps(row))
  return 0
 raise ValueError("expected initial failure")
def outer():
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(command,seconds=60,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside()if sys.argv[1:]==["--inside"]else outer())
