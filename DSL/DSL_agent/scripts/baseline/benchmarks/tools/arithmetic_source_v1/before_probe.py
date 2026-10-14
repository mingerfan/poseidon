import os,sys,shlex,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[4]/"scripts/baseline"
sys.path.insert(0,str(BASE));os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu"))
from test_pytorch_scalar_partitions import model,prepare
rows=[]
for name,body in [("operator_pow","return x**2"),("torch_pow","return torch.pow(x,4)"),("scalar_division","return x/2"),("inplace_power","x.pow_(2);return x")]:
 try:prepare(model(body),[[2]],["out"]);row=dict(case=name,status="accepted")
 except Exception as e:row=dict(case=name,status="rejected",layer="source_capture",error=type(e).__name__,reason=str(e))
 rows.append(row)
with(HERE/"before-gap-probe.json").open("x")as f:json.dump(dict(rows=rows,provider_calls=0,compiler_calls=0,encrypted_calls=0),f,sort_keys=True,indent=2)
print(json.dumps(rows))
