import hashlib,json,os,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 import numpy as np,torch
 from platform_config import require_python_packages
 require_python_packages(torch,np)
 cases=json.loads((HERE/"fixture-specs.json").read_text());rows=[]
 for name,c in cases.items():
  folder=HERE/"fixtures"/name;weights={}
  for key,values in c.get("weights",{}).items():
   array=np.array(values,dtype="<f8");fn=key.replace(".","_")+".npy";p=folder/fn
   with p.open("xb")as f:np.lib.format.write_array(f,array,version=(1,0),allow_pickle=False)
   weights[key]=dict(file=fn,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    shape=list(array.shape),dtype=array.dtype.str)
  spec=dict(format="poseidon-trusted-pytorch-v17",entry="model.make",
   files={"model.py":hashlib.sha256((folder/"model.py").read_bytes()).hexdigest()},
   factory=dict(args=[],kwargs={}),inputs=[dict(name="x",shape=c["shape"])],outputs=c["outputs"],
   weights=weights,dtype_policy="strict-float64-v1",
   approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[]))
  with(folder/"manifest.json").open("x")as f:json.dump(spec,f,sort_keys=True)
  rows.append(dict(case=name,manifest=str(folder/"manifest.json"),manifest_sha256=hashlib.sha256((folder/"manifest.json").read_bytes()).hexdigest()))
 with(HERE/"fixtures.json").open("x")as f:json.dump(rows,f,sort_keys=True)
 print(json.dumps(dict(manifests=len(rows),weights_saved=sum(bool(c.get("weights"))for c in cases.values()),
  source_format="poseidon-trusted-pytorch-v17",provider_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
