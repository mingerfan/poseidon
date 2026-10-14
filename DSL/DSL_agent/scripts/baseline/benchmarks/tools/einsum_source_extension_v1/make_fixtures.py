import pathlib,json,hashlib,os,sys,shlex
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
def main():
 if sys.argv[1:]!=["--inside"]:
  cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
  return enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu")
 import numpy as np
 assert np.__version__=="1.25.2"and pathlib.Path(sys.prefix)==VENV
 root=HERE/"fixtures";root.mkdir();rows=[]
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 for item in json.loads((HERE/"fixture-specs.json").read_text()):
  dest=root/item["case"];dest.mkdir()
  with(dest/"model.py").open("x")as f:f.write(item["source"])
  weights={}
  for name,values in item["weights"].items():
   a=np.asarray(values,dtype="<f8");p=dest/(name+".npy")
   with p.open("xb")as f:np.save(f,a,allow_pickle=False)
   weights[name]=dict(file=p.name,dtype=a.dtype.str,shape=list(a.shape),bytes=p.stat().st_size,sha256=sha(p))
  manifest=dict(format=item.get("format","poseidon-trusted-pytorch-v19"),entry="model.make",files={"model.py":sha(dest/"model.py")},
   factory=dict(args=[],kwargs={}),inputs=item["inputs"],outputs=item["outputs"],
   dtype_policy="strict-float64-v1",weights=weights,
   approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
  with(dest/"manifest.json").open("x")as f:json.dump(manifest,f,sort_keys=True)
  rows.append(dict(case=item["case"],manifest=str(dest/"manifest.json"),manifest_sha256=sha(dest/"manifest.json")))
 with(HERE/"fixtures.json").open("x")as f:json.dump(rows,f,sort_keys=True,indent=2)
 print(json.dumps(dict(fixtures=len(rows),npy_weights=sum(bool(v["weights"])for v in json.loads((HERE/"fixture-specs.json").read_text())),provider_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
