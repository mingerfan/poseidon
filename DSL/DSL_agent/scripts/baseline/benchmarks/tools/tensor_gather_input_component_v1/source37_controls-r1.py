import os,sys,subprocess,shlex,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2]
sys.path.insert(0,str(B))
from hecate_python_env import VENV,enter_nix
modules=["test_tensor_gather_source","test_pytorch_source_contract","test_pytorch_model_intake",
"test_provenance_json_transport","test_model_source_provenance","test_application_provenance_transport",
"test_logical_view_normalization","test_source_cli_package_policy"]
if sys.argv[1:]!=["--inside"]:
 from application_backend import PoseidonBackend
 identity=PoseidonBackend().identity_manifest()
 (HERE/"identity-before-source37-controls.json").write_text(json.dumps(identity,sort_keys=True,indent=2)+"\n")
 cmd='PYTHONPATH=scripts/baseline LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
started=time.monotonic()
mods=[m for m in modules if (B/(m+".py")).exists()]
with(HERE/"source37-controls.log").open("x")as f:
 code=subprocess.run([sys.executable,"-B","-m","unittest",*mods,"-v"],stdout=f,stderr=subprocess.STDOUT,timeout=240).returncode
text=(HERE/"source37-controls.log").read_text()
print(text[-4200:])
record=dict(format="poseidon-contract-scope-controls-v1",returncode=code,status="passed"if code==0 else"failed",
 interpreter=sys.version,modules=mods,seconds=time.monotonic()-started,api_calls=0,compiler_calls=0,encrypted_calls=0,
 log_sha256=hashlib.sha256(text.encode()).hexdigest())
record["binding"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
with(HERE/"source37-controls.json").open("x")as f:json.dump(record,f,sort_keys=True,indent=2)
raise SystemExit(code)
