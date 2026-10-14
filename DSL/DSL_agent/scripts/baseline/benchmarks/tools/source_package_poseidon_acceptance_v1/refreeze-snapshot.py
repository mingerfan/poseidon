import sys,shlex,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from candidate_bundle import load_model_provenance,check_provenance
from component_contract import prepare_task,reconstruct_request
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from deepseek_provider import public_request
from benchmark_graph import digest
old=json.loads((HERE/"identity.json").read_text());sdk=PoseidonBackend().identity_manifest()
assert old["backend"]==sdk["backend"]
deltas={}
for stage in("generation","compilation","execution","validation"):
 a=old[stage]["sources"];b=sdk[stage]["sources"];deltas[stage]=[n for n in set(a)|set(b)if a.get(n)!=b.get(n)]
 assert deltas[stage]==["application_model_source.py"]
prepared=json.loads((HERE/"prepared-requests.json").read_text())
rows=[]
for c in prepared["cases"]:
 q=json.loads(Path(c["request_file"]).read_text());folder=Path(c["model_import"])
 model=json.loads((folder/"model.json").read_text())
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2);check_provenance(provenance,q["model"])
 options=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",
  chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
 options=preparation_options(provenance,model,options);options=graph_options(provenance["original"],model,options)
 assert prepare_task(model,options)==q and reconstruct_request(q)==q
 assert public_request(q)==json.loads(Path(c["public_request_file"]).read_text())
 rows.append(dict(case=c["id"],request_id=c["request_id"],provenance_verified_current=True,
  model_weights_and_reference_unchanged=True,source_capture_repeated=False))
r=dict(format="poseidon-source-package-snapshot-rebinding-v1",old_sdk=old["manifest_sha256"],sdk=sdk["manifest_sha256"],
 deltas=deltas,cases=rows,provider_calls=0,compiler_calls=0,encrypted_calls=0,source_capture_calls=0)
r["binding"]=digest(r)
for name,v in(("identity-snapshot-final.json",sdk),("snapshot-rebinding.json",r)):
 with(HERE/name).open("x")as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
print(json.dumps({k:r[k]for k in("binding","old_sdk","sdk","deltas")}))
