"""Read-only recovery of completed source captures after fixture report path error."""
import sys,json,hashlib,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
if sys.argv[1:]!=["--inside"]:
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from candidate_bundle import load_model_provenance
from benchmark_graph import digest
from pytorch_ordered_contract import source_identity
root=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-tensor-gather-source-r1575")
identity=source_identity();rows=[]
for name in ("vector","rows","middle"):
 folder=root/name;model=json.loads((folder/"model.json").read_text())
 provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
 ref=provenance["reference_report"];capture=ref["capture_metadata"]["public_value_expressions"]
 flow=capture["tensor_gather_flow"]
 assert capture["source_intake_version"]=="poseidon-trusted-pytorch-v37"
 assert len(flow["entries"])==1 and capture["logical_view_reference"]["groups"]==16
 rows.append(dict(case=name,status="passed",model_sha256=digest(model),
  source_version=capture["source_intake_version"],normalization_binding=flow["binding"],
  original_input_shapes=[x["shape"]for x in model["inputs"]],
  reference_groups=16,provenance_reloaded=True))
 assert source_identity()==identity
result=dict(format="poseidon-public-tensor-gather-source-acceptance-v1",passed=3,failed=0,skipped=0,
 cases=rows,source_identity=identity,source_outputs_reused_without_recapture=True,
 old_report_path_failure_retained=True,source_captures=3,
 api_calls=0,compiler_calls=0,encrypted_calls=0,semantic_reference_groups=48)
result["binding"]=digest(result)
with(HERE/"source-acceptance.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2)
print(json.dumps({k:v for k,v in result.items()if k not in("cases","source_identity")}))
