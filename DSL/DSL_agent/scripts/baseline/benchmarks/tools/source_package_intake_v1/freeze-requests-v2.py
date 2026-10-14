"""Prepare finite source-package request identities without execution or generation."""
import hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
def write(path,value):
    with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=120,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
from candidate_bundle import load_model_provenance,check_provenance
from unified_graph_contract import validate_request
from component_contract import prepare_task,reconstruct_request
from pytorch_model_intake import preparation_options
from operator_arithmetic_graph import preparation_options as graph_options
from deepseek_provider import public_request
from application_backend import PoseidonBackend
from application_capabilities import input_sources
before=json.loads((HERE/"identity-after-controls.json").read_text())
sdk=PoseidonBackend().identity_manifest()
assert sdk==before
plan=json.loads((HERE/"capture-plan-v4.json").read_text());root=HERE/"prepared-requests";root.mkdir()
rows=[]
for c in plan["cases"]:
    folder=Path(plan["campaign"])/c["name"];model=json.loads((folder/"model.json").read_text())
    provenance=load_model_provenance(folder,model,max_bytes=4*1024**2)
    opts=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4,capacity_packing_version="capacity-public-tensors-v9")
    opts=preparation_options(provenance,model,opts)
    opts=graph_options(provenance["original"],model,opts)
    q=prepare_task(model,opts);validate_request(q);assert reconstruct_request(q)==q
    check_provenance(provenance,q["model"])
    case=root/c["name"];case.mkdir()
    write(case/"request.json",q);public=public_request(q);write(case/"public-request.json",public)
    rows.append(dict(id=c["name"],source_format=provenance["trusted_pytorch"]["manifest"]["format"],
     source_manifest=c["manifest"],source_manifest_sha256=c["manifest_sha256"],model_import=str(folder),
     model_sha256=hashlib.sha256((folder/"model.json").read_bytes()).hexdigest(),
     request_file=str(case/"request.json"),request_file_sha256=hashlib.sha256((case/"request.json").read_bytes()).hexdigest(),
     public_request_file=str(case/"public-request.json"),public_request_file_sha256=hashlib.sha256((case/"public-request.json").read_bytes()).hexdigest(),
     request_id=q["request_id"],generation_guidance=q["generation_guidance"]["version"],
     provenance_digest=digest(provenance),validation_level="numerical"))
report=dict(format="poseidon-source-package-prepared-requests-v1",sdk=sdk["manifest_sha256"],capture_plan_binding=plan["binding"],
 cases=rows,prepared=len(rows),distinct_requests=len({c["request_id"]for c in rows}),provider_calls=0,
 source_capture_calls=0,compiler_calls=0,encrypted_calls=0,historical_records_rewritten=False)
report["binding"]=digest(report);write(HERE/"prepared-requests.json",report)
write(HERE/"capabilities-after.json",input_sources())
print(json.dumps({k:report[k]for k in("binding","sdk","prepared","distinct_requests","provider_calls")}),flush=True)
