"""Read and validate actual captured source provenance; no compiler or Provider."""
import hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
if "--inside" not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=90,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
from candidate_bundle import load_model_provenance,audit_provenance
from candidate_contract import canonical
from pytorch_source_package_policy import PACKAGE_FORMAT
plan=json.loads((HERE/"capture-plan-v4.json").read_text())
run=Path(plan["campaign"]);rows=[]
for case in plan["cases"]:
    folder=run/case["name"];model=json.loads((folder/"model.json").read_text())
    value=load_model_provenance(folder,model,max_bytes=4*1024**2)
    audit_provenance(value,model)
    assert value["trusted_pytorch"]["format"]==PACKAGE_FORMAT
    raw=canonical(value)
    spec=json.loads(Path(case["manifest"]).read_text())
    assert value["trusted_pytorch"]["manifest"]==spec
    assert all(hashlib.sha256(text.encode("utf-8")).hexdigest()==spec["files"][name] for name,text in value["trusted_pytorch"]["files"].items())
    rows.append(dict(case=case["name"],status="passed",provenance_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),sources=len(spec["files"]),weight_files=len(spec["weights"])))
report=dict(format="poseidon-source-package-provenance-controls-v1",sdk=plan["sdk"],capture_plan_binding=plan["binding"],cases=rows,passed=len(rows),failed=0,skipped=0,source_capture_calls=0,compiler_calls=0,encrypted_calls=0,provider_calls=0)
report["binding"]=digest(report)
with(HERE/"provenance-controls.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({key:report[key]for key in("binding","passed","failed","skipped","cases")}),flush=True)
