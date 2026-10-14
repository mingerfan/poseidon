"""Current stage identity and retained source evidence, without recapture."""
import hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
def write(path,value):
    with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2)
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
from application_backend import PoseidonBackend
from candidate_bundle import load_model_provenance,audit_provenance
from component_contract import reconstruct_request
from deepseek_provider import public_request
before=json.loads((HERE/"identity-before.json").read_text());after=PoseidonBackend().identity_manifest()
assert after["backend"]==before["backend"]
diff={}
for stage in("generation","compilation","execution","validation"):
    a,z=before[stage]["sources"],after[stage]["sources"]
    changed=sorted(n for n in set(a)|set(z)if a.get(n)!=z.get(n))
    assert changed==["execution_input_domain.py"],(stage,changed)
    diff[stage]=dict(before=before[stage]["sha256"],after=after[stage]["sha256"],changed=changed)
write(HERE/"identity-after.json",after)
oldplan=ROOT/"docs/baseline/agent-source-package-validation-plan-r1438.json"
plan=json.loads(oldplan.read_text());assert plan["sdk"]==before["manifest_sha256"]and not plan["started"]
rows=[]
for case in plan["cases"]:
    folder=Path(case["model_import"]);model=json.loads((folder/"model.json").read_text())
    value=load_model_provenance(folder,model,max_bytes=4*1024**2);audit_provenance(value,model)
    q=json.loads(Path(case["request_file"]).read_text());assert q==reconstruct_request(q)
    assert public_request(q)==json.loads(Path(case["public_request_file"]).read_text())
    assert digest(value)==case["provenance_digest"]
    rows.append(dict(case=case["id"],status="retained_source_and_reference_revalidated",
      request_id=q["request_id"],provenance_digest=digest(value)))
state=dict(format="poseidon-spatial-domain-binding-validation-v1",status="passed",
 sdk_before=before["manifest_sha256"],sdk_after=after["manifest_sha256"],stage_differences=diff,
 backend_unchanged=True,retained_source_cases=rows,retained_source_captures_relabelled=False,
 superseded_unstarted_plan=dict(path=str(oldplan),sha256=hashlib.sha256(oldplan.read_bytes()).hexdigest(),
 binding=plan["binding"],reason="Agent execution-input checker changed; new execution/validation identity required"),
 source_capture_calls=0,compiler_calls=0,encrypted_calls=0,provider_calls=0)
state["binding"]=digest(state);write(HERE/"binding-validation.json",state)
print(json.dumps({k:state[k]for k in("status","binding","sdk_before","sdk_after","backend_unchanged","retained_source_cases")}),flush=True)
