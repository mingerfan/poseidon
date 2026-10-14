"""Reload and execute an exported RuntimePlan without generation or recompilation."""
import json,shutil,tempfile
from pathlib import Path
import numpy as np
from hecate_python_env import WORK,digest
from poseidon_backend_identity import identity
from poseidon_validation import audit_gate,prepare_inputs,execute
from poseidon_runtime_compile import audit_entrypoint
from candidate_contract import validate_candidate,trace_payload_json
from unified_graph_prepare import prepare_case
from native_execution_slots import native_slot
from poseidon_candidate_sandbox import run
from numerical_results import compare
def copy_checked_artifacts(folder,manifest,output):
    for name,expected in manifest["files"].items():
        if not name.startswith("artifacts/"):continue
        source=folder/name
        if source.is_symlink() or digest(source)!=expected:raise ValueError("Package changed before reload")
        destination=output/Path(name).relative_to("artifacts")
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        if digest(destination)!=expected:raise ValueError("Package changed during reload")

def replay(folder,manifest):
    folder=Path(folder)
    if identity()!=manifest["dependencies"]["backend_identity"]:
        raise ValueError("Poseidon package dependency mismatch")
    request=json.loads((folder/"request.json").read_text())
    candidate=dict(schema=1,request_id=request["request_id"],hecate_source=(folder/"candidate.py").read_text())
    validate_candidate(candidate,request)
    evidence=Path(tempfile.mkdtemp(prefix="poseidon-package-replay-",dir=WORK/"results"))
    output=evidence/"output";output.mkdir()
    copy_checked_artifacts(folder,manifest,output)
    record=json.loads((output/"validation.json").read_text())
    audit_entrypoint(output,request,record.get("runtime_plan_entry"),record.get("compile_command"))
    audit_gate(output,request,record["gate"])
    level=manifest["validation"]["level"]
    report=dict(backend="poseidon_cpu",validation_level=level,bundle_binding=manifest["binding"],status="compiled",
                agent_calls=0,compiler_calls=0,compiled_artifacts_reused=True,encrypted_execution=False,
                numerically_validated=False,evidence=str(evidence),backend_identity=identity())
    if level=="numerical":
        period=request["layout"]["input_slot_period"] if "logical_inputs" in request["layout"] else None
        prepare_case(request["model"],evidence,period)
        payload=evidence/"payload.json";payload.write_text(trace_payload_json(request,candidate))
        frozen=prepare_inputs(evidence,output,request)
        private_frozen={name:digest(evidence/name) for name in ("arrays.npz","weights.npz","payload.json")}
        with native_slot(WORK/"cache/agent-native-slots"):
            execution=execute(payload,output,request,run,evidence/"execute.log")
        with np.load(evidence/"arrays.npz",allow_pickle=False) as data:
            comparison=compare(np.load(output/"decrypted.npy",allow_pickle=False),data["reference"],1e-5,1e-4)
        if not all(digest(output/k)==v for k,v in frozen.items()) or not all(digest(evidence/k)==v for k,v in private_frozen.items()) or identity()!=report["backend_identity"]:
            raise ValueError("Package execution integrity changed")
        report.update(execution=execution,comparison=comparison,encrypted_execution=True,
                      numerically_validated=comparison["passed"],status="passed" if comparison["passed"] else "numerical_failed")
    (evidence/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ("backend_identity","comparison")},sort_keys=True))
    return 0 if report["status"] in ("passed","compiled") else 1
