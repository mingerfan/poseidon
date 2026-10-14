"""Finalize current prepared-source evidence and reviewed live tools; no SDK/API."""
import copy, os
from pathlib import Path
from scope import *

INITIAL = PREPARATION / "proposal-initial.json"

def candidate():
    require(sha(regular(INITIAL))==INITIAL_SHA,"fixed_initial_preparation")
    value=copy.deepcopy(read(INITIAL));check_binding(value)
    # The captured report and every per-case binding remain their original bytes.
    value["tool_hashes"]=tools()
    value["preparation"]["producer_tool_hashes"]=read(PREPARATION/"freeze.json")["tool_hashes"]
    origins={}
    for name in CASES:
        attempt=(RESULTS/"agent-v12-source-rebind-preparation-v2-attempt2") if name=="linear" else PREPARATION
        frozen=read(attempt/"freeze.json");snapshots={}
        for rel,want in frozen["tool_hashes"].items():
            original=ROOT/rel
            path=(attempt/"tool-snapshot"/original.name) if original.parent==HERE else original
            require(sha(regular(path))==want,"producer_snapshot_changed")
            snapshots[str(path)]=want
        origins[name]=dict(type="reuse_completed_capture" if name=="linear" else "new_capture",
            case_root=str(CASE_ROOTS[name]),attempt_root=str(attempt),freeze_sha256=sha(attempt/"freeze.json"),
            producer_tool_hashes=frozen["tool_hashes"],tool_snapshot_hashes=snapshots)
    value["preparation"]["capture_origins"]=origins
    value=rebound(value)
    verify(value)
    return value

def main():
    require(not PROPOSAL.exists(),"single_final_proposal_only")
    value=candidate()
    with PROPOSAL.open("xb") as stream:
        os.chmod(PROPOSAL,0o600);stream.write(canonical(value))
    verify()
    print(json.dumps(dict(status="prepared_awaiting_paid_approval",proposal=str(PROPOSAL),
        sha256=sha(PROPOSAL),binding=value["binding"],source_files=706,protected_files=654,
        cases=list(CASES),generations_max=12,http_attempts_max=48,provider_calls=0,
        compiler_calls=0,encrypted_executions=0),sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
