"""Freeze the existing two MHA cases under the accepted current sources."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from application_identity import current_identity_manifest
from benchmark_graph import canonical,digest
from component_contract import reconstruct_request
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_path=ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1317.json"
old=json.loads(old_path.read_text());value=copy.deepcopy(old)
identity=current_identity_manifest()
accepted=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-staged-context-package-acceptance-r1329")
accept=json.loads((accepted/"report.json").read_text())
assert accept["status"]=="passed"and identity==json.loads((accepted/"identity.json").read_text())
assert old["source_stage_identity"]["backend"]==identity["backend"]
for case in old["cases"]:
    p=Path(case["request_file"]);assert sha(p)==case["request_file_sha256"]
    request=json.loads(p.read_text());assert reconstruct_request(request)==request
    assert request["request_id"]==case["request_id"]
    for table in ("source_files","declared_weight_files"):
        assert all(sha(p)==h for p,h in case[table].items())
value["source_stage_identity"]=identity
value["source_capture_implementation"]={n:sha(BASE/n)for n in old["source_capture_implementation"]}
files=dict(old["source_implementation_files"])
names=["scripts/baseline/application_execution/context.py",
       "scripts/baseline/application_execution/context_identity.py",
       "scripts/baseline/application_execution/backend.py",
       "scripts/baseline/application_reuse.py",
       "scripts/baseline/application_planning/backend_parameters.py"]
for n in names:files[n]=None
value["source_implementation_files"]={n:sha(ROOT/n)for n in files}
phase=dict(previous_phase_binding=old["phase_binding"],worker_policy="staged-sdk-v1",
           source_identity_sha256=identity["manifest_sha256"],
           accepted_binding=accept["binding"],accepted_report_sha256=sha(accepted/"report.json"))
value["phase_binding"]=digest(phase)
value["source_rebound"]=dict(parent_binding=old["binding"],parent_sha256=sha(old_path),
 worker_identity_policy="staged-sdk-v1",phase_material=phase,
 new_implementation_files=sorted(set(files)-set(old["source_implementation_files"])),
 representative_acceptance="rules; existing packages, 0 recompiles, 4 real Poseidon executions; not new Agent generation",
 previous_scope_unapproved_and_unexecuted=True)
value.pop("binding");value["binding"]=digest(value)
dest=ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1330.json"
assert not dest.exists();dest.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
scope=BASE/"benchmarks/tools/mha_feedback_live_v4/scope.py"
s=scope.read_text();assert "NEW_BINDING_PLACEHOLDER"in s and "NEW_SHA_PLACEHOLDER"in s
s=s.replace("NEW_BINDING_PLACEHOLDER",value["binding"]).replace("NEW_SHA_PLACEHOLDER",sha(dest))
compile(s,str(scope),"exec");scope.write_text(s)
sys.path.insert(0,str(scope.parent));import scope as frozen
manifest=dict(format="poseidon-mha-feedback-live-runner-v1",proposal_binding=value["binding"],
 proposal_sha256=sha(dest),tool_hashes=frozen.tools())
manifest["binding"]=digest(manifest)
target=scope.with_name("runner-manifest.json");assert not target.exists()
target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
view=frozen.verify()
print(json.dumps(dict(proposal=str(dest),binding=value["binding"],cases=[c["id"]for c in view["cases"]],
 stage_sources={s:len(identity[s]["sources"])for s in ("generation","compilation","execution","validation")},
 new_sources=value["source_rebound"]["new_implementation_files"],approved=False,provider_calls=0)))
