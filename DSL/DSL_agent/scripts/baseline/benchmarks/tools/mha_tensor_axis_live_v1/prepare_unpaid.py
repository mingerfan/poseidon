"""Unpaid source capture and exact new request preflight. Never load credentials or run generation."""
import json,os,shlex,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def main():
 if sys.argv[1:]!=["--inside"]:
  from hecate_python_env import VENV,enter_nix
  command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
  return enter_nix(command,seconds=90,keep_env=(),environment="poseidon-cpu")
 from scope import verify,read,PROPOSAL,RESULTS,PROFILE,OPTIONS,require,digest,sha,canonical
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_model_source import FORMAT
 from unified_capacity_constants import request_version
 from deepseek_provider import public_request
 from application_identity import current_identity_manifest
 p=verify();require(current_identity_manifest()==p["source_stage_identity"],"frozen_application_identity")
 root=RESULTS/"agent-mha-tensor-axis-source-preflight-v1"
 require(not root.exists(),"preserve_preflight_evidence");root.mkdir(mode=0o700)
 app=application(WorkerContext.capture(root/"tasks"))
 rows=[];start=time.monotonic()
 for c in p["cases"]:
  frozen=read(c["full_request_file"])
  opts=dict(OPTIONS);opts["capacity_packing_version"]=request_version(frozen)
  source=dict(format=FORMAT,kind="trusted_pytorch",manifest=c["source_manifest"],trust_local_code=True)
  task=app.submit_source(source,resource_profile=PROFILE,options=opts,
   limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
  state=app.prepare_source(task);require(state["status"]=="queued","frozen_source_prepare")
  meta=app._checked_task_metadata(task);require(meta["request"]==frozen,"frozen_mathematical_request")
  require(meta["identity"]==p["application_stage_identity"],"frozen_identity")
  size=len(canonical(public_request(meta["request"])))
  require(size==c["public_context_bytes"] and size<=131072,"frozen_public_context")
  rows.append(dict(case=c["id"],task_id=task,request_id=frozen["request_id"],public_context_bytes=size,
   source_capture=True,mathematical_request_unchanged=True,paid_calls=0,compiled=False,encrypted_execution=False))
 report=dict(format="poseidon-mha-tensor-axis-unpaid-source-preflight-v1",proposal_binding=p["binding"],
  runner_manifest_sha256=sha(HERE/"runner-manifest.json"),status="passed",planned=2,passed=2,
  rows=rows,source_stage_identity_sha256=p["source_stage_identity"]["manifest_sha256"],seconds=time.monotonic()-start,
  generations=0,http_attempts=0,production_authorization_created=False,compiled=False,encrypted_execution=False)
 report["binding"]=digest(report);(root/"report.json").write_bytes(canonical(report))
 print(json.dumps(dict(status=report["status"],passed=2,rows=rows,seconds=report["seconds"],paid_calls=0,report=str(root/"report.json"))))
 return 0
if __name__=="__main__":raise SystemExit(main())
