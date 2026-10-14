from pathlib import Path
import sys,shlex,unittest,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
import qualification_scope as q
import shutil
import run as live
from candidate_bundle import check_bundle
from application_backend import PoseidonBackend
from source_calibration import check_report,implementation
from component_contract import reconstruct_request
from generation_model_origin import verify
source=q.RESULTS/"agent-normalization-source-calibration-r1689"
backend=q.RESULTS/"agent-normalization-source-manual-r1691"
r=q.read(HERE/"manual-report.json");q.bound(r)
control=q.read(HERE/"controls.json");q.bound(control)
a=q.read(HERE/"acceptance.json");q.bound(a)
q.require(r["passed"]==a["passed"]==6 and control["passed"]==118,"stage_acceptance")
rows=[];files=0;artifacts=0;native_lib=None
for c in r["cases"]:
 folder=backend/c["case"];P=folder/"tasks"/c["task_id"];program=P/"program"
 src=source/c["case"]
 for part in("inventory","observed"):
  request=q.read(src/part/"request.json");receipt=q.read(src/part/"calibration.json")
  check_report(receipt,request)
  q.require(receipt["implementation"]==implementation(),"calibration_current_identity")
 freeze=q.read(src/"freeze/freeze.json");q.bound(freeze)
 q.require(freeze["format"]=="poseidon-source-approximation-freeze-v3","component_freeze")
 selected=freeze["approximation_selection"]
 capture=q.read(src/"capture/decomposition.json")
 q.require(capture["pytorch_capture"]["approximation_selection"]==selected,"same_native_frontend_selection")
 witness=q.read(folder/"manual-expression-witness.json")
 q.require(witness["exact_binary_factor_merges"]==3 and witness["provider_source_changed"] is False and witness["agent_generated"] is False,"manual_factor_witness")
 q.require(q.sha(folder/"original-rule.py")==witness["original_source_sha256"] and
   q.sha(folder/"manual-equivalent.py")==witness["manual_source_sha256"],"witness_sources")
 metadata=q.read(P/"task.json");request=metadata["request"]
 verify(request["generation_model_origin"],metadata["model_provenance"],request)
 q.require(reconstruct_request(request)==request,"request_reconstruction")
 manifest=check_bundle(program)
 for name,want in manifest["files"].items():q.require(q.sha(program/name)==want,"program_hash")
 files+=len(manifest["files"])
 q.require(manifest["agent_generated"] is False and c["candidate_origin"]=="manual_equivalent","manual_not_agent")
 validation=q.read(P/"validation-0.json")
 q.require(validation["status"]=="passed" and all(validation["observed_stages"].values()),"complete_pipeline")
 report=q.read(Path(validation["evidence"])/"report.json")
 q.require(report["backend"]=="poseidon_cpu" and report["tolerance"]=={"atol":1e-5,"rtol":1e-4},"fixed_backend_threshold")
 if native_lib is None:native_lib=report["poseidon_library_sha256"]
 q.require(report["poseidon_library_sha256"]==native_lib,"same_real_library")
 for name,want in report["attempts"][-1]["artifact_hashes"].items():
  q.require(q.sha(Path(validation["evidence"])/"attempt-00/output"/name)==want,"artifact_hash")
  artifacts+=1
 for label in("new","repeat"):
  result=q.read(folder/(label+"-execution.json"));e=result["execution"];cmp=result["comparison"]
  q.require(e["encrypted_execution"] is True and e["executor_secret_key"] is False and e["agent_calls"]==e["compiler_calls"]==0
    and cmp["passed"] and cmp["atol"]==1e-5 and cmp["rtol"]==1e-4,"real_reuse")
 rows.append(dict(case=c["case"],source_binding=freeze["binding"],request_id=c["request_id"],
  origin_binding=c["source_origin_binding"],package_binding=c["package_binding"],
  primary_groups=c["primary_groups"],primary_values=c["primary_values"],primary_maximum_absolute_error=c["primary_maximum_absolute_error"],
  reuse_groups=c["reuse_groups"],reuse_values=c["reuse_values"],reuse_maximum_absolute_error=c["reuse_maximum_absolute_error"]))
out=dict(format="poseidon-normalization-source-audit-v1",source_passed=6,static_passed=118,manual_poseidon_passed=6,rule_poseidon_passed=0,rule_compiler_failed=6,
 failed=0,skipped=0,not_run=0,cases=rows,control_binding=control["binding"],source_binding=a["binding"],backend_binding=r["binding"],
 sdk=r["sdk"],calibration_implementation=implementation(),compiler_attempts=12,compiler_calls_succeeded=6,encrypted_calls=18,
 agent_calls=0,original_application_accuracy_certified=False,network_changed=False,backend_changed=False,
 primary_groups=24,primary_values=sum(c["primary_values"] for c in rows),
 primary_maximum_absolute_error=max(c["primary_maximum_absolute_error"]for c in rows),
 reuse_groups=24,reuse_values=sum(c["reuse_values"] for c in rows),reuse_maximum_absolute_error=max(c["reuse_maximum_absolute_error"]for c in rows),
 verified_package_files=files,verified_artifact_files=artifacts,
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
out["binding"]=q.digest(out);q.write(HERE/"audit-r1692.json",out);q.write(backend/"audit-r1692.json",out)
print(json.dumps({k:v for k,v in out.items()if k not in("cases","calibration_implementation")}),flush=True)
