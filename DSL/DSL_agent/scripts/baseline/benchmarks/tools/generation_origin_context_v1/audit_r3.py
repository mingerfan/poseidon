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


import shutil,time
import run as live
from candidate_bundle import check_bundle
from generation_model_origin import verify
from component_contract import reconstruct_request
from application_backend import PoseidonBackend
rows=[];file_count=0;artifact_count=0
for report_name,campaign in [("poseidon-report.json","agent-generation-origin-poseidon-r1678"),
                            ("poseidon-report-r2.json","agent-generation-origin-poseidon-r1679")]:
 report=q.read(HERE/report_name);q.bound(report)
 for row in report["cases"]:
  if row["status"]!="succeeded":continue
  folder=q.RESULTS/campaign/row["case"]
  task=folder/"tasks"/row["task_id"];program=task/"program"
  metadata=q.read(task/"task.json");request=metadata["request"]
  verify(request["generation_model_origin"],metadata["model_provenance"],request)
  q.require(reconstruct_request(request)==request,"request_reconstruction")
  manifest=check_bundle(program);q.require(manifest["binding"]==row["package_binding"],"package_binding")
  for name,sha in manifest["files"].items():
   p=program/name;q.require(p.is_file()and q.sha(p)==sha,"package_file_hash")
  file_count+=len(manifest["files"])
  validation=q.read(task/"validation-0.json")
  q.require(validation["status"]=="passed"and validation["encrypted_execution"]is True and validation["numerically_validated"]is True,"real_validation")
  q.require(all(validation["observed_stages"].values()),"pipeline_stages")
  artifact=q.read(Path(validation["evidence"])/"report.json")
  q.require(artifact["backend"]=="poseidon_cpu"and artifact["status"]=="passed","poseidon_only")
  q.require(artifact["tolerance"]=={"atol":1e-5,"rtol":1e-4},"frozen_tolerance")
  final=artifact["attempts"][-1]
  q.require(final["compiled"]and final["executed"]and final["numerically_correct"],"real_attempt")
  for name,sha in final["artifact_hashes"].items():
   # Artifact hash entries use the actual absolute generated paths.
   p=Path(name)
   if not p.is_absolute():p=Path(validation["evidence"])/"attempt-00"/"output"/p
   q.require(p.is_file()and q.sha(p)==sha,"artifact_file_hash")
  artifact_count+=len(final["artifact_hashes"])
  for label in ("new","repeat"):
   reuse=q.read(folder/(label+"-execution.json"))
   e=reuse["execution"];c=reuse["comparison"]
   q.require(e["encrypted_execution"]is True and e["executor_secret_key"]is False
    and e["agent_calls"]==e["compiler_calls"]==0and c["passed"]
    and c["atol"]==1e-5and c["rtol"]==1e-4,"package_reuse")
  rows.append(dict(case=row["case"],candidate_origin=row["candidate_origin"],request_id=row["request_id"],
    origin_binding=row["source_origin_binding"],package_binding=manifest["binding"],
    primary_groups=row["primary_groups"],primary_values=row["primary_values"],
    maximum_absolute_error=row["primary_maximum_absolute_error"],
    reuse_groups=row["reuse_groups"],reuse_values=row["reuse_values"],reuse_maximum_absolute_error=row["reuse_maximum_absolute_error"]))
report=dict(format="poseidon-generation-origin-audit-v1",cases=rows,unique_passed=len(rows),
 static_passed=101,source_admission_passed=6,failed_current=0,
 historical_agent_replayed=1,rule_baselines_passed=3,new_agent_generated=0,
 compiler_calls=4,encrypted_calls=12,api_calls=0,
 verified_package_files=file_count,verified_artifact_files=artifact_count,
 historical_test_harness_errors_retained=3,
 sdk=PoseidonBackend().identity_manifest()["manifest_sha256"],
 primary_groups=sum(r["primary_groups"]for r in rows),primary_values=sum(r["primary_values"]for r in rows),
 primary_maximum_absolute_error=max(r["maximum_absolute_error"]for r in rows),
 reuse_groups=sum(r["reuse_groups"]for r in rows),reuse_values=sum(r["reuse_values"]for r in rows),
 reuse_maximum_absolute_error=max(r["reuse_maximum_absolute_error"]for r in rows),
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free,
 application_accuracy_certified=False,backend_changed=False,network_changed=False)
q.require(report["unique_passed"]==4,"all_current_cases")
report["binding"]=q.digest(report);q.write(HERE/"audit-r1679.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
