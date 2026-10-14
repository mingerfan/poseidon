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
import shutil,collections
from generation_model_origin import verify
from component_contract import reconstruct_request
from candidate_bundle import check_bundle
from application_backend import PoseidonBackend
import run as live
D=q.RESULTS/"agent-generation-origin-live-r1680"
report=q.read(D/"report.json");q.bound(report)
plan=q.read(D/"live-plan.json");q.bound(plan)
q.require(report["passed"]==6 and report["failed"]==report["not_run"]==report["skipped"]==0,"six_terminal_successes")
q.require(report["plan_binding"]==plan["binding"],"same_live_plan")
q.require(all(q.sha(Path(n))==h for n,h in plan["source_files"].items()),"historical_frozen_files")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==report["sdk"],"same_audited_sdk")
rows=[];package_files=0;artifact_files=0;compiler_attempts=0;compiled=0;encrypted=0
failures=collections.Counter();usage=collections.Counter()
for c in report["cases"]:
 folder=D/c["case"];P=folder/"tasks"/c["task_id"];program=P/"program"
 meta=q.read(P/"task.json");request=meta["request"]
 verify(request["generation_model_origin"],meta["model_provenance"],request)
 q.require(reconstruct_request(request)==request,"request_roundtrip")
 manifest=check_bundle(program);q.require(manifest["binding"]==c["package_binding"] and manifest["agent_generated"] is True,"real_agent_bundle")
 for name,sha in manifest["files"].items():q.require(q.sha(program/name)==sha,"package_integrity")
 package_files+=len(manifest["files"])
 generation=q.read(P/"generation-provenance.json");q.bound(generation)
 source=(program/"candidate.py").read_text();raws=[]
 for p in P.glob("response-*.json"):
  raw=q.read(p)["raw"]
  try:value=json.loads(raw)
  except ValueError:continue
  raws.append(value.get("hecate_source"))
 q.require(source in raws and generation["candidate_sha256"]==q.sha(program/"candidate.py"),"original_provider_source")
 metrics=q.read(P/"provider-metrics.json")
 q.require(metrics["agent_calls"]==c["generations"] and metrics["service_provider"]=="deepseek","same_real_provider")
 for call in metrics["calls"]:
  if type(call.get("usage")) is dict:
   for key,value in call["usage"].items():
    if type(value) is int:usage[key]+=value
 for p in sorted(P.glob("validation-*.json")):
  value=q.read(p);a=q.read(Path(value["evidence"])/"report.json")
  q.require(a["backend"]=="poseidon_cpu" and a["tolerance"]=={"atol":1e-5,"rtol":1e-4},"pinned_backend_tolerance")
  for attempt in a["attempts"]:
   compiler_attempts+=1;compiled+=bool(attempt.get("compiled"));encrypted+=bool(attempt.get("executed"))
   if attempt["status"]!="passed":failures[value.get("failure",{}).get("layer","unknown")]+=1
   for name,sha in attempt.get("artifact_hashes",{}).items():
    f=Path(value["evidence"])/("attempt-%02d"%attempt["index"])/"output"/name
    q.require(q.sha(f)==sha,"artifact_hash")
    artifact_files+=1
 row=dict(case=c["case"],generations=c["generations"],http_attempts=c["http_attempts"],first_attempt_passed=c["first_attempt_passed"],
   request_id=c["request_id"],package_binding=c["package_binding"],
   primary_groups=4,primary_values=c["compared_values"],primary_maximum_absolute_error=c["maximum_absolute_error"],
   source_origin_binding=request["generation_model_origin"]["binding"],source_matches_original_response=True)
 values=0;maximum=0.
 for label in("new","repeat"):
  reuse=q.read(folder/(label+"-execution.json"));e=reuse["execution"];cmp=reuse["comparison"]
  q.require(e["encrypted_execution"] is True and e["executor_secret_key"] is False and e["agent_calls"]==e["compiler_calls"]==0
   and cmp["passed"] and cmp["atol"]==1e-5 and cmp["rtol"]==1e-4,"new_repeat_execution")
  values+=cmp["compared_values"];maximum=max(maximum,cmp["max_absolute_error"]);encrypted+=1
 row.update(reuse_groups=4,reuse_values=values,reuse_maximum_absolute_error=maximum)
 rows.append(row)
out=dict(format="poseidon-generation-origin-live-audit-v1",planned=6,passed=6,failed=0,skipped=0,not_run=0,
 first_attempt_passed=4,passed_after_repairs=2,cases=rows,report_binding=report["binding"],plan_binding=plan["binding"],
 sdk=report["sdk"],used=report["used"],compiler_attempts=compiler_attempts,compiled_successfully=compiled,
 encrypted_calls=encrypted,retained_failed_attempt_layers=dict(failures),
 primary_groups=24,primary_values=sum(r["primary_values"]for r in rows),
 primary_maximum_absolute_error=max(r["primary_maximum_absolute_error"]for r in rows),
 reuse_groups=24,reuse_values=sum(r["reuse_values"]for r in rows),
 reuse_maximum_absolute_error=max(r["reuse_maximum_absolute_error"]for r in rows),
 package_files_verified=package_files,artifact_files_verified=artifact_files,
 usage=dict(usage),monetary_cost_verified=False,manual_candidate_substitution=False,
 original_application_accuracy_certified=False,backend_changed=False,network_changed=False,
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
q.require(compiler_attempts==9 and compiled==7 and encrypted==19,"actual_execution_accounting")
out["binding"]=q.digest(out);q.write(HERE/"live-audit-r1681.json",out);q.write(D/"audit-r1681.json",out)
print(json.dumps({k:v for k,v in out.items()if k!="cases"}),flush=True)
