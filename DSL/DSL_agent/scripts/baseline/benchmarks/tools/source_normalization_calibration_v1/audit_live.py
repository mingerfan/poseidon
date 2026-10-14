"""Audit the terminal bounded live campaign without generating or executing."""
from pathlib import Path
import sys,shlex,json,collections,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from generation_model_origin import verify
from component_contract import reconstruct_request
from candidate_bundle import check_bundle
from application_backend import PoseidonBackend
D=q.RESULTS/"agent-normalization-source-live-r1693"
report=q.read(D/"report.json");q.bound(report)
plan=q.read(D/"live-plan.json");q.bound(plan)
live.no_active()
q.require(report["planned"]==len(plan["cases"])==6 and len(report["cases"])==6 and report["not_run"]==report["skipped"]==0,"terminal_scope")
q.require(report["plan_binding"]==plan["binding"],"same_live_plan")
q.require(all(q.sha(Path(n))==h for n,h in plan["source_files"].items()),"frozen_sources")
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==report["sdk"],"same_backend_identity")
rows=[];packages=artifacts=compiler_attempts=compiled=encrypted=0
failures=collections.Counter();usage=collections.Counter()
for c in report["cases"]:
 folder=D/c["case"];P=folder/"tasks"/c["task_id"];meta=q.read(P/"task.json");request=meta["request"]
 verify(request["generation_model_origin"],meta["model_provenance"],request)
 q.require(reconstruct_request(request)==request,"request_roundtrip")
 metrics=q.read(P/"provider-metrics.json")
 q.require(metrics["agent_calls"]==c["generations"] and metrics["service_provider"]=="deepseek","real_provider_accounting")
 for call in metrics["calls"]:
  if type(call.get("usage")) is dict:
   for key,value in call["usage"].items():
    if type(value) is int:usage[key]+=value
 for p in sorted(P.glob("validation-*.json")):
  validation=q.read(p);r=q.read(Path(validation["evidence"])/"report.json")
  q.require(r["backend"]=="poseidon_cpu" and r["tolerance"]=={"atol":1e-5,"rtol":1e-4},"pinned_backend_threshold")
  for attempt in r["attempts"]:
   compiler_attempts+=1;compiled+=bool(attempt.get("compiled"));encrypted+=bool(attempt.get("executed"))
   if attempt["status"]!="passed":failures[validation.get("failure",{}).get("layer","unknown")]+=1
   for name,want in attempt.get("artifact_hashes",{}).items():
    f=Path(validation["evidence"])/("attempt-%02d"%attempt["index"])/"output"/name
    q.require(q.sha(f)==want,"artifact_integrity");artifacts+=1
 row=dict(case=c["case"],status=c["status"],generations=c["generations"],http_attempts=c["http_attempts"],
   first_attempt_passed=c["first_attempt_passed"],request_id=c["request_id"],
   source_origin_binding=request["generation_model_origin"]["binding"])
 if c["status"]=="succeeded":
  program=P/"program";manifest=check_bundle(program)
  q.require(manifest["binding"]==c["package_binding"] and manifest["agent_generated"] is True,"generated_package")
  for name,want in manifest["files"].items():q.require(q.sha(program/name)==want,"package_integrity")
  packages+=len(manifest["files"])
  generation=q.read(P/"generation-provenance.json");q.bound(generation)
  source=(program/"candidate.py").read_text();sources=[]
  for p in P.glob("response-*.json"):
   try:value=json.loads(q.read(p)["raw"])
   except ValueError:continue
   sources.append(value.get("hecate_source"))
  q.require(source in sources and generation["candidate_sha256"]==q.sha(program/"candidate.py"),"unchanged_provider_source")
  values=0;maximum=0.
  for label in ("new","repeat"):
   reuse=q.read(folder/(label+"-execution.json"));execution=reuse["execution"];comparison=reuse["comparison"]
   q.require(execution["encrypted_execution"] is True and execution["executor_secret_key"] is False and
     execution["agent_calls"]==execution["compiler_calls"]==0 and comparison["passed"] and
     comparison["atol"]==1e-5 and comparison["rtol"]==1e-4,"real_package_reuse")
   values+=comparison["compared_values"];maximum=max(maximum,comparison["max_absolute_error"]);encrypted+=1
  row.update(package_binding=c["package_binding"],source_matches_original_response=True,
    primary_groups=4,primary_values=c["compared_values"],primary_maximum_absolute_error=c["maximum_absolute_error"],
    reuse_groups=4,reuse_values=values,reuse_maximum_absolute_error=maximum)
 else:row["failure_layer"]=(c.get("failure")or{}).get("layer")
 rows.append(row)
success=[r for r in rows if r["status"]=="succeeded"]
ledger=q.read(D/"global-call-ledger.json");q.bound(ledger)
q.require(ledger["used"]==report["used"] and compiler_attempts==report["used"]["qualification_reservations"],"final_accounting")
q.require(encrypted<=plan["budget"]["max_encrypted_calls"],"bounded_encrypted_count")
out=dict(format="poseidon-normalization-source-live-audit-v1",planned=6,passed=len(success),failed=6-len(success),skipped=0,not_run=0,
 first_attempt_passed=sum(r["first_attempt_passed"]for r in success),
 passed_after_repairs=sum(not r["first_attempt_passed"]for r in success),cases=rows,
 report_binding=report["binding"],plan_binding=plan["binding"],sdk=report["sdk"],used=report["used"],
 compiler_attempts=compiler_attempts,compiled_successfully=compiled,encrypted_calls=encrypted,
 retained_failed_attempt_layers=dict(failures),
 primary_groups=sum(r["primary_groups"]for r in success),primary_values=sum(r["primary_values"]for r in success),
 primary_maximum_absolute_error=max((r["primary_maximum_absolute_error"]for r in success),default=None),
 reuse_groups=sum(r["reuse_groups"]for r in success),reuse_values=sum(r["reuse_values"]for r in success),
 reuse_maximum_absolute_error=max((r["reuse_maximum_absolute_error"]for r in success),default=None),
 package_files_verified=packages,artifact_files_verified=artifacts,usage=dict(usage),monetary_cost_verified=False,
 manual_candidate_substitution=False,original_application_accuracy_certified=False,backend_changed=False,network_changed=False,
 new_agent_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
 result_bytes=live.measure(q.RESULTS),free_bytes=shutil.disk_usage(q.RESULTS).free)
out["binding"]=q.digest(out);q.write(HERE/"live-audit-r1694.json",out);q.write(D/"audit-r1694.json",out)
print(json.dumps({k:v for k,v in out.items()if k!="cases"}),flush=True)
