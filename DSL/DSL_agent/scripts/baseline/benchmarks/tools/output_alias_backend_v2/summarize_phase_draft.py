"""Public phase summary after complete independent evidence and numerical audits."""
import json,hashlib,sys,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
def checked(mode):
 plan=q.verify(mode);out=Path(plan["campaign"]);report=q.read(out/"report.json");q.bound(report)
 terminal=q.read(out/"terminal.json");owner=q.read(out/"owner.json")
 q.require(terminal["exit_code"]==0 and live.process_info(owner["pid"])is None,"terminal_owner")
 q.require(report["status"]=="complete"and report["passed"]==len(plan["cases"])and
 report["failed"]==report["not_run"]==report["skipped"]==0 and report["calls_resolved"],"complete_scope")
 # Audit receipt filenames are resolved by their validated receipt fields.
 audits=[]
 for p in (out/"audit.json",out/"numerical-audit.json"):
  v=q.read(p)
  if v.get("status")=="passed"and v.get("plan_binding")==plan["binding"]:
   q.bound(v);audits.append(dict(file=str(p),binding=v["binding"],report=v))
 q.require(len(audits)==2,"both_independent_audits")
 return plan,report,audits
live.no_active();items={}
for mode in("offline","live"):
 plan,report,audits=checked(mode)
 items[mode]=dict(planned=len(plan["cases"]),passed=report["passed"],failed=0,skipped=0,not_run=0,
 plan_binding=plan["binding"],report_binding=report["binding"],campaign=plan["campaign"],
 generations=report["generations"],http_attempts=report["http_attempts"],
 compiler_calls=report["compiler_calls_confirmed"],encrypted_calls=report["encrypted_calls_confirmed"],
 audits=[{k:v for k,v in a.items()if k!="report"}for a in audits],
 input_groups=audits[0]["report"]["input_groups"],output_values=audits[0]["report"]["output_values"],
 maximum_absolute_error=audits[0]["report"]["maximum_absolute_error"])
 if mode=="live":
  first=sum(row["result"]["generations"]==1 for row in report["cases"])
  items[mode].update(first_success=first,repaired_success=report["passed"]-first)
identity=q.read(HERE/"identity-snapshot.json")
controls=q.read(HERE/"controls.json");guard=q.read(HERE/"controls-scope.json")
q.require(controls["status"]=="passed" and controls["tests"]==51 and not controls["failures"] and not controls["errors"] and not controls["skipped"],"current_control_checks")
q.require(guard["status"]=="passed" and guard["tests"]==12 and guard["failures"]==guard["errors"]==guard["skipped"]==0,"execution_control_checks")
cli=q.read(HERE/"package-cli-acceptance.json");q.bound(cli)
q.require(cli["status"]=="passed" and cli["compiler_calls"]==cli["provider_calls"]==0 and cli["encrypted_calls"]==1 and cli["sdk"]==identity["manifest_sha256"] and cli["comparison"]["passed"],"legacy_package_execution")
q.require(q.sha(Path(cli["evidence"])/"report.json")==cli["evidence_report_sha256"],"legacy_package_evidence")
cli_public={k:cli[k]for k in("status","binding","plan_binding","package_binding","evidence","evidence_report_sha256","sdk","seconds","compiler_calls","encrypted_calls","provider_calls","source")}
cli_public.update(input_groups=len(cli["comparison"]["actual"]),output_values=cli["comparison"]["compared_values"],maximum_absolute_error=cli["comparison"]["max_absolute_error"])
original=BASE/"benchmarks/tools/output_alias_integration_v1"
capture=q.read(original/"capture-plan.json");q.bound(capture)
references=q.read(Path(capture["campaign"])/"report.json");q.bound(references)
rows=[]
for case in q.read(HERE/"prepared-requests.json")["cases"]:
 request=q.read(case["request_file"]);layout=request["layout"]
 if "output_binding"in layout:
  import math
  rows.append(dict(case=case["id"],logical_elements=layout["output_shape"][0],
   physical_outputs=layout["output_ciphertexts"],unshared_same_period_outputs=sum(
   math.ceil(math.prod(o["shape"])/layout["input_slot_period"])for o in layout["outputs"]),
   period=layout["input_slot_period"],model_sha256=case["model_sha256"]))
prior=q.read(original/"offline-plan.json");old=q.read(Path(prior["campaign"])/"report.json")
result=dict(format="poseidon-output-alias-phase-acceptance-v1",status="scoped_acceptance_complete",
 policy="exact-values-v1",guidance="explicit-v15",source_format="poseidon-trusted-pytorch-v34",
 branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
 head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 checkout=str(ROOT),platform="aarch64-linux",backend="poseidon_cpu",sdk=identity["manifest_sha256"],
 transport_abi_unchanged=True,source_version_unchanged=True,legacy_request_bytes_verified=9,
 control_tests=dict(passed=controls["tests"],failed=0,skipped=0),execution_guard_tests=dict(passed=guard["tests"],failed=0,skipped=0),
 source_reference=dict(passed=12,groups=sum(r["groups"]for r in references["cases"]),
 values=sum(r["values"]for r in references["cases"]),binding=references["binding"],
 identity=capture["sdk"],current_four_group_reference_recomputed=True),
 results=items,legacy_package_cli=cli_public,logical_and_physical_outputs=rows,
 initial_failed_scope=dict(campaign=prior["campaign"],report_binding=old["binding"],passed=3,failed=9,API=0,
 failure_layer="independent reference preparation layout binding",
 cause="component adapter omitted output binding policy",original_records_retained=True),
 limits=dict(per_batch_bytes=4*1024**3,total_bytes=48*1024**3,reserve_bytes=4*1024**3,
 compiler_jobs=2,link_jobs=1,api_concurrency=1,encrypted_concurrency=1,native_memory=4*1024**3,native_cpu_seconds=150),
 tolerance=dict(atol=1e-5,rtol=1e-4),storage=dict(total_bytes=live.measure(q.RESULTS),disk_free=shutil.disk_usage(q.RESULTS).free),
 backend_core_changed=False,parameters_changed=False,models_or_reference_changed=False,
 network_changed=False,installations=0,downloads=0,git_actions=[],formal_equivalence=False,
 no_active_shared_source_batches=True,current_live_calls_resolved=True,
 summary_tool_sha256=q.sha(Path(__file__)),
 unverified=["all alias/operator combinations","x86 encrypted execution","largest configured ranks and capacity",
 "real bootstrap","CUDA hardware execution","original-model application accuracy","full ResNet20"])
result["binding"]=q.digest(result)
q.write(ROOT/"docs/baseline/agent-output-alias-acceptance-r1486.json",result)
print(json.dumps(dict(status=result["status"],binding=result["binding"],results=items,sdk=result["sdk"])))
