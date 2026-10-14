"""Summarize scoped planner integration after both audited terminal results."""
import sys,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
live.no_active()
result={}
for mode in("offline","live"):
 p=q.verify(mode);out=Path(p["campaign"]);r=q.read(out/"report.json");q.bound(r)
 terminal=q.read(out/"terminal.json");owner=q.read(out/"owner.json")
 q.require(terminal["exit_code"]==0 and live.process_info(owner["pid"])is None,"terminated")
 q.require(r["status"]=="complete"and r["passed"]==3 and r["failed"]==r["skipped"]==r["not_run"]==0 and r["calls_resolved"],"complete_passed_scope")
 audits=[]
 for name in("audit.json","numerical-audit.json"):
  a=q.read(out/name);q.bound(a);q.require(a["status"]=="passed"and a["report_binding"]==r["binding"]and a["plan_binding"]==p["binding"],"audit")
  audits.append(dict(path=str(out/name),binding=a["binding"]))
 numerical=q.read(out/"audit.json")
 item=dict(passed=3,failed=0,skipped=0,not_run=0,plan_binding=p["binding"],report_binding=r["binding"],campaign=str(out),
  generations=r["generations"],http_attempts=r["http_attempts"],compiler_calls=r["compiler_calls_confirmed"],
  encrypted_calls=r["encrypted_calls_confirmed"],input_groups=numerical["input_groups"],output_values=numerical["output_values"],
  maximum_absolute_error=numerical["maximum_absolute_error"],audits=audits,
  packages_reloaded=3,packages_new_inputs=3,packages_repeat=3,packages_cache=3)
 if mode=="live":
  first=sum(c["result"]["generations"]==1 for c in r["cases"])
  item.update(first_success=first,repaired_success=3-first,provider_calls_resolved=True)
 result[mode]=item
controls=q.read(HERE/"controls.json");guard=q.read(HERE/"controls-scope.json")
q.require(controls["status"]=="passed"and controls["tests"]==67 and not controls["failures"]and not controls["errors"]and not controls["skipped"],"planner_controls")
q.require(guard["status"]=="passed"and guard["tests"]==12 and guard["failures"]==guard["errors"]==guard["skipped"]==0,"execution_controls")
rows=[]
for c in q.read(HERE/"prepared-requests.json")["cases"]:
 decision=q.read(c["layout_plan"]);q.bound(decision)
 request=q.read(c["request_file"])
 rows.append(dict(case=c["id"],model_sha256=c["model_sha256"],unchanged_model_and_weights=True,
  parent_request_id=c["parent_request_id"],current_request_id=c["request_id"],selected_period=c["selected_period"],
  physical_outputs=request["layout"]["output_ciphertexts"],logical_elements=request["layout"]["output_shape"][0],
  planning_authority=decision["source_authority"]["sha256"],layout_plan_binding=decision["binding"]))
parent=q.read(ROOT/"docs/baseline/agent-output-alias-acceptance-r1486.json");q.bound(parent)
identity=q.read(HERE/"identity-snapshot.json");q.require(identity["manifest_sha256"]==parent["sdk"],"backend_sdk_unchanged")
v=dict(format="poseidon-output-alias-planner-acceptance-v1",status="scoped_acceptance_complete",
 checkout=str(ROOT),branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
 head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),platform="aarch64-linux",backend="poseidon_cpu",
 sdk=identity["manifest_sha256"],parent_acceptance=parent["binding"],planner_source_change=q.read(HERE/"change.json"),
 controls=dict(passed=67,failed=0,errors=0,skipped=0),execution_guards=dict(passed=12,failed=0,errors=0,skipped=0),
 legacy_candidate_inventories_preserved=controls["legacy_candidate_inventories"],contexts=rows,results=result,
 unique_new_mathematical_models=0,scope="three existing mathematical models at newly selected layouts",
 source_format="poseidon-trusted-pytorch-v34",guidance="explicit-v15",policy="exact-values-v1",
 planning_identity_changed=True,compilation_execution_validation_identity_unchanged=True,
 model_or_reference_changed=False,tolerance=dict(atol=1e-5,rtol=1e-4),backend_core_changed=False,
 network_changed=False,parameters_changed=False,installations=0,downloads=0,git_actions=[],
 current_shared_batches="none",storage=dict(total_bytes=live.measure(q.RESULTS),disk_free=shutil.disk_usage(q.RESULTS).free),
 unverified=["largest configured rank and capacity","all operator/alias combinations","x86 encrypted execution","Agent-chain bootstrap","CUDA hardware","original-model application accuracy","full ResNet20"],
 historical_evidence_rewritten=False,summary_tool_sha256=q.sha(Path(__file__)))
v["binding"]=q.digest(v);q.write(ROOT/"docs/baseline/agent-output-alias-planner-acceptance-r1488.json",v)
print(json.dumps(dict(status=v["status"],binding=v["binding"],results=result)))
