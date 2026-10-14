"""One bounded old package-entry replay, with current identity and no generation."""
import json,sys,time,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle
from poseidon_package import replay
live.no_active();plan=q.verify("offline");out=Path(plan["campaign"]);r=q.read(out/"report.json");q.bound(r)
q.require(r["passed"]==12 and r["failed"]==r["not_run"]==0 and q.read(out/"terminal.json")["exit_code"]==0,"offline_complete")
q.require(PoseidonBackend().identity_manifest()==q.read(HERE/"identity-snapshot.json"),"sdk")
row=next(c for c in r["cases"]if c["case"]=="structure_2")
pkg=Path(row["package"]);bundle=check_bundle(pkg)
before=live.measure(q.RESULTS);q.require(before+128*1024**2<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
scope=dict(format="poseidon-output-alias-package-cli-scope-v1",authorization="human_explicit_current_plan_test_approval",
sdk=plan["sdk"],offline_plan_binding=plan["binding"],package=str(pkg),package_binding=bundle["binding"],
case="structure_2",max_encrypted_calls=1,max_compiler_calls=0,max_provider_calls=0,wall_seconds=240,
new_artifact_bytes=128*1024**2,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3,
models_changed=False,reference_changed=False,tolerance=dict(atol=1e-5,rtol=1e-4))
scope["binding"]=q.digest(scope);q.write(HERE/"package-cli-plan.json",scope)
started=time.time();existing=set(q.RESULTS.glob("poseidon-package-replay-*"))
code=replay(pkg,bundle)
new=set(q.RESULTS.glob("poseidon-package-replay-*"))-existing
q.require(len(new)==1,"one_replay_evidence");folder=new.pop();report=q.read(folder/"report.json")
q.require(code==0 and report["status"]=="passed"and report["encrypted_execution"]is True and report["numerically_validated"]is True,"replay_pass")
q.require(report["agent_calls"]==report["compiler_calls"]==0 and check_bundle(pkg)==bundle,"zero_generation_compilation_immutable_package")
q.require(time.time()-started<=240 and live.measure(folder)<=128*1024**2 and live.measure(q.RESULTS)<=48*1024**3 and shutil.disk_usage(q.RESULTS).free>=4*1024**3,"final_budget")
v=dict(format="poseidon-output-alias-package-cli-acceptance-v1",status="passed",plan_binding=scope["binding"],
sdk=scope["sdk"],package_binding=bundle["binding"],evidence=str(folder),evidence_report_sha256=q.sha(folder/"report.json"),
compiler_calls=0,provider_calls=0,encrypted_calls=1,source="deterministic_rule_package",comparison=report["comparison"],
seconds=time.time()-started,bytes=live.measure(folder))
v["binding"]=q.digest(v);q.write(HERE/"package-cli-acceptance.json",v)
print(json.dumps({k:v[k]for k in("status","binding","encrypted_calls","compiler_calls","provider_calls")}))
