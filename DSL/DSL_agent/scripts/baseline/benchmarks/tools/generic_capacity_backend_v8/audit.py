"""Read-only compiled, encrypted and reusable-package evidence audit."""
import sys,json,shlex,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as scope
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 mode=sys.argv[-1]
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",mode])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
mode=sys.argv[-1];plan=scope.verify(mode);out=Path(plan["campaign"]);report=scope.read(out/"report.json")
scope.bound(report);terminal=scope.read(out/"terminal.json")
assert terminal["exit_code"]==0 and report["status"]=="complete"and report["passed"]==len(plan["cases"])
assert report["failed"]==report["not_run"]==report["skipped"]==0 and report["calls_resolved"]
assert report["sdk"]==plan["sdk"]
from application_backend import PoseidonBackend
from candidate_bundle import check_bundle,check_execution_dependencies
assert PoseidonBackend().identity_manifest()==scope.read(HERE/"identity-snapshot.json")
stats=[];groups=values=0;maximum=0
for row in report["cases"]:
 name=row["case"];case=next(c for c in plan["cases"]if c["id"]==name)
 package=Path(row["package"]);manifest,q=check_execution_dependencies(package)
 assert check_bundle(package)==manifest and q==scope.read(case["request_file"])
 assert manifest["agent_generated"]is(mode=="live")
 assert manifest["validation"]["level"]=="numerical"
 artifacts=manifest["files"]
 required=[any(n.endswith(suffix)for n in artifacts)for suffix in("earth.mlir","ckks.mlir","runtime-plan.json","operator-spec.json")]
 assert all(required)and any("/bound/bundle/manifest.json"in n for n in artifacts)
 task_dir=package.parent
 qualifications=[scope.read(p)for p in sorted(task_dir.glob("validation-*.json"))]
 qualification=next(v for v in qualifications if v["status"]=="passed" and v.get("encrypted_execution") is True)
 evidence=scope.read(Path(qualification["evidence"])/"report.json")
 assert evidence["backend"]=="poseidon_cpu"and evidence["poseidon_gpu_validated"]is False
 assert evidence["tolerance"]==dict(atol=1e-5,rtol=1e-4)
 comparisons=[]
 for validation in sorted(task_dir.glob("validation-*.json")):
  v=scope.read(validation)
  if v["status"]=="passed":
   ev=scope.read(Path(v["evidence"])/"report.json")
   for a in ev["attempts"]:
    if a.get("numerically_correct"):
     assert all(a.get(k)is True for k in("compiled","executed","numerically_correct"))
     gate=a["artifact_gate"]
     assert set(gate)=={"raw","binding","bound"}
     assert gate["raw"]["format"]==gate["bound"]["format"]=="poseidon-artifact-gate-v2"
     assert gate["raw"]["security"]==gate["bound"]["security"]=="tc128"
     assert gate["binding"]["source_plan_sha256"]==gate["raw"]["plan_sha256"]
     assert gate["binding"]["plan_sha256"]==gate["bound"]["plan_sha256"]
     comparisons.append(a["comparison"])
 initial=row["result"]["acceptance"]
 assert initial["input_groups"]==4 and initial["numerically_validated"]is True
 groups+=4;values+=initial["compared_values"];maximum=max(maximum,initial["maximum_absolute_error"])
 extended=name in scope.reuse_cases(mode)
 for phase,n in ((("new_inputs",3),("repeat",1)) if extended else ()):
  receipt=scope.read(out/name/(phase+".json"));comp=receipt["comparison"]
  assert receipt["status"]=="passed"and comp["passed"]and comp["atol"]==1e-5 and comp["rtol"]==1e-4
  execution=receipt["execution"];assert execution["agent_calls"]==execution["compiler_calls"]==0
  assert execution["encrypted_execution"]and not execution["executor_secret_key"]
  groups+=n;values+=comp["compared_values"];maximum=max(maximum,comp["max_absolute_error"])
 cache=scope.read(out/name/"cache.json");assert cache["package"]["reused"]and cache["generations"]==cache["http_attempts"]==0
 if mode=="live":
  from candidate_contract import strict_json
  provenance=scope.read(task_dir/"generation-provenance.json")
  raw=provenance["answers"][-1]["raw"];answer=strict_json(raw)
  assert answer["request_id"]==q["request_id"]
  candidate=(package/"candidate.py").read_bytes()
  assert candidate==answer["hecate_source"].encode() and hashlib.sha256(candidate).hexdigest()==provenance["candidate_sha256"]
  assert provenance["answers"][-1]["feedback"]==dict(layer="complete",status="passed")
 stats.append(dict(case=name,request_id=q["request_id"],package_binding=manifest["binding"],
  source_origin_preserved="model_source"in manifest["origin"],runtime_artifacts_verified=True,
  numerical_level="numerical",agent_generated=manifest["agent_generated"],package_reloaded=True,
  new_inputs_verified=extended,repeated_execution_verified=extended,cache_verified=True,
  comparison_count=len(comparisons),compiled=True,encrypted_execution=True))
result=dict(format="poseidon-generic-capacity-qualified-evidence-audit-v1",status="passed",mode=mode,
 report_binding=report["binding"],plan_binding=plan["binding"],sdk=plan["sdk"],cases=stats,
 input_groups=groups,output_values=values,maximum_absolute_error=maximum,
 compiler_calls_confirmed=report["compiler_calls_confirmed"],encrypted_calls_confirmed=report["encrypted_calls_confirmed"],
 generations=report["generations"],http_attempts=report["http_attempts"],
 origin="current_real_agent"if mode=="live"else"deterministic_rule",historical_records_rewritten=False)
result["binding"]=scope.digest(result);scope.write(out/"audit.json",result)
print(json.dumps({k:result[k]for k in("status","mode","input_groups","output_values","maximum_absolute_error","binding")}))
