"""Freeze application tasks and bounded execution under standing human approval."""
import sys,shlex,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=q.read(HERE/"identity-snapshot.json")
assert PoseidonBackend().identity_manifest()==identity and identity["backend"]==q.read(HERE/"prechange-identity.json")["backend"]
controls=q.read(HERE/"controls.json");assert controls["status"]=="passed"and controls["tests"]==325 and controls["skipped"]==0
prepared=q.read(HERE/"prepared-requests.json");q.bound(prepared)
capture=q.read(HERE/"capture-plan.json");q.bound(capture)
report=q.read(Path(capture["campaign"])/"report.json");q.bound(report)
assert report["passed"]==21 and report["failed"]==report["skipped"]==report["not_run"]==0
guard=q.read(HERE/"controls-scope.json");assert guard["status"]=="passed"and guard["tests"]==12 and guard["skipped"]==0
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
assert activation["approved"]is True
frozen=dict(capture["sdk_sources"])
for name in("qualification.py","qualification_scope.py","test_qualification_scope.py","audit.py","numerical_audit_v3.py","controls.py","controls-scope.json","controls.json"):
 frozen[str((HERE/name).relative_to(ROOT))]=q.sha(HERE/name)
for case in capture["cases"]:frozen.update(case["files"])
for name,want in frozen.items():assert q.sha(ROOT/name)==want
for mode,names,generations,http,compiles,encrypted,wall,seconds in [
 ("offline",q.OFFLINE_CASES,21,0,21,27,2400,180),("live",q.LIVE_CASES,72,288,72,78,43200,14400)]:
 cases=[next(c for c in prepared["cases"]if c["id"]==name)for name in names]
 plan=dict(format="poseidon-public-join-qualification-plan-v1",authorization="human_explicit_current_plan_test_approval",
  user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
  mode=mode,platform="aarch64-linux",backend="poseidon_cpu",sdk=identity["manifest_sha256"],
  generation_guidance="explicit-v14",source_format="poseidon-trusted-pytorch-v34",
  source_root=str(ROOT),campaign=str(q.RESULTS/("agent-public-join-"+mode+"-r1479")),
  numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=cases,package_new_input_cases=list(q.REUSE_CASES),
  budget=dict(global_result_bytes=48*1024**3,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,
   reserve_bytes=4*1024**3,wall_seconds=wall,case_seconds=seconds,max_generations=generations,max_http=http,
   max_compiler_calls=compiles,max_encrypted_calls=encrypted,compiler_jobs=2,link_jobs=1,
   encrypted_concurrency=1,api_concurrency=1),frozen_files=frozen,
  provider=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
   api_timeout=1200,max_tokens=384000,max_repairs=3,provider_retries=3,money_limit=None),
  storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
  outward_content=["public normalized graph","fixed public weights or descriptors","layout","DSL rules","controlled diagnostic"],
  historical_records_rewritten=False,models_or_reference_changed=False,backend_core_changed=False,parameters_changed=False,
  network_changed=False,installations=0,downloads=0,git_actions=[],transmission_retry_may_duplicate_billing=True)
 plan["binding"]=q.digest(plan);q.write(HERE/(mode+"-plan.json"),plan)
 q.write(ROOT/("docs/baseline/agent-public-join-"+mode+"-scope-r1479.json"),plan)
 q.verify(mode)
 print(json.dumps(dict(mode=mode,planned=len(cases),binding=plan["binding"],sdk=plan["sdk"])))
