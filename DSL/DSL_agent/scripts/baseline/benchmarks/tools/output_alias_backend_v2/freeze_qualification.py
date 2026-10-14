"""Freeze new adapter identity; old failed bindings and source references stay historical."""
import sys,shlex,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as q
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
 [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=PoseidonBackend().identity_manifest()
old=BASE/"benchmarks/tools/output_alias_integration_v1"
assert identity["backend"]==q.read(old/"identity-snapshot.json")["backend"]
q.write(HERE/"identity-snapshot.json",identity)
controls=q.read(HERE/"controls.json");assert controls["status"]=="passed"and not controls["failures"]and not controls["errors"]and not controls["skipped"]
parent=q.read(old/"offline-plan.json");q.bound(parent)
r=q.read(Path(parent["campaign"])/"report.json");q.bound(r)
assert r["status"]=="complete"and r["passed"]==3 and r["failed"]==9 and r["http_attempts"]==0
guard=q.read(HERE/"controls-scope.json");assert guard["status"]=="passed"and guard["tests"]==12
captured=q.read(old/"capture-plan.json");q.bound(captured)
report=q.read(Path(captured["campaign"])/"report.json");q.bound(report)
assert report["passed"]==12 and report["failed"]==report["skipped"]==report["not_run"]==0
prepared=q.read(HERE/"prepared-requests.json");q.bound(prepared)
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation);assert activation["approved"]is True
frozen={}
for stage in ("generation","compilation","execution","validation"):
 for section in ("sources","configuration_files"):
  for name,want in identity[stage][section].items():
   path=BASE/name if section=="sources"else ROOT/name
   relative=str(path.relative_to(ROOT));assert q.sha(path)==want
   assert relative not in frozen or frozen[relative]==want
   frozen[relative]=want
for case in captured["cases"]:frozen.update(case["files"])
for name in ("qualification.py","qualification_scope.py","test_qualification_scope.py","audit.py","numerical_audit_v3.py",
             "controls.py","controls.json","controls-scope.json","prepared-requests.json"):
 frozen[str((HERE/name).relative_to(ROOT))]=q.sha(HERE/name)
for case in prepared["cases"]:
 for key in ("request_file","public_request_file"):frozen[str(Path(case[key]).relative_to(ROOT))]=q.sha(case[key])
for rel,want in frozen.items():assert q.sha(ROOT/rel)==want
for mode,names,gens,http,compiles,encrypted,wall,seconds in(
 ("offline",q.OFFLINE_CASES,12,0,12,18,2400,180),("live",q.LIVE_CASES,36,144,36,42,43200,14400)):
 cases=[next(c for c in prepared["cases"]if c["id"]==name)for name in names]
 p=dict(format="poseidon-output-alias-qualification-plan-v2",authorization="human_explicit_current_plan_test_approval",
 user_authorization="目前执行的计划内后续所有的离线和真实测试全部批准同意",
 mode=mode,platform="aarch64-linux",backend="poseidon_cpu",sdk=identity["manifest_sha256"],
 source_root=str(ROOT),campaign=str(q.RESULTS/("agent-output-alias-"+mode+"-r1485")),
 numeric_tolerance=dict(atol=1e-5,rtol=1e-4),cases=cases,package_new_input_cases=list(q.REUSE_CASES),
 source_reference_parent_binding=report["binding"],prior_offline_binding=r["binding"],
 source_reference_reused=True,request_values_unchanged=True,original_models_or_references_modified=False,
 backend_core_changed=False,parameters_changed=False,network_changed=False,installations=0,downloads=0,git_actions=[],
 generation_guidance="explicit-v15; compatibility v14",source_format="poseidon-trusted-pytorch-v34",
 budget=dict(global_result_bytes=48*1024**3,new_artifact_bytes=4*1024**3,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3,
 wall_seconds=wall,case_seconds=seconds,max_generations=gens,max_http=http,max_compiler_calls=compiles,
 max_encrypted_calls=encrypted,compiler_jobs=2,link_jobs=1,encrypted_concurrency=1,api_concurrency=1),
 frozen_files=frozen,storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 provider=dict(provider="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,api_timeout=1200,
 max_tokens=384000,max_repairs=3,provider_retries=3,money_limit=None),
 outward_content=["public graph","fixed public weights or descriptors","layout","DSL rules","controlled diagnostic"],
 transmission_retry_may_duplicate_billing=True,historical_records_rewritten=False)
 p["binding"]=q.digest(p);q.write(HERE/(mode+"-plan.json"),p)
 q.write(ROOT/("docs/baseline/agent-output-alias-"+mode+"-scope-r1485.json"),p)
 q.verify(mode)
 print(json.dumps(dict(mode=mode,planned=len(cases),sdk=p["sdk"],binding=p["binding"])))
