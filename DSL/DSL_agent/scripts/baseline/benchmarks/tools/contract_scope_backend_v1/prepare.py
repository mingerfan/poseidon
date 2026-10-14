"""Freeze new guidance/backend controls; no API or encrypted execution."""
from pathlib import Path
import sys,json,copy,shlex
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from component_contract import request_options,reconstruct_request
from application_planning.layout import plan_model,check_plan
from benchmark_suite import Builder
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest();q.write(HERE/"identity-snapshot.json",identity)
oldid=q.read(B/"benchmarks/tools/capacity_shallow_live_v1/identity-snapshot.json")
q.require(identity["backend"]==oldid["backend"],"backend_unchanged")
oldplan=q.read(B/"benchmarks/tools/capacity_shallow_backend_v2/offline-plan.json");q.bound(oldplan)
rows=[];frozen={};compat=[]
# Reconstruct original requests against current code; do not re-label prior results.
for original in oldplan["cases"]:
 old=q.read(original["request_file"]);q.require(reconstruct_request(old)==old,"legacy_request_bytes")
 compat.append(dict(case=original["id"],request_id=old["request_id"],request_sha256=q.sha(original["request_file"])))
for name in q.OFFLINE_CASES:
 if name=="public_square":
  b=Builder([(4,)]);model=b.finish(b.node("square",["input0"]))
  options=dict(generation_guidance="explicit-v16",construction_profile="hecate-unified-public-v1")
  plan=plan_model(model,options=options);request=check_plan(plan,model)
  x=request["layout"]["inputs"][0]["dsl_name"]
  source=request["generation_guidance"]["entry_header"]+"\n    a = "+x+" * "+x+"\n    if 1 < 2:\n        a = a + 0.0\n    return [a]\n"
  origin="manual_public_construct"
 else:
  original=next(c for c in oldplan["cases"]if c["id"]==name)
  old=q.read(original["request_file"]);model=old["model"];options=copy.deepcopy(original["options"])
  options["generation_guidance"]="explicit-v16"
  plan=plan_model(model,options=options);request=check_plan(plan,model)
  q.require(all(request[k]==v for k,v in old.items()if k not in("generation_guidance","request_id")),"guidance_only_change")
  source=Path(original["candidate"]).read_text();origin="new_rule_baseline"
 dest=HERE/"prepared"/name;dest.mkdir(parents=True)
 for key,data in(("model",model),("plan",plan),("request",request),("public-request",public_request(request))):
  q.write(dest/(key+".json"),data)
 (dest/"rule.py").write_text(source)
 rows.append(dict(id=name,execution_source_kind="graph",source_intake_failure=None,
  execution_source_graph=str(dest/"model.json"),execution_source_graph_sha256=q.sha(dest/"model.json"),
  request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),request_id=request["request_id"],
  public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"),
  layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),options=plan["selection"]["options"],
  candidate=str(dest/"rule.py"),candidate_sha256=q.sha(dest/"rule.py"),origin=origin))
 for f in dest.iterdir():frozen[str(f.relative_to(ROOT))]=q.sha(f)
 for rel,h in plan["source_authority"]["files"].items():frozen[str((B/rel).relative_to(ROOT))]=h
for f in HERE.glob("*.py"):frozen[str(f.relative_to(ROOT))]=q.sha(f)
for name in("run.py","scope.py"):
 f=B/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(f.relative_to(ROOT))]=q.sha(f)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
activation=q.read(ROOT/"docs/baseline/agent-result-storage-activation-r1442.json");q.bound(activation)
p=dict(format="poseidon-contract-scope-offline-v1",mode="offline",authorization="human_explicit_current_plan_test_approval",
 storage_activation="docs/baseline/agent-result-storage-activation-r1442.json",storage_activation_binding=activation["binding"],
 sdk=identity["manifest_sha256"],platform="aarch64-linux",backend="poseidon_cpu",numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
 cases=rows,frozen_files=frozen,legacy_request_compatibility=compat,
 budget=dict(wall_seconds=1800,case_seconds=600,max_generations=3,max_http=0,max_compiler_calls=3,max_encrypted_calls=9,
 api_concurrency=1,encrypted_concurrency=1,compiler_jobs=2,link_jobs=1,new_artifact_bytes=4*1024**3,
 per_case_bytes=4*1024**3,global_result_bytes=48*1024**3,reserve_bytes=4*1024**3),
 package_new_input_cases=list(q.REUSE_CASES),campaign=str(q.RESULTS/"agent-contract-scope-offline-r1545"),
 origin="mixed_rule_and_manual_public_construct",api_calls=0,network_changed=False,backend_core_changed=False,parameters_changed=False)
p["binding"]=q.digest(p);q.write(HERE/"offline-plan.json",p);q.verify("offline")
q.write(ROOT/"docs/baseline/agent-contract-scope-offline-proposal-r1545.json",p)
print(json.dumps(dict(binding=p["binding"],sdk=p["sdk"],cases=3,legacy_requests=len(compat),api_calls=0)))
