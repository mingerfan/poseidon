"""Freeze three source-level live requests only after independent offline acceptance."""
from pathlib import Path
import sys,json,shlex
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(HERE),str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
from deepseek_provider import public_request
identity=PoseidonBackend().identity_manifest()
audit=q.read(BASE/"benchmarks/tools/tensor_gather_acceptance_audit_v1/report.json");q.bound(audit)
q.require(audit["status"]=="passed"and audit["sdk"]==identity["manifest_sha256"]and len(audit["cases"])==3,
 "three_case_offline_accepted")
prior=q.read(BASE/"benchmarks/tools/tensor_gather_backend_v1/offline-plan.json");q.bound(prior)
q.write(HERE/"identity-snapshot.json",identity);rows=[];frozen={}
for old in prior["cases"]:
 request=q.read(old["request_file"]);plan=q.read(old["layout_plan"])
 dest=HERE/"prepared"/old["id"];dest.mkdir(parents=True)
 for kind,data in(("request",request),("plan",plan),("public-request",public_request(request))):
  q.write(dest/(kind+".json"),data)
 row=dict(old);row.pop("candidate");row.pop("candidate_sha256");row["origin"]="current_real_agent"
 row.update(request_file=str(dest/"request.json"),request_file_sha256=q.sha(dest/"request.json"),
 layout_plan=str(dest/"plan.json"),layout_plan_sha256=q.sha(dest/"plan.json"),
 public_request_file=str(dest/"public-request.json"),public_request_file_sha256=q.sha(dest/"public-request.json"))
 rows.append(row)
 for p in dest.iterdir():frozen[str(p.relative_to(ROOT))]=q.sha(p)
 for rel,want in plan["source_authority"]["files"].items():frozen[str((BASE/rel).relative_to(ROOT))]=want
 for p in Path(row["source_manifest"]).parent.rglob("*"):
  if p.is_file():frozen[str(p)]=q.sha(p)
for p in HERE.glob("*.py"):frozen[str(p.relative_to(ROOT))]=q.sha(p)
for name in("run.py","scope.py"):
 p=BASE/"benchmarks/tools/static_broadcast_live_v1"/name;frozen[str(p.relative_to(ROOT))]=q.sha(p)
frozen[str((HERE/"identity-snapshot.json").relative_to(ROOT))]=q.sha(HERE/"identity-snapshot.json")
scope=dict(prior);scope.pop("binding")
scope.update(format="poseidon-public-tensor-gather-live-v1",mode="live",cases=rows,frozen_files=frozen,
 sdk=identity["manifest_sha256"],campaign=str(q.RESULTS/"agent-tensor-gather-live-r1577"),
 source_pipeline_required=True,source_format="poseidon-trusted-pytorch-v37",
 complete_source_cases=list(q.LIVE_CASES),retained_graph_cases=[],
 offline_acceptance_binding=audit["binding"],new_agent_generation=True,origin="current_real_agent")
scope["budget"]=dict(scope["budget"],wall_seconds=43200,case_seconds=14400,max_generations=12,max_http=48,
 max_compiler_calls=12,max_encrypted_calls=18)
scope["provider"]=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,
 api_timeout_seconds=1200,max_tokens=384000,max_repairs=3,provider_retries=3)
scope["explicit_currency_limit"]=None;scope["transmission_retry_may_duplicate_billing"]=True
scope["outward_content"]=["public graph","weights or descriptors","layout","DSL rules","controlled feedback"]
scope["excluded_content"]=["rule or manual DSL source","private inputs","reference","credentials","secret keys"]
scope["historical_records_rewritten"]=False;scope["previous_unknown_api_outcomes_retried"]=False
scope["binding"]=q.digest(scope);q.write(HERE/"live-plan.json",scope);q.verify("live")
q.write(ROOT/"docs/baseline/agent-tensor-gather-live-proposal-r1577.json",scope)
print("LIVE_FROZEN",scope["binding"],scope["sdk"])
