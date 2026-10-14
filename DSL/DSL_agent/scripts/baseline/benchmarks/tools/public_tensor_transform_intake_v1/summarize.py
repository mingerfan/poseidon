"""Final current-identity qualification summary; historical records remain unchanged."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
def read(p):return json.loads(Path(p).read_text())
def bound(v):assert v["binding"]==digest({k:x for k,x in v.items()if k!="binding"})
plans={m:read(HERE/(m+"-plan.json"))for m in("offline","live")}
audits={};reports={}
for mode,p in plans.items():
 bound(p);r=read(Path(p["campaign"])/"report.json");a=read(Path(p["campaign"])/"audit.json")
 bound(r);bound(a);assert a["status"]=="passed"and a["report_binding"]==r["binding"]and r["plan_binding"]==p["binding"]
 audits[mode]=a;reports[mode]=r
prepared=read(HERE/"prepared-requests.json");bound(prepared)
controls=[read(HERE/name)for name in("controls-qualified.json","controls-routing-initial.json")]
assert all(r["passed"]and r["failures"]==r["errors"]==r["skipped"]==0 for r in controls)
old=read(HERE/"prechange-identity.json");identity=read(HERE/"identity-snapshot.json")
assert old["backend"]==identity["backend"]
capture_plan=read(HERE/"capture-plan.json");capture=read(Path(capture_plan["campaign"])/"report.json")
bound(capture_plan);bound(capture);assert capture["passed"]==18 and capture["failed"]==capture["not_run"]==capture["skipped"]==0
record=dict(format="poseidon-public-tensor-transform-phase-acceptance-v1",sdk=identity["manifest_sha256"],
 backend="poseidon_cpu",pytorch_source_format="poseidon-trusted-pytorch-v31",
 public_value_policy="poseidon-public-value-expressions-v4",
 source_contexts=15,unique_public_generation_requests=len({c["request_id"]for c in plans["live"]["cases"]}),
 compatibility_contexts=3,operators=["repeat","tile","repeat_interleave","flip","roll"],
 controls=dict(passed=sum(c["tests"]for c in controls),failed=0,skipped=0),
 plaintext=dict(groups=sum(c["plaintext_groups"]for c in prepared["cases"]),values=sum(c["plaintext_values"]for c in prepared["cases"])),
 source_capture_report=str(Path(capture_plan["campaign"])/"report.json"),source_capture_binding=capture["binding"],
 offline=audits["offline"],live=audits["live"],
 provider_usage_records=[r["result"]["generation"].get("usage")for r in reports["live"]["cases"]],
 first_attempt_passed=sum(r["result"]["generation"]["first_attempt_passed"]is True for r in reports["live"]["cases"]),
 repairs=sum(r["result"]["generation"]["repairs_used"]for r in reports["live"]["cases"]),
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),backend_unchanged=True,parameters_unchanged=True,network_unchanged=True,
 installations=0,downloads=0,git_actions=[],history_rewritten=False,
 default_source="poseidon-trusted-pytorch-v18",latest_source="poseidon-trusted-pytorch-v28",
 unverified=["vector/zero repeat_interleave counts","dynamic shape","public transforms in JSON graph",
 "large frontend constant or physical capacity","original-function application accuracy","real CPU bootstrap","CUDA hardware","full ResNet20"])
record["binding"]=digest(record)
target=ROOT/"docs/baseline/agent-public-tensor-transforms-acceptance-r1458.json"
assert not target.exists();target.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:record[k]for k in("binding","sdk","source_contexts","compatibility_contexts","first_attempt_passed","repairs")}))
