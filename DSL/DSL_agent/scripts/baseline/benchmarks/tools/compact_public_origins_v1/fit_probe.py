import sys,json
from pathlib import Path
B=Path("scripts/baseline");sys.path.insert(0,str(B))
from component_contract import prepare_task
from deepseek_provider import public_request
from public_origin_columns import restore
from benchmark_graph import canonical,digest
T=B/"benchmarks/tools/compact_public_origins_v1"
cases=json.loads((B/"benchmarks/tools/generic_capacity_intake_v1/cases.json").read_text())
m=json.loads(Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-generic-capacity-capture-r1489/rank_axis_8/model.json").read_text())
o=dict(cases[0]["options"],chunk_period=256,public_weight_policy="descriptors-compact-origins-v3")
q=prepare_task(m,o);p=public_request(q)
assert restore(p["constant_origins"])==q["constant_origins"] and public_request(p)==p
r=dict(format="poseidon-public-origin-fit-probe-v1",status="passed",request_id=q["request_id"],
 public_context_bytes=len(canonical(p)),expanded_origin_bytes=len(canonical(q["constant_origins"])),
 compact_origin_bytes=len(canonical(p["constant_origins"])),model_sha256=digest(m),
 budget_bytes=131072,paid_calls=0,compiler_calls=0,encrypted_calls=0)
with (T/"fit-probe.json").open("x")as f:json.dump(r,f,indent=2)
print(json.dumps(r))
