"""Read-only reproduction of previously frozen request bytes."""
import sys,json
from pathlib import Path
B=Path("scripts/baseline");sys.path.insert(0,str(B))
from component_contract import reconstruct_request
from benchmark_graph import canonical,digest
T=B/"benchmarks/tools/generic_capacity_backend_v5"
p=json.loads((B/"benchmarks/tools/output_alias_backend_v2/prepared-requests.json").read_text())
rows=[]
for c in p["cases"]:
 q=json.loads(Path(c["request_file"]).read_text())
 assert canonical(reconstruct_request(q))==canonical(q)
 rows.append(dict(case=c["id"],request_id=q["request_id"],policy=q["public_weight_binding"]["policy"],bytes_unchanged=True))
v=dict(format="poseidon-legacy-request-byte-replay-v1",status="passed",cases=rows,
 paid_calls=0,compiler_calls=0,encrypted_calls=0,historical_records_rewritten=False)
v["binding"]=digest(v)
with (T/"legacy-request-checks.json").open("x")as f:json.dump(v,f,sort_keys=True,indent=2)
print(json.dumps(dict(status=v["status"],cases=len(rows),binding=v["binding"])))
