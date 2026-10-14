import sys,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path[:0]=[str(B/"benchmarks/tools/capacity_shallow_backend_v4"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
from application_backend import PoseidonBackend
live.no_active();plan=q.read(B/"benchmarks/tools/capacity_shallow_backend_v4/offline-plan.json");q.bound(plan)
campaign=Path(plan["campaign"]);report=q.read(campaign/"report.json");q.bound(report)
q.require(report["status"]=="complete"and report["not_run"]==0 and report["calls_resolved"]and report["http_attempts"]==0,"terminal_batch")
identity=PoseidonBackend().identity_manifest();q.require(identity["manifest_sha256"]==plan["sdk"],"current_sdk")
q.write(HERE/"identity.json",identity)
cases=[];frozen={};excluded=[];totals=[0,0]
def freeze(p):
 p=Path(p);frozen[str(p)]=q.sha(p)
for rel,want in plan["frozen_files"].items():
 path=ROOT/rel;q.require(q.sha(path)==want,"original_frozen");freeze(path)
freeze(campaign/"report.json");freeze(campaign/"terminal.json")
for row in report["cases"]:
 folder=campaign/row["case"];case=next(c for c in plan["cases"]if c["id"]==row["case"])
 freeze(folder/"case-report.json")
 if row["status"]!="passed":
  excluded.append(dict(case=row["case"],status="failed",failure_layer=row.get("failure_layer"),
    compiler_calls=row["compiler_calls_confirmed"],encrypted_calls=row["encrypted_calls_confirmed"]))
  continue
 task=folder/"tasks"/row["task_id"]
 count=4
 receipts=[]
 for phase,groups in (("new_inputs",3),("repeat",1)):
  file=folder/(phase+".json")
  if file.exists():receipts.append([phase,str(file),groups]);freeze(file);count+=groups
 request=q.read(case["request_file"]);values=math.prod(request["layout"]["output_shape"])
 totals[0]+=count;totals[1]+=count*values
 for path in (task/"program").rglob("*"):
  if path.is_file():freeze(path)
 for path in task.glob("validation-*.json"):
  freeze(path);v=q.read(path)
  if v.get("evidence"):
   ev=Path(v["evidence"])
   for name in ("report.json","arrays.npz","logical-bindings.json"):freeze(ev/name)
   for path in ev.glob("attempt-*/output/decrypted.npy"):freeze(path)
 cases.append(dict(id=case["id"],origin=row["origin"],request_file=case["request_file"],task=str(task),
    expected_pass=True,candidate=case["candidate"],reuse_receipts=receipts))
for f in HERE.glob("*.py"):freeze(f)
freeze(HERE/"identity.json")
scope=dict(format="poseidon-capacity-shallow-audit-scope-v1",sdk=plan["sdk"],backend_batch_binding=plan["binding"],
 cases=cases,retained_failed=excluded,expected_pass_totals=totals,expected_fail_totals=[0,0],
 frozen_files=frozen,budget=dict(wall_seconds=300),api_calls=0,compiler_calls=0,encrypted_calls=0)
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope)
print(json.dumps(dict(binding=scope["binding"],cases=len(cases),retained_failed=len(excluded),totals=totals)))
