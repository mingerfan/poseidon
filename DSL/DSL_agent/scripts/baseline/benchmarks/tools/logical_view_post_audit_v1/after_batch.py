"""Wait for the frozen owner to exit, then audit once; no generation or retry."""
import sys,time,subprocess,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2]
sys.path[:0]=[str(B/"benchmarks/tools/logical_view_backend_v1"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
scope=q.read(HERE/"scope.json");q.bound(scope)
for path,want in scope["files"].items():q.require(q.sha(path)==want,"audit_scope")
started=time.time();me=live.process_info(os.getpid())
v=dict(format="poseidon-logical-view-readonly-owner-v1",pid=os.getpid(),start=str(me["start"]),scope_binding=scope["binding"])
v["binding"]=q.digest(v);q.write(HERE/"owner.json",v)
while True:
 q.require(time.time()<scope["wait_deadline"],"original_owner_wait_budget")
 info=live.process_info(scope["main_owner"]["pid"])
 if info is None or str(info["start"])!=scope["main_owner"]["start"]or info["state"]in("Z","X"):break
 time.sleep(5)
live.no_active();plan=q.verify("live")
q.require(plan["binding"]==scope["live_plan_binding"],"parent_plan")
out=Path(plan["campaign"]);q.require((out/"terminal.json").exists(),"terminal")
q.require(q.calls(q.states(out),12,48)["calls_resolved"],"unresolved_HTTP_preserved")
q.require(live.measure(live.RESULTS)<=48*1024**3 and __import__("shutil").disk_usage(live.RESULTS).free>=4*1024**3,"storage")
with (HERE/"audit.log").open("x")as log:
 result=subprocess.run([sys.executable,"-B",str(HERE/"mixed_results_audit.py")],stdout=log,stderr=subprocess.STDOUT,timeout=210)
record=dict(format="poseidon-logical-view-readonly-post-batch-v1",scope_binding=scope["binding"],returncode=result.returncode,
 status="passed"if result.returncode==0 else"failed",log_sha256=q.sha(HERE/"audit.log"),provider_calls=0,compiler_calls=0,
 encrypted_calls=0,production_changes=False,wall_seconds=time.time()-started)
record["binding"]=q.digest(record);q.write(HERE/"report.json",record)
print({k:v for k,v in record.items()if k!="wall_seconds"},flush=True)
raise SystemExit(result.returncode)
