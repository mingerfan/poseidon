"""Wait for the existing frozen batch, then perform one read-only evidence audit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def require(ok,code):
 if not ok:raise ValueError(code)
def write(p,v):
 with Path(p).open("x")as f:json.dump(v,f,sort_keys=True,separators=(",",":"),allow_nan=False)
scope=read(HERE/"scope.json")
require(scope["binding"]==digest({k:v for k,v in scope.items()if k!="binding"}),"dispatch_scope")
def verify():
 for p,h in scope["files"].items():require(sha(p)==h,"dispatch_file_changed")
verify()
sys.path[:0]=[str(B/"benchmarks/tools/provenance_transport_live_v2"),str(B/"benchmarks/tools/static_broadcast_live_v1"),str(B)]
import qualification_scope as q,run as live
p=q.verify("live");require(p["binding"]==scope["live_plan_binding"],"live_plan")
me=live.process_info(os.getpid());own=dict(pid=os.getpid(),start=str(me["start"]),scope_binding=scope["binding"])
write(HERE/"owner.json",own)
started=time.time();owner=scope["main_owner"];result=None
try:
 while True:
  require(time.time()<scope["wait_deadline"],"wait_deadline")
  info=live.process_info(owner["pid"])
  if info is None or str(info["start"])!=owner["start"]or info["state"]in("Z","X"):break
  time.sleep(5)
 verify();q.verify("live")
 terminal=Path(p["campaign"])/"terminal.json"
 require(terminal.is_file(),"terminal_missing_no_restart")
 require(read(terminal)["plan_binding"]==scope["live_plan_binding"],"terminal_binding")
 live.no_active()
 require(live.measure(q.RESULTS)<=48*1024**3,"artifact_budget")
 require(__import__("shutil").disk_usage(q.RESULTS).free>=4*1024**3,"disk_reserve")
 with (HERE/"audit.log").open("x")as stream:
  cp=subprocess.run([sys.executable,"-B",scope["audit_script"],"--audit"],stdin=subprocess.DEVNULL,
    stdout=stream,stderr=subprocess.STDOUT,timeout=330)
 result=dict(status="passed"if cp.returncode==0 else"incomplete",returncode=cp.returncode,
   audit_report=str(Path(scope["audit_script"]).parent/"report.json"),log_sha256=sha(HERE/"audit.log"))
except BaseException as e:
 result=dict(status="incomplete",error_type=type(e).__name__,diagnostic=str(e)[:160])
finally:
 result.update(format="poseidon-capacity-shallow-post-batch-dispatch-v1",scope_binding=scope["binding"],
  new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0,seconds=time.time()-started)
 result["binding"]=digest(result);write(HERE/"report.json",result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result["status"]=="passed"else 2)
