"""Readonly post-batch sequence; never restarts generation or modifies sources."""
import sys,os,json,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
SCOPE=BASE/"benchmarks/tools/generic_capacity_backend_v10"
sys.path[:0]=[str(BASE),str(SCOPE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import qualification_scope as q
import run as live
scope=q.read(HERE/"after-batch-scope.json");q.bound(scope)
for path,want in scope["frozen_files"].items():q.require(q.sha(path)==want,"queued_tool_changed")
q.require(q.verify("live")["binding"]==scope["live_plan_binding"],"queued_plan_changed")
root=Path(scope["campaign"]);expected=scope["live_owner"]
info=live.process_info(os.getpid())
state=dict(format="poseidon-capacity-post-batch-worker-v1",pid=os.getpid(),start=info["start"],
    phase="waiting_for_frozen_batch_exit",started=time.time(),scope_binding=scope["binding"],
    provider_calls=0,compiler_calls=0,encrypted_calls=0)
q.write(HERE/"after-batch-owner.json",state)
print(json.dumps(state),flush=True)
def group_live():
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        v=live.process_info(int(path.name))
        if v is not None and v["group"]==expected["group"]and v["state"]not in("Z","X"):
            return True
    return False
while True:
    q.require(time.time()<expected["deadline"],"original_live_wall_budget")
    info=live.process_info(expected["pid"])
    owner_live=info is not None and str(info["start"])==expected["start"]and info["state"]not in("Z","X")
    if not owner_live and not group_live():break
    time.sleep(10)
live.no_active()
plan=q.verify("live");q.require((root/"terminal.json").is_file(),"queued_missing_terminal")
report=q.read(root/"report.json");q.bound(report)
q.require(report["status"]in("complete","failed","stopped"),"queued_nonterminal_report")
calls=q.calls(q.states(root),52,208);q.require(calls["calls_resolved"],"queued_unsettled_http")
for path,want in scope["frozen_files"].items():q.require(q.sha(path)==want,"queued_tool_changed")
q.require(scope["audit_wall_seconds"]==720,"queued_wall_budget")
started=time.monotonic();commands=[];status="passed"
for script in scope["sequence"]:
    live.no_active();remain=720-int(time.monotonic()-started)
    q.require(remain>0,"queued_audit_budget")
    log=HERE/(Path(script).stem+"-post-batch.log")
    q.require(not log.exists(),"preserve_post_batch_log")
    with log.open("xb")as output:
        process=subprocess.run(["timeout","-k","5s",str(remain)+"s",sys.executable,"-B",script],
            stdout=output,stderr=subprocess.STDOUT,timeout=remain+10)
    commands.append(dict(tool=script,returncode=process.returncode,log=str(log),log_sha256=q.sha(log)))
    print(json.dumps(commands[-1]),flush=True)
    if process.returncode:
        status="failed";break
result=dict(format="poseidon-capacity-readonly-post-batch-report-v1",status=status,
    scope_binding=scope["binding"],live_report_binding=report["binding"],commands=commands,
    wall_seconds=time.monotonic()-started,provider_calls=0,compiler_calls=0,encrypted_calls=0,
    production_changes=False,historical_records_rewritten=False)
result["binding"]=q.digest(result);q.write(HERE/"after-batch-report.json",result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if status=="passed"else 1)
