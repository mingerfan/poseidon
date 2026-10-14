"""Wait for this frozen six-case offline run, freeze evidence and audit once."""
from pathlib import Path
import sys,time,subprocess,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools";H=T/"tensor_contraction_backend_v1"
sys.path[:0]=[str(T/"static_broadcast_live_v1"),str(B),str(H)]
import run as live,qualification_scope as q
scope=q.read(HERE/"scope.json");q.bound(scope)
for path,want in scope["files"].items():q.require(q.sha(path)==want,"post_frozen_source")
p=q.verify("offline");q.require(p["binding"]==scope["offline_binding"],"post_scope")
started=time.time();result=None
try:
    while True:
        q.require(time.time()<scope["deadline"],"post_deadline")
        info=live.process_info(scope["main"]["pid"])
        if info is None or str(info["start"])!=scope["main"]["start"]or info["state"]in("Z","X"):break
        time.sleep(5)
    for path,want in scope["files"].items():q.require(q.sha(path)==want,"post_frozen_source")
    q.verify("offline");out=Path(p["campaign"]);terminal=q.read(out/"terminal.json");report=q.read(out/"report.json");q.bound(report)
    q.require(terminal["plan_binding"]==p["binding"]and terminal["exit_code"]==0 and
        report["status"]=="complete"and report["passed"]==6 and report["failed"]==0,"six_offline_passes")
    live.no_active()
    audit=Path(scope["audit"]);cases=[];rows={c["case"]:c for c in report["cases"]};files={}
    q.write(audit.parent/"identity.json",q.read(H/"identity-snapshot.json"))
    def freeze(path):
        path=Path(path).resolve();q.regular(path);files[str(path)]=q.sha(path)
    for relative in p["frozen_files"]:freeze(q.ROOT/relative)
    for path in (audit,audit.parent/"identity.json",out/"owner.json",out/"report.json",out/"terminal.json"):freeze(path)
    for c0 in p["cases"]:
        row=rows[c0["id"]];folder=out/c0["id"]
        c=dict(c0,task=str(Path(row["package"]).parent),expected_pass=True,
            reuse_receipts=[(phase,str(folder/(phase+".json")),count)for phase,count in(("new_inputs",3),("repeat",1))])
        cases.append(c)
        for path in (folder/"case-report.json",folder/"task-id.json",folder/"cache.json",*(Path(x[1])for x in c["reuse_receipts"])):freeze(path)
        for path in Path(c["task"]).rglob("*"):
            if path.is_file()and not path.is_symlink():freeze(path)
    a_scope=dict(format="poseidon-contraction-rule-audit-scope-v1",sdk=p["sdk"],cases=cases,frozen_files=files,
        expected_pass_totals=[48,136],expected_fail_totals=[0,0],backend_batch_binding=p["binding"],
        budget=dict(wall_seconds=180,new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,new_bytes=8*1024**2))
    a_scope["binding"]=q.digest(a_scope);q.write(audit.parent/"scope.json",a_scope)
    with(HERE/"audit.log").open("xb")as stream:
        child_result=subprocess.run([sys.executable,"-B",str(audit),"--audit"],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,timeout=210)
    result=dict(status="passed"if child_result.returncode==0 else"incomplete",returncode=child_result.returncode,
        audit_report=str(audit.parent/"report.json"),log_sha256=q.sha(HERE/"audit.log"))
except BaseException as error:
    result=dict(status="incomplete",error_type=type(error).__name__,diagnostic=str(error)[:200])
finally:
    result.update(format="poseidon-contraction-offline-post-audit-v1",scope_binding=scope["binding"],
        seconds=time.time()-started,new_api_calls=0,new_compiler_calls=0,new_encrypted_calls=0)
    result["binding"]=q.digest(result);q.write(HERE/"report.json",result)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result["status"]=="passed"else 2)
