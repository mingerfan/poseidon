"""Durable paid-task claims and resume gates. No provider calls or credentials."""
import hashlib,json,math,os
from pathlib import Path
from stage2_agent_campaign import identity
from benchmark_graph import digest
from benchmark_runner import strict_file


def sealed(value):
    result=dict(value);result["binding"]=digest(value);return result


def check(value):
    if type(value) is not dict or value.get("binding")!=digest({k:v for k,v in value.items() if k!="binding"}):
        raise ValueError("State binding changed")
    return value


def exclusive_json(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    # Persistent pre-launch marker: a crash can cause a conservative stop, never
    # a second API call. Do not unlink or silently repair uncertain claims.
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(d)
    finally:os.close(d)


def claim(registry,spec,sources,owner,plan_binding):
    if registry.is_symlink():raise ValueError("Unsafe claim registry")
    registry.mkdir(exist_ok=True)
    token=identity(spec,sources)
    if spec.get("evaluation_identity")!=token:raise ValueError("Evaluation identity changed")
    path=registry/(token+".json")
    value=sealed(dict(format="poseidon-paid-task-claim-v1",evaluation_identity=token,
                      task_id=spec["id"],model_sha256=spec["model_sha256"],request_id=spec["request_id"],
                      owner=str(owner.resolve()),plan_binding=plan_binding,
                      state="claimed_before_launch",automatic_retry_allowed=False))
    try:exclusive_json(path,value)
    except FileExistsError:
        old=check(strict_file(path,1024**2))
        if old["evaluation_identity"]!=token or old["task_id"]!=spec["id"]:raise ValueError("Claim identity mismatch")
        return False,old
    return True,value


def resume_rows(plan,report,output):
    check(report)
    if report["plan_binding"]!=plan["binding"]:raise ValueError("Resume plan mismatch")
    elapsed=report.get("seconds")
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError("Invalid prior wall usage")
    rows=report.get("rows",[]);specs={s["id"]:s for s in plan["cases"]}
    if len({r["id"] for r in rows})!=len(rows) or not {r["id"] for r in rows}<=set(specs):
        raise ValueError("Resume task identity")
    for r in rows:
        if r["status"] not in ("failed","runner_reported_pass_pending_audit","reserved_elsewhere"):
            raise ValueError("Uncertain prior task state")
        if r.get("evaluation_identity")!=specs[r["id"]]["evaluation_identity"]:
            raise ValueError("Resume evaluation identity changed")
        if r["status"]=="reserved_elsewhere":continue
        folder=Path(r["evidence"])
        if folder.is_symlink() or folder.resolve().parent!=output.resolve().parent:
            raise ValueError("Unexpected completed evidence path")
        path=folder/"report.json"
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=r["report_sha256"]:
            raise ValueError("Completed evidence changed")
        data=strict_file(path,8*1024**2)
        if data["status"]=="running":raise ValueError("Prior evidence still running")
    completed={r["id"] for r in rows}
    for spec in plan["cases"]:
        # A launch without a terminal checkpoint is never interpreted as no bill.
        if spec["id"] not in completed and ((output/spec["id"]/"launch.json").exists() or (output/spec["id"]/"launch.json").is_symlink()):
            raise ValueError("Uncertain launched attempt; audit before any resume")
    return rows,float(elapsed)
