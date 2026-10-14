"""Developer-only guarded source cutover, never part of an application request."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
STAGE=Path(__file__).resolve().parent
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
OLD=WORK/"results/agent-approved-pending-continuation-r1099"
LIVE=WORK/"results/agent-approved-unstarted-continuation-r1141"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def eligible(root,groups,expected=None):
    expected={"matrix_instance60":31,"attention1":1} if expected is None else expected
    if root.get("status")!="complete" or root.get("planned")!=sum(expected.values()):return False
    entries=root.get("groups",[])
    if type(entries) is not list or len(entries)!=len(expected) or any(
            x.get("status")!="complete" or x.get("exit_code")!=0 for x in entries):return False
    if {x.get("id") for x in entries}!=set(expected):return False
    for group,count in expected.items():
        d=groups.get(group,{})
        if d.get("status")!="complete" or d.get("planned")!=count or d.get("not_run")!=0:return False
        rows=d.get("rows")
        if type(rows) is not list or len(rows)!=count:return False
        if any(r.get("status") not in ("passed","failed") or
               r.get("result",{}).get("phase")!="complete" for r in rows):return False
        if any(type(d.get(k)) is not int or d[k]<0 for k in ("passed","failed")):return False
        if any(sum(r["status"]==status for r in rows)!=d[key] for status,key in
               (("passed","passed"),("failed","failed"))):return False
    return True
def closed_approval(current,groups,original):
    """Require exact original cases split into closed history and closed live tasks."""
    try:
        source=[e["case"] for g in original["groups"].values() for e in g["cases"]]
        approved={e["id"]:e for e in source}
        if len(approved)!=76 or len(source)!=76:return False
        closed=current["closed_contexts"];seen=set()
        for item in closed:
            name=item["id"]
            if name not in approved or name in seen or item["request_id"]!=approved[name]["request_id"]:return False
            if item.get("status") not in ("passed","failed") or item.get("calls_resolved") is not True or item.get("not_retried") is not True:return False
            for key,limit in (("generations",4),("http_attempts",16)):
                if type(item.get(key)) is not int or not 0<=item[key]<=limit:return False
            seen.add(name)
        if len(closed)!=51:return False
        for group,entries in current["groups"].items():
            actual=groups[group]["rows"]
            expected={e["case"]["id"]:e["case"] for e in entries["cases"]}
            if len(expected)!=len(entries["cases"]) or set(expected)!={r["id"] for r in actual}:return False
            for row in actual:
                name=row["id"]
                if name not in approved or name in seen or expected[name]!=approved[name] or row["request_id"]!=approved[name]["request_id"]:return False
                seen.add(name)
        if seen!=set(approved):return False
        for field,status in (("historical_passed","passed"),("historical_failed","failed")):
            if current.get(field)!=sum(c["status"]==status for c in closed):return False
        for initial,previous,key in (("prior_generation_reservations","prior_generation_reservations","generations"),
                                     ("prior_http_attempts","prior_http_attempts","http_attempts")):
            if current[previous]!=original[initial]+sum(c[key] for c in closed):return False
        return True
    except (KeyError,TypeError,ValueError):return False

def active_processes():
    endings=("approved_pending_execution_v3/run.py","approved_pending_execution_v4/run.py",
             "approved_pending_execution_v5/run.py",
             "run_candidate.py","candidate_worker.py","deepseek_http_worker.py")
    ids=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/"cmdline").read_bytes().split(bytes([0]))
        except OSError:continue
        if any(any(os.fsdecode(a).endswith(e) for e in endings) for a in args if a):
            ids.append(int(entry.name))
    return ids
def verify_sources():
    p=load(STAGE/"plan.json")
    if p["stage"]!="not_applied":raise ValueError("Staged source already applied")
    # Bind both the full component regression and the later portable tests.
    a,b=load(STAGE/"acceptance-r1147.json"),load(STAGE/"acceptance-r1148.json")
    for item in (a,b):
        if item["failed"] or item["errors"] or item["skipped"]:raise ValueError("Incomplete static regression")
    for relative,h in p["staged_files"].items():
        source=STAGE/relative
        if source.parent!=STAGE/"targets" or source.is_symlink() or sha(source)!=h:
            raise ValueError("Staged source changed")
        if a["source_hashes"].get(relative,b["source_hashes"].get(relative))!=h:
            raise ValueError("Staged source not covered by test binding")
    for relative,h in p["original_files"].items():
        target=ROOT/relative
        if target.is_symlink() or sha(target)!=h:raise ValueError("Production source changed; never overwrite")
        backup=STAGE/"originals"/target.name
        if sha(backup)!=h:raise ValueError("Original source recovery snapshot")
    for name in p["new_files"]:
        if (ROOT/"scripts/baseline"/name).exists():raise ValueError("New file collides with current source")
    original=load(OLD/"plan.json")
    for label,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        if any(sha(base/k)!=h for k,h in original[label].items()):raise ValueError("Original frozen binding changed")
    if subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()!="lhy-agent-dsl":
        raise ValueError("Development branch changed")
    return p
def check():
    if not (LIVE/"report.json").exists():return dict(status="waiting_for_current_batch",applied=False)
    root=load(LIVE/"report.json")
    groups={g:load(LIVE/g/"report.json") for g in ("matrix_instance60","attention1") if (LIVE/g/"report.json").exists()}
    if not eligible(root,groups,{"matrix_instance60":24,"attention1":1}):return dict(status="waiting_for_current_batch",applied=False,binding=root["binding"])
    if active_processes():return dict(status="waiting_for_process_exit",applied=False)
    plan=verify_sources()
    current=load(LIVE/"plan.json")
    if not closed_approval(current,groups,load(OLD/"plan.json")):
        raise ValueError("Original 76-case scope is not entirely closed")
    payload={k:v for k,v in current.items() if k!="binding"}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=current["binding"] or root["binding"]!=current["binding"]:
        raise ValueError("Current batch plan binding")
    if any(sha(p)!=h for p,h in current["retained_hashes"].items()):
        raise ValueError("Current batch retained evidence changed")
    for group,data in groups.items():
        approved={e["case"]["id"]:e["case"]["request_id"] for e in current["groups"][group]["cases"]}
        actual={r["id"]:r["request_id"] for r in data["rows"]}
        if approved!=actual or len(actual)!=len(data["rows"]) or data["binding"]!=current["binding"]:
            raise ValueError("Current batch case identity")
        for row in data["rows"]:
            task=LIVE/group/row["id"]/"tasks"/row["task_id"]
            state=load(task/"state.json")
            calls=[e for e in state["ledger"] if e.get("event")=="provider_checkpoint"]
            final=calls[-1]["outcomes"] if calls else []
            if state["phase"]!="complete" or len(final)!=state["http_attempts"] or any(
                    c["status"] not in ("failed","response_received") for c in final):
                raise ValueError("Provider outcome not terminal")
    return dict(status="ready",applied=False,targets=len(plan["staged_files"]),binding=root["binding"])
def apply():
    result=check()
    if result["status"]!="ready":return result
    lock=WORK/"deps/nix-portable-v012/launcher.lock"
    with lock.open("r+") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Recheck after excluding any new SDK launch.
        if check()["status"]!="ready":raise ValueError("Cutover boundary changed")
        plan=verify_sources()
        recovery=STAGE/("applied-"+time.strftime("%Y%m%dT%H%M%S",time.gmtime()))
        recovery.mkdir()
        shutil.copytree(STAGE/"originals",recovery/"originals")
        shutil.copy2(STAGE/"plan.json",recovery/"plan.json")
        before=subprocess.check_output(["git","status","--short","--branch"],cwd=ROOT,text=True)
        (recovery/"git-status-before.txt").write_text(before)
        queue=[]
        for name,h in plan["staged_files"].items():
            source=STAGE/name;target=ROOT/"scripts/baseline"/source.name
            temporary=target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
            with temporary.open("xb") as out:out.write(source.read_bytes());out.flush();os.fsync(out.fileno())
            os.chmod(temporary,target.stat().st_mode&0o777 if target.exists() else 0o644)
            queue.append((temporary,target,h))
        # Install trusted dependency modules before their callers.
        order=["candidate_check_diagnostic.py","unified_binding_guidance.py","decorated_functions.py",
               "native_array_core.py","component_contract.py","application_diagnostics.py","unified_graph_contract.py",
               "validation_adapter.py","sandbox_policy.py","run_candidate.py","test_binding_feedback.py","application_component.py"]
        queue.sort(key=lambda x:order.index(x[1].name))
        try:
            for temporary,target,h in queue:
                os.replace(temporary,target)
                if sha(target)!=h:raise ValueError("Installed source integrity")
        except BaseException:
            (recovery/"incomplete.txt").write_text("Recover only this exact patch from originals; do not reset the checkout.\n")
            raise
        record=dict(format="poseidon-agent-source-cutover-v1",source_batch=result["binding"],
                    targets={str(t.relative_to(ROOT)):h for _,t,h in queue},
                    recovery=str(recovery),provider_calls=0,compilations=0,encrypted_executions=0,
                    backend_changes=False,post_cutover_tests_required=True,
                    historical_evidence_unchanged=True)
        (recovery/"record.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
        plan["stage"]="applied_pending_post_cutover_acceptance";plan["cutover_record"]=str(recovery/"record.json")
        (STAGE/"plan.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
        return dict(status="applied_pending_post_cutover_acceptance",applied=True,record=str(recovery/"record.json"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--apply",action="store_true")
    args=p.parse_args();print(json.dumps(apply() if args.apply else check()))
