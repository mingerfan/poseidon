"""Continue only the untouched 31+1 authorized contexts; never replay a stopped call."""
import copy,fcntl,hashlib,importlib.util,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
OLD_TOOL=ROOT/"scripts/baseline/benchmarks/tools/approved_pending_execution_v3/run.py"
sys.path.insert(0,str(OLD_TOOL.parent))
from accounting import select_unstarted,closed_calls,deadline
s=importlib.util.spec_from_file_location("approved_previous_driver",OLD_TOOL)
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
OLD=d.OUT
OUT=d.WORK/"agent-approved-unstarted-continuation-r1113"
d.OUT=OUT
from hecate_python_env import VENV
from benchmark_graph import require,canonical,digest
def no_running_previous():
    matches=[]
    paths={str(OLD_TOOL),str(Path(__file__).resolve()),
           str(ROOT/"scripts/baseline/run_candidate.py")}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try: tokens=(entry/"cmdline").read_bytes().split(bytes([0]))
        except (OSError,PermissionError):continue
        if any(os.fsdecode(x) in paths for x in tokens if x):matches.append(int(entry.name))
    require(not matches,"Existing authorized execution process")
def verify_no_environment_owner():
    p=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/deps/nix-portable-v012/launcher.lock")
    with p.open("r+") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
def prepare():
    require(not OUT.exists(),"Fresh unstarted continuation; no automatic replay")
    old=d.load(OLD/"plan.json")
    d.verify_parent(old)
    parent=d.load(OLD/"report.json")
    require(parent["status"]=="stopped","Original batch closed")
    no_running_previous()
    retained=copy.deepcopy(old["retained_hashes"])
    groups={};consumed_g=old["prior_generation_reservations"];consumed_h=old["prior_http_attempts"]
    closed=[];unresolved=[]
    reports={}
    for group in ("functional15","matrix_instance60"):
        path=OLD/group/"report.json";report=d.load(path);reports[group]=report
        selected=select_unstarted(old["groups"][group]["cases"],report["rows"])
        for row in report["rows"]:
            consumed_g+=row["result"]["generations"];consumed_h+=row["result"]["http_attempts"]
            task=OLD/group/row["id"]/"tasks"/row["task_id"]
            state=d.load(task/"state.json")
            require(state["generations"]==row["result"]["generations"] and
                    state["http_attempts"]==row["result"]["http_attempts"],"Terminal task accounting")
            if not closed_calls(state):unresolved.append(row["id"])
            closed.append(dict(id=row["id"],status=row["status"],request_id=row["request_id"],
                               task_id=row["task_id"],generations=state["generations"],
                               http_attempts=state["http_attempts"],
                               calls_resolved=closed_calls(state),not_retried=True))
            for p in task.glob("*"):
                if p.is_file() and p.suffix in (".json",".jsonl",".py"):retained[str(p)]=d.sha(p)
            for p in (OLD/group/row["id"]).glob("*.json"):retained[str(p)]=d.sha(p)
        retained[str(path)]=d.sha(path)
        if selected:
            groups[group]=copy.deepcopy(old["groups"][group])
            groups[group].update(cases=selected,deadline_wall=deadline(report["deadline_wall"],time.time()),
                                 max_generations=4*len(selected),max_http_attempts=16*len(selected))
    require(not unresolved,"Previous in-flight provider outcome; retain without automatic retry")
    require(not (OLD/"attention1").exists(),"Attention originally unstarted")
    groups["attention1"]=copy.deepcopy(old["groups"]["attention1"])
    require("functional15" not in groups and [len(groups[g]["cases"]) for g in
            ("matrix_instance60","attention1")]==[31,1],"Exact remaining 31 plus 1")
    for p in (OLD/"plan.json",OLD/"report.json",OLD/"execution-claim.json"):
        retained[str(p)]=d.sha(p)
    retained.update(old["tool_hashes"])
    plan=copy.deepcopy(old)
    plan.update(format="poseidon-approved-unstarted-continuation-v1",
                parent_continuation=old["binding"],parent_terminal_sha256=d.sha(OLD/"report.json"),
                groups=groups,used_by_case={},closed_contexts=closed,
                prior_generation_reservations=consumed_g,prior_http_attempts=consumed_h,
                max_new_generation_reservations=128,max_new_http_attempts=512,
                original_unused_case_quota_not_transferred=True,
                failed_contexts_not_retried=True,unresolved_call_retries=False,
                source_hashes=old["source_hashes"],retained_hashes=retained,
                tool_hashes={str(p):d.sha(p) for p in Path(__file__).parent.glob("*.py")})
    require(consumed_g+128<=old["max_total_generations"] and
            consumed_h+512<=old["max_total_http_attempts"],"No approved budget expansion")
    plan.pop("binding")
    plan["binding"]=digest(plan)
    d.verify_parent(plan)
    OUT.mkdir(mode=0o700)
    (OUT/"plan.json").write_bytes(canonical(plan))
    print(json.dumps(dict(status="prepared",cases=32,binding=plan["binding"],
        prior_generations=consumed_g,prior_http_attempts=consumed_h,
        new_max_generations=128,new_max_http_attempts=512,
        original_matrix_deadline=groups["matrix_instance60"]["deadline_wall"],
        tls="Deferred credential-free verification at pinned run boundary")),flush=True)
def run():
    plan=d.load(OUT/"plan.json");d.verify_parent(plan);no_running_previous()
    tls=d.tls_check()
    with (OUT/"tls-at-run.json").open("x") as f:json.dump(tls,f,sort_keys=True)
    with (OUT/"execution-claim.json").open("x") as f:
        json.dump(dict(pid=os.getpid(),started=time.time(),binding=plan["binding"]),f)
    groups=[];started=time.monotonic()
    def save(status):
        (OUT/"report.json").write_bytes(canonical(dict(
            format="poseidon-approved-unstarted-status-v1",status=status,binding=plan["binding"],
            groups=groups,planned=32,original_planned=76,
            historical_passed=38,historical_failed=6,
            prior_generations=plan["prior_generation_reservations"],
            prior_http_attempts=plan["prior_http_attempts"],
            automatic_uncertain_retry=False,automatic_restart=False,seconds=time.monotonic()-started)))
    save("running")
    try:
        for name in ("matrix_instance60","attention1"):
            d.verify_parent(plan)
            stop=plan["groups"][name]["deadline_wall"]
            if stop is None:stop=time.time()+43200
            require(stop>time.time(),"Original group deadline retained")
            group=dict(id=name,status="running",deadline_wall=stop);groups.append(group);save("running")
            with (OUT/(name+".log")).open("x") as log:
                child=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"--group",name,str(stop)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=max(1,int(stop-time.time())+3))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    group.update(status="stopped",exit_code=124);save("stopped");return 124
            report=d.load(OUT/name/"report.json")
            group.update(exit_code=code,**{k:report[k] for k in ("status","passed","failed","not_run",
                "cumulative_generations","cumulative_http_attempts")})
            save("running")
            if report["status"]!="complete" or code!=0:save("stopped");return code or 1
        save("complete");return 0
    except BaseException:
        save("stopped");raise
if __name__=="__main__":
    if sys.argv[1:]==["--prepare"]:prepare()
    else:
        require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned live environment")
        if sys.argv[1:]==["--run"]:raise SystemExit(run())
        elif len(sys.argv)==4 and sys.argv[1]=="--group":raise SystemExit(d.run_group(sys.argv[2],float(sys.argv[3])))
        else:raise SystemExit("Explicit prepare/run/group only")
