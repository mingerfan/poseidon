"""Continue the 24+1 untouched cases in the original 76-case approval."""
import copy,fcntl,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
from accounting import select_unstarted,closed_calls,deadline,merge_closed
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
OLD_TOOL=ROOT/"scripts/baseline/benchmarks/tools/approved_pending_execution_v3/run.py"
sys.path.insert(0,str(OLD_TOOL.parent))
s=importlib.util.spec_from_file_location("approved_previous_driver",OLD_TOOL)
d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
ORIGINAL=d.OUT
PREVIOUS=d.WORK/"agent-approved-unstarted-continuation-r1113"
OUT=d.WORK/"agent-approved-unstarted-continuation-r1141"
d.OUT=OUT
from hecate_python_env import VENV
from benchmark_graph import require,canonical,digest
def no_running_previous():
    paths={str(OLD_TOOL),str(Path(__file__).resolve()),
           str(ROOT/"scripts/baseline/benchmarks/tools/approved_pending_execution_v4/run.py"),
           str(ROOT/"scripts/baseline/run_candidate.py"),
           str(ROOT/"scripts/baseline/candidate_worker.py"),
           str(ROOT/"scripts/baseline/deepseek_http_worker.py")}
    matches=[]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:tokens=(entry/"cmdline").read_bytes().split(bytes([0]))
        except OSError:continue
        if any(os.fsdecode(x) in paths for x in tokens if x):matches.append(int(entry.name))
    require(not matches,"Existing authorized execution process")
def verify_no_environment_owner():
    p=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/deps/nix-portable-v012/launcher.lock")
    with p.open("r+") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f.fileno(),fcntl.LOCK_UN)
def prepare():
    require(not OUT.exists(),"Fresh continuation only; no replay")
    previous=d.load(PREVIOUS/"plan.json");d.verify_parent(previous)
    parent=d.load(PREVIOUS/"report.json")
    require(parent["status"]=="stopped" and parent["binding"]==previous["binding"],"Previous terminal binding")
    no_running_previous();verify_no_environment_owner()
    report_path=PREVIOUS/"matrix_instance60/report.json"
    report=d.load(report_path)
    require(report["status"]=="stopped" and report["binding"]==previous["binding"],"Previous group terminal")
    selected=select_unstarted(previous["groups"]["matrix_instance60"]["cases"],report["rows"])
    retained=copy.deepcopy(previous["retained_hashes"]);closed=[]
    consumed_g=previous["prior_generation_reservations"];consumed_h=previous["prior_http_attempts"]
    for row in report["rows"]:
        task=PREVIOUS/"matrix_instance60"/row["id"]/"tasks"/row["task_id"]
        state=d.load(task/"state.json")
        require(state["generations"]==row["result"]["generations"] and state["http_attempts"]==row["result"]["http_attempts"],"Closed task accounting")
        require(closed_calls(state),"Unresolved provider outcome; retain without retry")
        consumed_g+=state["generations"];consumed_h+=state["http_attempts"]
        closed.append(dict(id=row["id"],status=row["status"],request_id=row["request_id"],task_id=row["task_id"],
            generations=state["generations"],http_attempts=state["http_attempts"],calls_resolved=True,not_retried=True))
        for p in task.glob("*"):
            if p.is_file() and p.suffix in (".json",".jsonl",".py"):retained[str(p)]=d.sha(p)
        for p in task.parent.parent.glob("*.json"):retained[str(p)]=d.sha(p)
    require(report["generations"]==sum(x["generations"] for x in closed) and
            report["http_attempts"]==sum(x["http_attempts"] for x in closed),"Previous group counters")
    require(not (PREVIOUS/"attention1").exists() and not (ORIGINAL/"attention1").exists(),"Attention not started")
    groups={g:copy.deepcopy(previous["groups"][g]) for g in ("matrix_instance60","attention1")}
    groups["matrix_instance60"].update(cases=selected,deadline_wall=deadline(report["deadline_wall"],time.time()),
        max_generations=4*len(selected),max_http_attempts=16*len(selected))
    groups["attention1"].update(max_generations=4,max_http_attempts=16)
    remaining=selected+groups["attention1"]["cases"]
    original=d.load(ORIGINAL/"plan.json")
    approved=[x for g in original["groups"].values() for x in g["cases"]]
    all_closed=merge_closed(previous["closed_contexts"],closed,approved,remaining)
    require(len(selected)==24 and len(remaining)==25 and len(all_closed)==51,"Exact remaining 24 plus 1")
    require(consumed_g==87 and consumed_h==87,"Exact retained ledger; no unaccounted calls")
    for p in (PREVIOUS/"plan.json",PREVIOUS/"report.json",PREVIOUS/"execution-claim.json",report_path):
        retained[str(p)]=d.sha(p)
    retained.update(previous["tool_hashes"])
    plan=copy.deepcopy(previous)
    plan.update(format="poseidon-approved-unstarted-continuation-v2",parent_continuation=previous["binding"],
        parent_terminal_sha256=d.sha(PREVIOUS/"report.json"),groups=groups,closed_contexts=all_closed,
        planned=25,original_planned=76,historical_passed=sum(x["status"]=="passed" for x in all_closed),
        historical_failed=sum(x["status"]=="failed" for x in all_closed),
        prior_generation_reservations=consumed_g,prior_http_attempts=consumed_h,
        max_new_generation_reservations=100,max_new_http_attempts=400,retained_hashes=retained,
        tool_hashes={str(p):d.sha(p) for p in Path(__file__).parent.glob("*.py")})
    require(consumed_g+100<=plan["max_total_generations"] and consumed_h+400<=plan["max_total_http_attempts"],"No approved budget expansion")
    plan.pop("binding");plan["binding"]=digest(plan);d.verify_parent(plan)
    OUT.mkdir(mode=0o700);(OUT/"plan.json").write_bytes(canonical(plan))
    print(json.dumps(dict(status="prepared",cases=25,binding=plan["binding"],prior_generations=consumed_g,
        prior_http_attempts=consumed_h,new_max_generations=100,new_max_http_attempts=400,
        matrix_deadline=groups["matrix_instance60"]["deadline_wall"],historical_passed=42,historical_failed=9)),flush=True)
def run():
    plan=d.load(OUT/"plan.json");d.verify_parent(plan);no_running_previous()
    with (OUT/"tls-at-run.json").open("x") as f:json.dump(d.tls_check(),f,sort_keys=True)
    with (OUT/"execution-claim.json").open("x") as f:
        json.dump(dict(pid=os.getpid(),started=time.time(),binding=plan["binding"]),f)
    groups=[];started=time.monotonic()
    def save(status):
        (OUT/"report.json").write_bytes(canonical(dict(format="poseidon-approved-unstarted-status-v2",
            status=status,binding=plan["binding"],groups=groups,planned=25,original_planned=76,
            historical_passed=42,historical_failed=9,prior_generations=87,prior_http_attempts=87,
            automatic_uncertain_retry=False,automatic_restart=False,seconds=time.monotonic()-started)))
    save("running")
    try:
        for name in ("matrix_instance60","attention1"):
            d.verify_parent(plan);stop=plan["groups"][name]["deadline_wall"]
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
            group.update(exit_code=code,**{k:report[k] for k in ("status","passed","failed","not_run","cumulative_generations","cumulative_http_attempts")})
            save("running")
            if report["status"]!="complete" or code!=0:save("stopped");return code or 1
        save("complete");return 0
    except BaseException:save("stopped");raise
if __name__=="__main__":
    if sys.argv[1:]==["--prepare"]:prepare()
    else:
        require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned live environment")
        if sys.argv[1:]==["--run"]:raise SystemExit(run())
        elif len(sys.argv)==4 and sys.argv[1]=="--group":raise SystemExit(d.run_group(sys.argv[2],float(sys.argv[3])))
        else:raise SystemExit("Explicit prepare/run/group only")
