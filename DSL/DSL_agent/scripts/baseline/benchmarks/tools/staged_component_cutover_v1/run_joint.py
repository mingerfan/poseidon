"""Compose staged Agent changes in an isolated, bounded developer test."""
import hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
TOOLS=HERE.parent
ORDERED=TOOLS/"staged_ordered_intake_v1"
FEEDBACK=TOOLS/"staged_binding_feedback_v1"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
LIVE=WORK/"results/agent-approved-unstarted-continuation-r1174"
OLD=WORK/"results/agent-approved-pending-continuation-r1099"
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from process_resource_limits import limits
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def freeze(plan):
    checked={}
    for field,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        for rel,want in plan[field].items():
            if sha(base/rel)!=want:raise SystemExit("Frozen source/evidence mismatch")
        checked[field]=len(plan[field])
    return checked
if len(sys.argv) not in (2,3) or not re.fullmatch(r"agent-joint-component-tests-r[0-9]+",sys.argv[1]):
    raise SystemExit("Usage: run_joint.py agent-joint-component-tests-rN [--request-plan]")
if len(sys.argv)==3 and sys.argv[2]!="--request-plan":raise SystemExit("Registered staging scope only")
request_preflight=len(sys.argv)==3
worker="inside_request_plan.py" if request_preflight else "inside_tests.py"
out=WORK/"results"/sys.argv[1]
if out.exists():raise SystemExit("Fresh result directory")
plan=read(LIVE/"plan.json");before=freeze(plan)
joint=read(HERE/"plan.json")
if joint["stage"]!="not_applied":raise SystemExit("Staging only")
overlays={name:Path(item["source"]) for name,item in joint["source_overlays"].items()}
for name,path in overlays.items():
    if name!=path.name or path.parent not in (FEEDBACK/"targets",ORDERED/"targets") or path.is_symlink():
        raise SystemExit("Joint source path")
    if sha(path)!=joint["source_overlays"][name]["sha256"]:raise SystemExit("Staged target changed")
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=read(installation)["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
venv=WORK/"venvs/hecate-2.0.1-cpu"
if not store.is_dir():raise SystemExit("Existing store missing")
for path in library.split(":"):
    if not Path(path).is_relative_to("/nix/store") or not (store/Path(path).relative_to("/nix/store")).is_dir():
        raise SystemExit("Pinned library unavailable")
out.mkdir(mode=0o700)
test_sources=[HERE/worker,HERE/"test_joint_component.py",FEEDBACK/"test_staged.py",
              ORDERED/"test_capture_compatibility.py",ORDERED/"test_intake.py",ORDERED/"test_facade_package.py",Path(__file__),HERE/"plan.json"]
source_paths=set(overlays.values())|set(test_sources)|{p for p in (ORDERED/"fixtures").rglob("*")
                                                    if p.is_file() and p.suffix in (".py",".json",".npy")}
sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
for rel,want in sources.items():
    raw=(ROOT/rel).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=want:raise SystemExit("Source snapshot mismatch")
    path=out/"source-snapshot"/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
     "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
     "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
     "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/tests",
     "--ro-bind",str(ORDERED/"originals"),"/originals","--ro-bind",str(ORDERED/"fixtures"),"/fixtures",
     "--bind",str(out),"/out"]
for p in sorted((ROOT/"scripts/baseline").iterdir()):
    if p.is_file() and p.suffix in (".py",".json"):
        cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(p)]
for name,path in sorted(overlays.items()):
    cmd+=["--ro-bind",str(path),"/app/"+name,"--ro-bind",str(path),str(ROOT/"scripts/baseline"/name),
          "--ro-bind",str(path),str(path)]
for filename in ("test_capture_compatibility.py","test_intake.py","test_facade_package.py"):
    cmd+=["--ro-bind",str(ORDERED/filename),"/tests/"+filename]
for filename in (worker,"test_joint_component.py"):
    cmd+=["--ro-bind",str(HERE/filename),"/tests/"+filename]
cmd+=["--ro-bind",str(FEEDBACK/"test_staged.py"),str(FEEDBACK/"test_staged.py")]
for case in ("trusted-pytorch-arithmetic-v8","trusted-pytorch-common-v11",
             "trusted-pytorch-arithmetic-shape-v9","trusted-pytorch-operator-shape-v10"):
    p=ROOT/"scripts/baseline/cases"/case
    if p.is_dir():cmd+=["--ro-bind",str(p),str(p)]
readonly={OLD/"functional15/report.json"}
old=read(OLD/"functional15/report.json")
for row in old["rows"]:
    task=OLD/"functional15"/row["id"]/"tasks"/row["task_id"]
    readonly.add(task/"task.json")
    for index in range(row["result"]["generations"]):
        readonly.add(task/("validation-%d.json"%index))
        response=task/("response-%d.json"%index)
        if response.is_file():readonly.add(response)
    if row["status"]=="passed":readonly.add(Path(row["package"])/"candidate.py")
readonly.update((ROOT/"scripts/baseline/golden_cases/native_arrays").glob("*.py"))
evidence={}
for p in sorted(readonly):
    if p.suffix not in (".json",".py") or p.is_symlink() or not p.is_file() or p.stat().st_size>16*1024**2:
        raise SystemExit("Retained test evidence path")
    evidence[str(p)]=sha(p)
    cmd+=["--ro-bind",str(p),str(p)]
variables=dict(PATH="/nonexistent",LD_LIBRARY_PATH=library,POSEIDON_PLATFORM="aarch64-linux",
    POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",HOME="/nonexistent",PYTHONPATH="/tests:/app",
    PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",LANG="C.UTF-8")
for key,value in variables.items():cmd+=["--setenv",key,value]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/"+worker]
started=time.monotonic()
with (out/"runner.log").open("x") as log:
    run=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],stdout=log,stderr=subprocess.STDOUT,
        env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"noncredential-test-marker"},
        timeout=130,close_fds=True,preexec_fn=limits)
after=freeze(plan)
if any(sha(ROOT/rel)!=want for rel,want in sources.items()):raise SystemExit("Staging source changed during tests")
if any(sha(p)!=want for p,want in evidence.items()):raise SystemExit("Historical test evidence changed")
receipt=dict(format="poseidon-joint-component-sandbox-v1",exit_code=run.returncode,seconds=time.monotonic()-started,
    source_hashes=sources,readonly_evidence_hashes=evidence,freeze_before=before,freeze_after=after,
    installation_record_sha256=sha(installation),source_batch_binding=plan["binding"],
    host_network_namespace=str(Path("/proc/self/ns/net").readlink()),clearenv=True,unshare_all=True,
    limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
    backend_changes=False,provider_calls=0,new_dependencies=0,nix_launcher_calls=0,production_applied=False,test_scope="request_preflight" if request_preflight else "full_component")
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=run.returncode,freeze_after=after)))
if (out/"report.json").exists():
    r=read(out/"report.json");print(json.dumps({k:r[k] for k in ("status","tests","models","failed","errors","skipped","runtime") if k in r}))
if run.returncode:print((out/"runner.log").read_text()[-18000:])
sys.exit(run.returncode)
