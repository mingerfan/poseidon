"""Bounded real intake-boundary test using only the existing portable store."""
import hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
LIVE=WORK/"results/agent-approved-unstarted-continuation-r1174"
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from process_resource_limits import limits
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(plan):
    for rel,want in plan["source_hashes"].items():
        if sha(ROOT/rel)!=want:raise SystemExit("Frozen production source mismatch")
if len(sys.argv)!=2:raise SystemExit("Usage: run_capture_boundary.py FRESH_RESULT_NAME")
name=sys.argv[1]
if not re.fullmatch(r"agent-ordered-intake-boundary-r[0-9]+",name):raise SystemExit("Result name")
out=WORK/"results"/name
if out.exists():raise SystemExit("Fresh result directory required")
plan=json.loads((LIVE/"plan.json").read_text());check(plan)
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=json.loads(installation.read_text())["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
venv=WORK/"venvs/hecate-2.0.1-cpu"
if not store.is_dir():raise SystemExit("Existing store unavailable")
for path in library.split(":"):
    if not Path(path).is_relative_to("/nix/store") or not (store/Path(path).relative_to("/nix/store")).is_dir():
        raise SystemExit("Pinned library unavailable")
binary=Path("/usr/bin/bwrap")
dependencies=subprocess.run(["/usr/bin/ldd",str(binary)],capture_output=True,text=True,check=True,timeout=5).stdout
libraries=sorted(set(re.findall(r"(/[^\s()]+)",dependencies)))
if not libraries or any(not Path(p).is_file() or not p.startswith(("/lib/","/usr/lib/")) for p in libraries):
    raise SystemExit("Unexpected bwrap dependency")
os_identity={str(binary):sha(binary),**{p:sha(p) for p in libraries}}
out.mkdir(mode=0o700)
source={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*")
        if p.is_file() and p.suffix in (".py",".json",".npy")}
for rel,want in source.items():
    raw=(ROOT/rel).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=want:raise SystemExit("Staging snapshot mismatch")
    snapshot=out/"source-snapshot"/rel;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
     "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
     "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
     "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app",
     "--ro-bind",str(HERE/"fixtures"),"/fixtures","--bind",str(out),str(out)]
cmd+=["--ro-bind",str(binary.resolve()),str(binary)]
for path in libraries:cmd+=["--ro-bind",str(Path(path).resolve()),path]
for p in sorted((ROOT/"scripts/baseline").iterdir()):
    if p.is_file() and p.suffix in (".py",".json"):cmd+=["--ro-bind",str(p),"/app/"+p.name]
for p in sorted((HERE/"targets").iterdir()):
    if p.is_file() and p.suffix in (".py",".json"):cmd+=["--ro-bind",str(p),"/app/"+p.name]
cmd+=["--ro-bind",str(HERE/"inside_boundary.py"),"/app/inside_boundary.py"]
variables=dict(PATH="/usr/bin:/bin",LD_LIBRARY_PATH=library,HECATE_PYTHON_LIBRARY_PATH=library,
    POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",
    POSEIDON_BOUNDARY_RESULTS=str(out),HOME="/nonexistent",PYTHONPATH="/app",
    PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",LANG="C.UTF-8",POSEIDON_PROBE_SECRET="noncredential-test-marker")
for key,value in variables.items():cmd+=["--setenv",key,value]
cmd+=["--chdir",str(out),"--",str(venv/"bin/python"),"-B","/app/inside_boundary.py"]
started=time.monotonic()
with (out/"runner.log").open("x") as log:
    run=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],env={"PATH":"/usr/bin:/bin"},
        stdout=log,stderr=subprocess.STDOUT,timeout=130,close_fds=True,preexec_fn=limits)
check(plan)
for p,want in os_identity.items():
    if sha(p)!=want:raise SystemExit("Existing OS isolation tool changed")
for rel,want in source.items():
    if sha(ROOT/rel)!=want:raise SystemExit("Staging source changed")
receipt=dict(format="poseidon-staged-ordered-boundary-sandbox-v1",exit_code=run.returncode,
    seconds=time.monotonic()-started,source_hashes=source,production_freeze_checks=len(plan["source_hashes"]),
    isolation_os_hashes=os_identity,installation_record_sha256=sha(installation),
    host_network_namespace=str(Path("/proc/self/ns/net").readlink()),clearenv=True,unshare_all=True,
    actual_nested_capture_boundary=True,read_only_store=str(store),new_dependencies=0,nix_launcher_calls=0,
    limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
    paid_calls=0,backend_changed=False)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=run.returncode,production_freeze_checks=len(plan["source_hashes"]))))
if (out/"report.json").exists():
    r=json.loads((out/"report.json").read_text())
    print(json.dumps({k:r[k] for k in ("status","source_capture_runs","wrong_platform_rejections","source_packages_reloaded","runtime")}))
if run.returncode:print((out/"runner.log").read_text()[-14000:])
sys.exit(run.returncode)
