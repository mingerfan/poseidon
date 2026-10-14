"""Bounded staging-only Torch tests in the existing bubblewrap environment."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
LIVE=WORK/"results/agent-approved-unstarted-continuation-r1141"
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from process_resource_limits import limits
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p):
 for name,want in p["source_hashes"].items():
  if sha(ROOT/name)!=want:raise SystemExit("Frozen production source mismatch")
if len(sys.argv)!=2:raise SystemExit("Usage: run_isolated.py FRESH_RESULT_NAME")
name=sys.argv[1]
if not name.startswith("agent-alias-capture-tests-r") or not name.removeprefix("agent-alias-capture-tests-r").isdigit():raise SystemExit("Result name")
out=WORK/"results"/name
if out.exists():raise SystemExit("Fresh result directory required")
plan=json.loads((LIVE/"plan.json").read_text());check(plan)
installation=json.loads((WORK/"results/hecate-python-install-besyznhj/report.json").read_text())
library=installation["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
if not store.is_dir():raise SystemExit("Existing portable Nix store unavailable")
if any(not Path(v).is_relative_to("/nix/store") or not (store/Path(v).relative_to("/nix/store")).is_dir()
       for v in library.split(":")):raise SystemExit("Pinned library paths")
venv=WORK/"venvs/hecate-2.0.1-cpu"
out.mkdir()
source={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*.py")}
for rel,expected in source.items():
    snapshot=out/"source-snapshot"/rel
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    raw=(ROOT/rel).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise SystemExit("Source changed before snapshot")
    snapshot.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
 "--ro-bind",str(store),"/nix/store","--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu","--ro-bind",str(venv),str(venv),"--proc","/proc","--dev","/dev","--tmpfs","/tmp",
 "--dir","/app","--ro-bind",str(HERE/"targets"),"/staging","--ro-bind",str(HERE/"originals"),"/originals",
 "--dir","/tests","--ro-bind",str(HERE/"test_capture.py"),"/tests/test_capture.py",
 "--ro-bind",str(HERE/"inside_tests.py"),"/tests/inside_tests.py","--bind",str(out),"/out"]
# Mount only existing Agent code/configuration files, never the checkout root or .env.
for p in sorted((ROOT/"scripts/baseline").iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):
  cmd+=["--ro-bind",str(p),"/app/"+p.name]
variables=dict(PATH="/nonexistent",LD_LIBRARY_PATH=library,POSEIDON_PLATFORM="aarch64-linux",
 POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",HOME="/nonexistent",PYTHONPATH="/staging:/tests:/app",
 PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8")
for k,v in variables.items():cmd+=["--setenv",k,v]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside_tests.py"]
started=time.monotonic()
with (out/"runner.log").open("x") as log:
 r=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],
  env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"must-not-survive"},
  stdout=log,stderr=subprocess.STDOUT,timeout=130,close_fds=True,preexec_fn=limits)
check(plan)
if any(sha(ROOT/k)!=v for k,v in source.items()):raise SystemExit("Staging source changed while tests ran")
receipt=dict(format="poseidon-staged-alias-capture-sandbox-v1",exit_code=r.returncode,seconds=time.monotonic()-started,
 source_hashes=source,production_freeze_checks=len(plan["source_hashes"]),installation_record_sha256=sha(WORK/"results/hecate-python-install-besyznhj/report.json"),
 host_network_namespace=str(Path("/proc/self/ns/net").readlink()),clearenv=True,unshare_all=True,
 limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
 read_only_store=str(store),backend_changed=False,paid_calls=0,new_dependencies=0,nix_launcher_calls=0)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=r.returncode,production_freeze_checks=receipt["production_freeze_checks"])))
if (out/"report.json").is_file():
 data=json.loads((out/"report.json").read_text())
 print(json.dumps({k:data[k] for k in ("status","tests","failed","errors","skipped","runtime")}))
if r.returncode:print((out/"runner.log").read_text()[-24000:])
sys.exit(r.returncode)
