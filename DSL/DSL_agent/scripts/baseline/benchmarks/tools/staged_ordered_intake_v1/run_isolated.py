"""Staging-only isolated source/weight/reference tests; no second Nix launch."""
import hashlib,json,os,subprocess,sys,time
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
if len(sys.argv)!=2:raise SystemExit("Usage: run_isolated.py FRESH_RESULT_NAME")
name=sys.argv[1]
if not name.startswith("agent-ordered-intake-tests-r") or not name.removeprefix("agent-ordered-intake-tests-r").isdigit():
 raise SystemExit("Result name")
out=WORK/"results"/name
if out.exists():raise SystemExit("Fresh result directory required")
plan=json.loads((LIVE/"plan.json").read_text());check(plan)
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=json.loads(installation.read_text())["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
if not store.is_dir():raise SystemExit("Existing portable Nix store unavailable")
if any(not Path(p).is_relative_to("/nix/store")
       or not (store/Path(p).relative_to("/nix/store")).is_dir() for p in library.split(":")):
 raise SystemExit("Pinned library paths")
venv=WORK/"venvs/hecate-2.0.1-cpu"
out.mkdir(mode=0o700)
source={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*") if p.is_file()
        and p.suffix in (".py",".json",".npy")}
for rel,want in source.items():
 p=out/"source-snapshot"/rel;p.parent.mkdir(parents=True,exist_ok=True)
 raw=(ROOT/rel).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=want:raise SystemExit("Staging source changed before snapshot")
 p.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
 "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
 "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
 "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app",
 "--ro-bind",str(HERE/"originals"),"/originals","--dir","/tests",
 "--ro-bind",str(HERE/"fixtures"),"/fixtures","--bind",str(out),"/out"]
for p in sorted((ROOT/"scripts/baseline").iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):cmd+=["--ro-bind",str(p),"/app/"+p.name]
for p in sorted((HERE/"targets").iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):cmd+=["--ro-bind",str(p),"/app/"+p.name]
for name in ("test_capture_compatibility.py","test_intake.py","test_facade_package.py","inside_tests.py"):
 cmd+=["--ro-bind",str(HERE/name),"/tests/"+name]
# Existing compatibility tests resolve their repository fixtures by their native
# source location. Bind individual Agent files and public case directories only.
for p in sorted((ROOT/"scripts/baseline").iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):
  cmd+=["--ro-bind",str(p),str(p)]
for p in sorted((HERE/"targets").iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):
  cmd+=["--ro-bind",str(p),str(ROOT/"scripts/baseline"/p.name)]
for case in ("trusted-pytorch-arithmetic-v8","trusted-pytorch-common-v11",
             "trusted-pytorch-arithmetic-shape-v9","trusted-pytorch-operator-shape-v10"):
 p=ROOT/"scripts/baseline/cases"/case
 if p.is_dir():cmd+=["--ro-bind",str(p),str(p)]
variables=dict(PATH="/nonexistent",LD_LIBRARY_PATH=library,POSEIDON_PLATFORM="aarch64-linux",
 POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",HOME="/nonexistent",PYTHONPATH="/tests:/app",
 PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="1",
 OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8")
for k,v in variables.items():cmd+=["--setenv",k,v]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside_tests.py"]
started=time.monotonic()
with (out/"runner.log").open("x") as log:
 run=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],
  env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"must-not-survive"},
  stdout=log,stderr=subprocess.STDOUT,timeout=130,close_fds=True,preexec_fn=limits)
check(plan)
if any(sha(ROOT/k)!=v for k,v in source.items()):raise SystemExit("Staging source changed while tests ran")
receipt=dict(format="poseidon-staged-ordered-intake-sandbox-v1",exit_code=run.returncode,
 seconds=time.monotonic()-started,source_hashes=source,production_freeze_checks=len(plan["source_hashes"]),
 installation_record_sha256=sha(installation),host_network_namespace=str(Path("/proc/self/ns/net").readlink()),
 clearenv=True,unshare_all=True,read_only_fixtures=True,read_only_store=str(store),
 limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
 backend_changed=False,paid_calls=0,new_dependencies=0,nix_launcher_calls=0)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=run.returncode,production_freeze_checks=len(plan["source_hashes"]))))
if (out/"report.json").exists():
 r=json.loads((out/"report.json").read_text())
 print(json.dumps({k:r[k] for k in ("status","tests","failed","errors","skipped","runtime")}))
if run.returncode:print((out/"runner.log").read_text()[-16000:])
sys.exit(run.returncode)
