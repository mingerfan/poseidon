"""Isolated draft normalization/reference tests using the retained pinned SDK."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);HERE=Path(__file__).resolve().parent
BASE=ROOT/"scripts/baseline";WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
sys.path.insert(0,str(BASE))
from process_resource_limits import limits
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check():
 p=json.loads((ROOT/"docs/baseline/agent-v12-live-proposal-r1232.json").read_text())
 for name,want in p["source_hashes"].items():
  if sha(ROOT/name)!=want:raise RuntimeError("Frozen production changed")
 return p
if len(sys.argv) not in (2,3):raise SystemExit("Usage: run_isolated.py FRESH_RESULT_NAME [reference|source|retained|component]")
name=sys.argv[1];suite=sys.argv[2] if len(sys.argv)==3 else "reference"
if suite not in ("reference","source","retained","component"):raise SystemExit("Explicit test suite")
if not name.startswith("agent-ordered-common-tests-r") or not name.removeprefix("agent-ordered-common-tests-r").isdigit():
 raise SystemExit("Result name")
out=WORK/"results"/name
if out.exists():raise SystemExit("Fresh result directory")
frozen=check();plan=json.loads((HERE/"plan.json").read_text())
for n,i in plan["targets"].items():
 actual=sha(BASE/n) if (BASE/n).exists() else None
 if actual!=i["original_sha256"] or sha(HERE/"targets"/n)!=i["target_sha256"]:raise SystemExit("Draft source binding")
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=json.loads(installation.read_text())["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store";venv=WORK/"venvs/hecate-2.0.1-cpu"
if not store.is_dir():raise SystemExit("Retained store unavailable")
if any(not Path(p).is_relative_to("/nix/store") or not (store/Path(p).relative_to("/nix/store")).is_dir() for p in library.split(":")):
 raise SystemExit("Pinned library paths")
source={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py",".json")}
out.mkdir(mode=0o700)
for name,want in source.items():
 p=out/"source-snapshot"/name;p.parent.mkdir(parents=True,exist_ok=True);raw=(ROOT/name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=want:raise SystemExit("Draft changed while snapshotting")
 p.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
 "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
 "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
 "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/tests","--bind",str(out),"/out"]
for p in sorted(BASE.iterdir()):
 if p.is_file() and p.suffix in (".py",".json"):cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(p)]
for p in sorted((HERE/"targets").iterdir()):
 cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(BASE/p.name)]
for p in (HERE/"test_composition.py",HERE/"test_source_package.py",HERE/"common_models.py",HERE/"inside_tests.py"):
 cmd+=["--ro-bind",str(p),"/tests/"+p.name]
retained=BASE/"benchmarks/tools/staged_ordered_intake_v1"
for p in (retained/"test_intake.py",retained/"test_facade_package.py"):
 cmd+=["--ro-bind",str(p),"/tests/"+p.name]
cmd+=["--ro-bind",str(retained/"fixtures"),"/fixtures"]
cmd+=["--ro-bind",str(BASE/"cases"),str(BASE/"cases")]
for name in ("trusted-pytorch-common-v11",):
 p=BASE/"cases"/name
 if p.is_dir():cmd+=["--ro-bind",str(p),str(p)]
vars=dict(PATH="/nonexistent",LD_LIBRARY_PATH=library,POSEIDON_PLATFORM="aarch64-linux",
 POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",HOME="/nonexistent",PYTHONPATH="/tests:"+str(BASE),
 PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8")
for key,value in vars.items():cmd+=["--setenv",key,value]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside_tests.py",suite]
start=time.monotonic()
with (out/"runner.log").open("x") as log:
 p=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"synthetic-test-marker"},
  stdout=log,stderr=subprocess.STDOUT,timeout=130,close_fds=True,preexec_fn=limits)
check()
if any(sha(ROOT/p)!=h for p,h in source.items()):raise RuntimeError("Draft changed while testing")
receipt=dict(format="poseidon-ordered-common-draft-sandbox-v1",exit_code=p.returncode,seconds=time.monotonic()-start,
 source_hashes=source,production_freeze_checks=len(frozen["source_hashes"]),production_applied=False,
 installation_record_sha256=sha(installation),clearenv=True,unshare_all=True,
 host_network_namespace=str(Path("/proc/self/ns/net").readlink()),
 limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
 backend_changed=False,provider_calls=0,new_dependencies=0,nix_launcher_calls=0)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=p.returncode,seconds=receipt["seconds"])))
if (out/"report.json").exists():
 report=json.loads((out/"report.json").read_text())
 print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","contexts","input_groups","compared_values","runtime")}))
if p.returncode:print((out/"runner.log").read_text()[-18000:])
raise SystemExit(p.returncode)
