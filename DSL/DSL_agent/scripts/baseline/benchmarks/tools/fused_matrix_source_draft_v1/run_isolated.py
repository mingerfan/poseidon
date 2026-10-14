"""Finite isolated source/provenance checks using the retained CPU SDK."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);HERE=Path(__file__).resolve().parent;BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
sys.path.insert(0,str(BASE))
from process_resource_limits import limits
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    plan=json.loads((HERE/"plan.json").read_text())
    for name,want in plan["production_hashes"].items():
        if sha(ROOT/name)!=want:raise RuntimeError("Production binding changed")
    for name,record in plan["parents"].items():
        parent=HERE.parent/name
        if sha(parent/"plan.json")!=record["plan_sha256"]:raise RuntimeError("Parent plan changed")
        for target,want in record["targets"].items():
            if sha(parent/"targets"/target)!=want:raise RuntimeError("Parent target changed")
    for name,want in plan["targets"].items():
        if sha(HERE/"targets"/name)!=want:raise RuntimeError("Source draft changed")
    for name in ("protected-backend.json","protected-references.json"):
        data=json.loads((WORK/"results/agent-common-pytorch-offline-prepared-r1040"/name).read_text())
        if any(sha(ROOT/n)!=w for n,w in data.items()):raise RuntimeError("Protected implementation changed")
    return plan
if len(sys.argv) not in (2,3):raise SystemExit("Usage: run_isolated.py FRESH_RESULT_NAME [source|affected]")
selected=sys.argv[2] if len(sys.argv)==3 else "source"
if selected not in ("source","affected","legacy_rejection"):raise SystemExit("Explicit test selection")
name=sys.argv[1]
if not name.startswith("agent-fused-matrix-intake-r") or not name.removeprefix("agent-fused-matrix-intake-r").isdigit():
    raise SystemExit("Fresh numeric result name")
out=WORK/"results"/name
if out.exists():raise SystemExit("Fresh result directory required")
plan=check()
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=json.loads(installation.read_text())["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store";venv=WORK/"venvs/hecate-2.0.1-cpu"
if any(not Path(p).is_relative_to("/nix/store") or not (store/Path(p).relative_to("/nix/store")).is_dir()
       for p in library.split(":")):raise SystemExit("Pinned library path")
sources={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py",".json")}
out.mkdir(mode=0o700)
for name,want in sources.items():
    path=out/"source-snapshot"/name;path.parent.mkdir(parents=True,exist_ok=True);raw=(ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=want:raise RuntimeError("Source changed while snapshotting")
    path.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
     "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
     "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu","--proc","/proc","--dev","/dev",
     "--tmpfs","/tmp","--dir","/app","--dir","/tests","--bind",str(out),"/out"]
for p in sorted(BASE.iterdir()):
    if p.is_file() and p.suffix in (".py",".json"):
        cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(p)]
folders=[HERE.parent/name/"targets" for name in plan["parents"]]+[HERE/"targets"]
# The prerequisite insertion order is part of the draft. The fused source
# overrides are last; unchanged modules retain their original byte identity.
for folder in folders:
    for p in sorted(folder.iterdir()):
        cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(BASE/p.name)]
cmd+=["--ro-bind",str(HERE.parent/"staged_ordered_intake_v1/fixtures/linear"),"/compat-v12",
      "--ro-bind",str(HERE.parent/"staged_ordered_common_v1/fixtures/alpha"),"/compat-v13",
      "--ro-bind",str(HERE.parent/"fused_matrix_intake_draft_v1/test_fused.py"),"/tests/test_fused.py"]
for name in ("source_models.py","source_helpers.py","test_source_intake.py","inside_tests.py"):
    cmd+=["--ro-bind",str(HERE/name),"/tests/"+name]
vars=dict(PATH="/nonexistent",HOME="/nonexistent",LD_LIBRARY_PATH=library,
          POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",
          PYTHONPATH="/tests:"+str(BASE),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
          OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",LANG="C.UTF-8")
for key,value in vars.items():cmd+=["--setenv",key,value]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside_tests.py",selected]
start=time.monotonic()
with (out/"runner.log").open("x") as log:
    process=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],
             env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"synthetic-test-marker"},
             stdout=log,stderr=subprocess.STDOUT,timeout=130,close_fds=True,preexec_fn=limits)
check()
if any(sha(ROOT/n)!=h for n,h in sources.items()):raise RuntimeError("Source draft changed during tests")
receipt=dict(format="poseidon-fused-matrix-intake-sandbox-v1",exit_code=process.returncode,seconds=time.monotonic()-start,
             source_hashes=sources,production_checks=len(plan["production_hashes"]),
             prerequisite_checks=sum(len(r["targets"]) for r in plan["parents"].values()),
             protected_backend_checks=648,protected_reference_checks=6,
             installation_sha256=sha(installation),clearenv=True,unshare_all=True,
             host_network_namespace=str(Path("/proc/self/ns/net").readlink()),
             limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,file_bytes=16*1024**2,wall_seconds=120),
             compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0,nix_launcher_calls=0)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=process.returncode,seconds=receipt["seconds"])))
if (out/"report.json").exists():
    report=json.loads((out/"report.json").read_text())
    print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models","input_groups","compared_values","runtime")}))
if process.returncode:print((out/"runner.log").read_text()[-22000:])
raise SystemExit(process.returncode)
