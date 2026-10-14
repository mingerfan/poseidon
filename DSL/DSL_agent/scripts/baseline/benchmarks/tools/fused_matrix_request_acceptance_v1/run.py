"""Isolated preparation of the reviewed source cases; no compiler or Provider."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE=Path(__file__).resolve().parent
BASE=ROOT/"scripts/baseline"
SOURCE=HERE.parent/"fused_matrix_source_draft_v1"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
sys.path.insert(0,str(BASE))
from process_resource_limits import limits

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def check():
    plan=read(HERE/"plan.json")
    acceptance=read(SOURCE/"acceptance.json")
    require(sha(SOURCE/"acceptance.json")==plan["source_acceptance_sha256"] and
            acceptance["binding"]==plan["acceptance_binding"],"Reviewed source acceptance")
    parent=read(SOURCE/"plan.json")
    require(all(sha(ROOT/p)==h for p,h in parent["production_hashes"].items()),"Frozen production changed")
    require(parent["targets"]==plan["target_hashes"] and
            all(sha(SOURCE/"targets"/p)==h for p,h in plan["target_hashes"].items()),"Reviewed draft changed")
    for name,entry in parent["parents"].items():
        folder=SOURCE.parent/name
        require(sha(folder/"plan.json")==entry["plan_sha256"] and
                all(sha(folder/"targets"/p)==h for p,h in entry["targets"].items()),"Prerequisite changed")
    for name in ("protected-backend.json","protected-references.json"):
        record=read(RESULTS/"agent-common-pytorch-offline-prepared-r1040"/name)
        require(all(sha(ROOT/p)==h for p,h in record.items()),"Protected backend or reference changed")
    prior=RESULTS/plan["source_result"]
    require(sha(prior/"report.json")==plan["source_report_sha256"],"Retained report changed")
    require(read(prior/"report.json")["source_models"]==9,"Reviewed source inventory")
    return plan,parent

if len(sys.argv) not in (2,4):raise SystemExit("Usage: run.py FRESH_RESULT_NAME [--prepared PREVIOUS_RESULT_NAME]")
name=sys.argv[1];prefix="agent-fused-matrix-requests-r"
prepared=None
if len(sys.argv)==4:
    require(sys.argv[2]=="--prepared" and sys.argv[3].startswith(prefix) and
            sys.argv[3][len(prefix):].isdigit(),"Explicit prepared developer input")
    prepared=RESULTS/sys.argv[3]
require(name.startswith(prefix) and name[len(prefix):].isdigit(),"Fresh numeric result name")
out=RESULTS/name
require(not out.exists(),"Fresh result directory")
plan,parent=check()
prior=RESULTS/plan["source_result"]
inputs={str(p):sha(p) for p in prior.rglob("*") if p.is_file()}
prepared_inputs={}
if prepared is not None:
    prior_receipt=read(prepared/"sandbox.json")
    require(prior_receipt["staged_implementation_hashes"]==plan["target_hashes"] and
            prior_receipt["source_acceptance_binding"]==plan["acceptance_binding"],"Same prepared implementation")
    previous=read(prepared/"report.json")
    require(previous["source_models"]==9 and previous["planned_source_models"]==9 and
            previous["preparation_failures"]==[] and
            {r["case"] for r in previous["request_rows"]}==set(plan["source_cases"]),
            "Same prepared source inventory")
    prepared_inputs={str(p):sha(p) for p in prepared.rglob("*") if p.is_file()}
source_files={str(p.relative_to(ROOT)):sha(p) for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py",".json")}
installation=WORK/"results/hecate-python-install-besyznhj/report.json"
library=read(installation)["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
venv=WORK/"venvs/hecate-2.0.1-cpu"
require(all(Path(p).is_relative_to("/nix/store") and
            (store/Path(p).relative_to("/nix/store")).is_dir() for p in library.split(":")),
        "Pinned runtime libraries")
out.mkdir(mode=0o700)
for rel,want in source_files.items():
    raw=(ROOT/rel).read_bytes();require(hashlib.sha256(raw).hexdigest()==want,"Source changed during snapshot")
    path=out/"source-snapshot"/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
     "--ro-bind",str(store),"/nix/store","--ro-bind",str(venv),str(venv),
     "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
     "--proc","/proc","--dev","/dev","--tmpfs","/tmp","--dir","/app","--dir","/tests",
     "--ro-bind",str(prior),"/captures","--bind",str(out),"/out"]
if prepared is not None:cmd+=["--ro-bind",str(prepared),"/prepared"]
for p in sorted(BASE.iterdir()):
    if p.is_file() and p.suffix in (".py",".json"):
        cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(p)]
for folder in [SOURCE.parent/n/"targets" for n in parent["parents"]]+[SOURCE/"targets"]:
    for p in sorted(folder.iterdir()):
        cmd+=["--ro-bind",str(p),"/app/"+p.name,"--ro-bind",str(p),str(BASE/p.name)]
for item in ("inside_tests.py","plan.json"):cmd+=["--ro-bind",str(HERE/item),"/tests/"+item]
variables=dict(PATH="/nonexistent",HOME="/nonexistent",LD_LIBRARY_PATH=library,
    POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",
    PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",LANG="C.UTF-8",
    OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
for key,value in variables.items():cmd+=["--setenv",key,value]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside_tests.py",
      "privacy" if prepared is not None else "all"]
started=time.monotonic()
with (out/"runner.log").open("x") as log:
    proc=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],
        env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"synthetic-test-marker"},
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        timeout=130,close_fds=True,preexec_fn=limits)
check()
require(all(sha(path)==want for path,want in inputs.items()),"Retained evidence changed")
require(all(sha(path)==want for path,want in prepared_inputs.items()),"Prepared evidence changed")
require(all(sha(ROOT/rel)==want for rel,want in source_files.items()),"Request tests changed while running")
receipt=dict(format="poseidon-fused-matrix-request-sandbox-v1",exit_code=proc.returncode,
    seconds=time.monotonic()-started,source_hashes=source_files,retained_hashes=inputs,
    prepared_inputs=prepared_inputs,prepared_result=str(prepared) if prepared is not None else None,
    source_acceptance_binding=plan["acceptance_binding"],installation_sha256=sha(installation),
    staged_implementation_hashes=plan["target_hashes"],production_checks=704,
    protected_backend_checks=648,protected_reference_checks=6,clearenv=True,unshare_all=True,
    host_network_namespace=str(Path("/proc/self/ns/net").readlink()),
    limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,wall_seconds=120,file_bytes=16*1024**2),
    source_capture_calls=0,compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0)
(out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=proc.returncode,seconds=receipt["seconds"])))
if (out/"report.json").exists():
    report=read(out/"report.json")
    print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models","planned_source_models","preparation_failures")}))
if proc.returncode:print((out/"runner.log").read_text()[-18000:])
raise SystemExit(proc.returncode)
