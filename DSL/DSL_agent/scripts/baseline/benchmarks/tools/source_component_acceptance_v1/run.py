"""Read-only overlay test driver for real-identity source admission."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
JOINT=HERE.parent/"staged_common_fused_integration_v1"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
sys.path[:0]=[str(JOINT),str(BASE)]
import check
from process_resource_limits import limits
from application_identity import current_identity_manifest
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def hashes(folder):
    return {str(p):sha(p) for p in sorted(folder.rglob("*")) if p.is_file() and not p.is_symlink()}
if len(sys.argv) not in (2,4):raise SystemExit("Usage: run.py agent-source-component-admission-rNEW [--only integrity]")
selection="all"
if len(sys.argv)==4:
    require(sys.argv[2:]==["--only","integrity"],"Explicit affected test selector")
    selection="integrity"
name=sys.argv[1];prefix="agent-source-component-admission-r"
require(name.startswith(prefix) and name[len(prefix):].isdigit(),"Fresh numeric result name")
out=RESULTS/name;require(not out.exists(),"Fresh output directory")
boundary=check.check();require(boundary["status"]=="awaiting_backend_qualification" and not boundary["active_processes"],
                             "Idle verified integration boundary")
joint=read(JOINT/"plan.json")
prepared=RESULTS/"agent-fused-matrix-requests-r1274"
captures=RESULTS/"agent-fused-matrix-intake-r1268"
request_acceptance=read(HERE.parent/"fused_matrix_request_acceptance_v1/acceptance.json")
require(request_acceptance["binding"]==joint["source_acceptance"]["fused_matrix_request_acceptance_v1"]["binding"],
        "Reviewed source/request acceptance")
old=read(prepared/"report.json")
require(old["source_models"]==9 and old["preparation_failures"]==[],"Nine captured requests")
require(shutil.disk_usage(RESULTS).free>=4*1024**3+128*1024**2,"Disk reserve")
require(os.environ.get("POSEIDON_PLATFORM")=="aarch64-linux" and
        os.environ.get("POSEIDON_WORK_ROOT")=="/home/lhohy/poseidon-work",
        "Select the verified ARM platform and work root explicitly")
from platform_config import require_platform
require_platform()
original_identity=current_identity_manifest()
inputs=dict(hashes(prepared),**hashes(captures))
tools=hashes(HERE)
library=read(RESULTS/"hecate-python-install-besyznhj/report.json")["library_path"]
store=WORK/"deps/nix-portable-v012/.nix-portable/nix/store"
venv=WORK/"venvs/hecate-2.0.1-cpu"
require(all(Path(p).is_relative_to("/nix/store") and
            (store/Path(p).relative_to("/nix/store")).is_dir() for p in library.split(":")),
        "Pinned libraries")
plan=dict(format="poseidon-source-admission-execution-plan-v1",combined_plan_binding=joint["binding"],
          targets={n:v["target_sha256"] for n,v in joint["targets"].items()},
          inside_source_hashes=joint["inside_source_hashes"],production_identity_sha256=original_identity["manifest_sha256"],
          backend_identity=original_identity["backend"],host_network_namespace=str(Path("/proc/self/ns/net").readlink()),
          source_cases=[row["case"] for row in old["request_rows"]],
          compiler_calls=0,encrypted_executions=0,provider_calls=0)
out.mkdir(mode=0o700)
(out/"plan.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
snapshot=out/"source-snapshot"
for path,want in tools.items():
    source=Path(path);raw=source.read_bytes();require(hashlib.sha256(raw).hexdigest()==want,"Test source changed")
    target=snapshot/source.relative_to(HERE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
cmd=["/usr/bin/bwrap","--unshare-all","--die-with-parent","--new-session","--cap-drop","ALL","--clearenv",
     "--ro-bind",str(ROOT),str(ROOT),
     "--ro-bind","/home/lhohy/poseidon-work","/home/lhohy/poseidon-work",
     "--ro-bind",str(store),"/nix/store",
     "--ro-bind","/sys/devices/system/cpu","/sys/devices/system/cpu",
     "--proc","/proc","--dev","/dev","--tmpfs","/tmp",
     "--ro-bind",str(prepared),"/prepared","--ro-bind",str(captures),"/captures",
     "--bind",str(out),"/out","--dir","/tests",
     "--ro-bind",str(out/"plan.json"),"/tests/plan.json",
     "--ro-bind",str(snapshot/"inside.py"),"/tests/inside.py"]
# A temporary mount tree permits the two declared new modules to exist without
# writing placeholders into the retained checkout. All final entries are read-only.
cmd+=["--tmpfs",str(BASE)]
for source in sorted(BASE.iterdir()):
    cmd+=["--ro-bind",str(source),str(source)]
for name in sorted(joint["targets"]):
    cmd+=["--ro-bind",str(JOINT/"targets"/name),str(BASE/name)]
cmd+=["--remount-ro",str(BASE)]
variables=dict(PATH="/nonexistent",HOME="/nonexistent",LD_LIBRARY_PATH=library,
    POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work",
    PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",LANG="C.UTF-8",
    OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
for k,v in variables.items():cmd+=["--setenv",k,v]
cmd+=["--chdir","/out","--",str(venv/"bin/python"),"-B","/tests/inside.py",selection]
started=time.monotonic();code=None
try:
    with (out/"runner.log").open("x") as log:
        proc=subprocess.run(["/usr/bin/timeout","-k","3s","120s",*cmd],
            env={"PATH":"/usr/bin:/bin","POSEIDON_PROBE_SECRET":"synthetic-test-marker"},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            timeout=130,close_fds=True,preexec_fn=limits)
        code=proc.returncode
finally:
    protected=check.check()
    require(all(sha(path)==h for path,h in inputs.items()),"Original captures/request evidence changed")
    require(all(sha(path)==h for path,h in tools.items()),"Admission test source changed")
    size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    require(size<=128*1024**2,"Bounded admission test output")
    receipt=dict(format="poseidon-source-admission-sandbox-v1",exit_code=code,
        seconds=time.monotonic()-started,source_hashes=tools,retained_hashes=inputs,
        combined_plan_binding=joint["binding"],combined_sources=706,production_checks=704,
        protected_backend_checks=648,protected_reference_checks=6,
        boundary=protected,clearenv=True,unshare_all=True,implementation_overlay_readonly=True,
        new_bytes=size,limits=dict(address_space_bytes=4*1024**3,cpu_seconds=150,
            wall_seconds=120,file_bytes=16*1024**2,new_output_bytes=128*1024**2),
        source_factory_calls=0,compiler_calls=0,encrypted_executions=0,provider_calls=0,new_dependencies=0)
    (out/"sandbox.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),exit_code=code,seconds=time.monotonic()-started)))
if (out/"report.json").exists():
    report=read(out/"report.json")
    print(json.dumps({k:report[k] for k in ("status","tests","failed","errors","skipped","source_models","identity_sha256")}))
if code:print((out/"runner.log").read_text()[-18000:])
raise SystemExit(code if code is not None else 1)
