"""Run preparation controls in the fixed existing Python; no source execution."""
import sys, json, shlex, resource, unittest, hashlib, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside" not in sys.argv:
    sys.path.insert(0,str(BASE/"benchmarks/tools/static_broadcast_live_v1"))
    import run as live
    live.no_active()
    from hecate_python_env import enter_nix,VENV
    command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*([flag for flag in("--corrected","--snapshot")if flag in sys.argv])])
    raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
assert tuple(sys.version_info[:3])==(3,10,14)
names=("test_pytorch_source_package",)
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
def ids(value):
    if isinstance(value,unittest.TestSuite):
        return [name for item in value for name in ids(item)]
    return [value.id()]
test_ids=ids(suite)
assert len(set(test_ids))==len(test_ids)
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(format="poseidon-source-package-preparation-controls-v1",
 tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,status="passed"if result.wasSuccessful()else"failed",
 test_ids=test_ids,python=list(sys.version_info[:3]),python_executable=sys.executable,
 source_hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()for name in(
     "pytorch_source_package.py","test_pytorch_source_package.py")},
 api_calls=0,source_capture_calls=0,compiler_calls=0,encrypted_calls=0,
 scope="Explicit source files, package/import declarations, hashes, bounded AST and secure file reading. No source import or execution.")
with(HERE/("controls-snapshot.json"if"--snapshot"in sys.argv else"controls-corrected.json"if"--corrected"in sys.argv else"controls-first.json")).open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({key:report[key]for key in("tests","passed","failed","errors","skipped","status","seconds","python")}),flush=True)
raise SystemExit(not result.wasSuccessful())
