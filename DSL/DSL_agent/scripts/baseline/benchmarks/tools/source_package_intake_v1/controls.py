"""Affected component, source and bundle controls in fixed existing Python."""
import sys,json,shlex,resource,unittest,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(ROOT/"scripts")]
if "--inside"not in sys.argv:
    sys.path.insert(0,str(BASE/"benchmarks/tools/static_broadcast_live_v1"))
    import run as live
    live.no_active()
    from hecate_python_env import enter_nix,VENV
    command='PYTHONPATH='+shlex.quote(str(BASE)+":"+str(ROOT/"scripts"))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*(["--corrected"]if"--corrected"in sys.argv else[])])
    raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
assert tuple(sys.version_info[:3])==(3,10,14)
names=("test_pytorch_source_package_intake","test_pytorch_source_package",
 "test_application_source_versions","test_application_capabilities","test_pytorch_input_binding",
 "test_application_component","test_application_diagnostics","test_staged_application_reuse",
 "test_application_model_source","test_candidate_bundle")
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
def ids(value):
 if isinstance(value,unittest.TestSuite):return[name for item in value for name in ids(item)]
 return[value.id()]
test_ids=ids(suite);assert len(test_ids)==len(set(test_ids))
started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(format="poseidon-source-package-intake-controls-v1",tests=result.testsRun,
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 status="passed"if result.wasSuccessful()else"failed",seconds=time.monotonic()-started,
 test_ids=test_ids,python=list(sys.version_info[:3]),
 source_hashes={name:hashlib.sha256((BASE/(name+".py")).read_bytes()).hexdigest()for name in names},
 provider_calls=0,real_capture_calls=0,compiler_calls=0,encrypted_calls=0,
 scope="Static source preparation, snapshots, provenance and injected loader/Backend/Provider; no isolated source capture or FHE.")
with(HERE/("controls-corrected.json"if"--corrected"in sys.argv else"controls-first.json")).open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({key:report[key]for key in("tests","passed","failed","errors","skipped","status","seconds")}),flush=True)
raise SystemExit(not result.wasSuccessful())
