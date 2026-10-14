"""One bounded component batch, with evidence separated from real backend work."""
import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
directory=Path(__file__).resolve().parent;started=time.monotonic()
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in (
    "test_task_numerical_assessment","test_task_native_assessment.NativeAssessmentTests",
    "test_task_source_reference"))
with (directory/"controls.stdout").open("x") as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
files=[ROOT/"scripts/baseline"/n for n in (
    "task_numerical_assessment.py","test_task_numerical_assessment.py","test_task_native_assessment.py",
    "task_source_reference.py","test_task_source_reference.py","source_precision_reference.py")]+[Path(__file__)]
report=dict(format="poseidon-native-assessment-controls-v1",
    status="passed" if result.wasSuccessful() else "failed",
    tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
    elapsed_seconds=time.monotonic()-started,
    source_forward_executed=False,encrypted_execution=False,api_calls=0,compiler_calls=0,
    backend_outputs="Injected control fixtures; actual native/FHE acceptance has a separate report",
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
report["binding"]=digest(report)
with (directory/"controls.json").open("x") as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
