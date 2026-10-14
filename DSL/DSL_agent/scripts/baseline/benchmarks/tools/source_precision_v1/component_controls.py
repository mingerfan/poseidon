import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
directory=Path(__file__).resolve().parent;started=time.monotonic()
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)
    for n in ("test_task_source_reference","test_task_numerical_assessment"))
with (directory/"component-controls.stdout").open("x") as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
names=("source_precision_reference.py","test_source_precision_reference.py",
       "task_source_reference.py","test_task_source_reference.py",
       "task_numerical_assessment.py","test_task_numerical_assessment.py")
files=[ROOT/"scripts/baseline"/n for n in names]+[Path(__file__)]
report=dict(format="poseidon-source-and-numerical-component-controls-v1",
    status="passed" if result.wasSuccessful() else "failed",
    tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
    elapsed_seconds=time.monotonic()-started,api_calls=0,compiler_calls=0,encrypted_execution=False,
    backend_outputs="Injected fixtures; SourceReference positive execution validated separately",
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
report["binding"]=digest(report)
with (directory/"component-controls.json").open("x") as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
