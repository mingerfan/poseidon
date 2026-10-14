import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
if __name__=="__main__":
    directory=Path(__file__).resolve().parent;started=time.monotonic()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in (
        "test_task_dataset_assessment.DatasetTests","test_task_numerical_assessment.AssessmentTests",
        "test_task_native_assessment.NativeAssessmentTests"))
    with (directory/"controls.final2.stdout").open("x") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    files=[ROOT/"scripts/baseline"/n for n in (
        "task_dataset_assessment.py","test_task_dataset_assessment.py",
        "task_numerical_assessment.py","test_task_numerical_assessment.py","test_task_native_assessment.py")]+[Path(__file__)]
    report=dict(format="poseidon-dataset-component-controls-v1",
        status="passed" if result.wasSuccessful() else "failed",
        tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
        elapsed_seconds=time.monotonic()-started,new_encrypted_calls=0,source_forward_calls=0,api_calls=0,compiler_calls=0,
        fixture_kind="Injected Application and source outputs; no real Agent, FHE or source execution",
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    report["binding"]=digest(report)
    with (directory/"controls.final2.json").open("x") as f:json.dump(report,f,allow_nan=False,indent=2)
    print(json.dumps(report));raise SystemExit(0 if result.wasSuccessful() else 1)
