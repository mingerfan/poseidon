"""Bounded numerical-assessment control run; injected execution, no FHE/API."""
import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from test_task_numerical_assessment import AssessmentTests
HERE=Path(__file__).resolve().parent
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AssessmentTests))
report=dict(format="poseidon-task-assessment-controls-v1",
 status="passed" if result.wasSuccessful() else "failed",
 tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
 api_calls=0,compiler_calls=0,encrypted_execution=False,
 execution_fixtures="Injected backend outputs; not encryption or numerical backend acceptance",
 elapsed_seconds=time.monotonic()-started,
 files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
 ROOT/"scripts/baseline/task_numerical_assessment.py",
 ROOT/"scripts/baseline/test_task_numerical_assessment.py",Path(__file__))})
report["binding"]=digest(report)
with (HERE/"report.json").open("x") as stream:json.dump(report,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
print(json.dumps({k:report[k] for k in ("status","tests","elapsed_seconds","api_calls","compiler_calls","encrypted_execution","binding")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
