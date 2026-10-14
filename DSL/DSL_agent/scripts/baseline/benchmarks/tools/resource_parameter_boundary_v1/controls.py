"""Run bounded Agent resource-parameter boundary controls; no FHE or API."""
import hashlib,json,sys,time,unittest
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest
from test_backend_parameters_boundary import ParameterBoundaryTests
HERE=Path(__file__).resolve().parent
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ParameterBoundaryTests))
report=dict(format="poseidon-resource-parameter-boundary-controls-v1",
 status="passed" if result.wasSuccessful() else "failed",
 tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
 source_only_copy_proof=ParameterBoundaryTests.graphs,
 production_activation=False,api_calls=0,compiler_calls=0,encrypted_execution=False,
 parameter_contract_changed=False,process_limits_changed=False,
 elapsed_seconds=time.monotonic()-started,
 tool_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
 ROOT/"scripts/baseline/application_planning/backend_parameters.py",
 ROOT/"scripts/baseline/test_backend_parameters_boundary.py",
 HERE/"activate.patch",Path(__file__))})
report["binding"]=digest(report)
with (HERE/"report.json").open("x") as stream:json.dump(report,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
print(json.dumps({k:report[k] for k in ("status","tests","source_only_copy_proof","production_activation","api_calls","compiler_calls","encrypted_execution","binding")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
