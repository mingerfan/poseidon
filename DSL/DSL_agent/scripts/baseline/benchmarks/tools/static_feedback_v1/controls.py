import sys,json,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
modules=("test_static_work_feedback","test_native_diagnostics","test_decorated_functions",
"test_binding_feedback","test_application_diagnostics","test_application_component",
"test_component_contract","test_cli_feedback_policy","test_unified_graph",
"test_application_provenance_transport","test_poseidon_sandbox_policy",
"test_resource_feedback_chain")
base=Path("scripts/baseline")
available=[x for x in modules if (base/(x+".py")).is_file()]
start=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(available)
r=unittest.TextTestRunner(verbosity=1).run(suite)
out=dict(format="poseidon-static-feedback-controls-v1",modules=available,
tests=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),
failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
elapsed_seconds=time.monotonic()-start,provider_calls=0,compiler_calls=0,encrypted_calls=0)
Path(__file__).with_name("controls-report.json").write_text(json.dumps(out,sort_keys=True,indent=2))
print(json.dumps(out),flush=True)
raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)
