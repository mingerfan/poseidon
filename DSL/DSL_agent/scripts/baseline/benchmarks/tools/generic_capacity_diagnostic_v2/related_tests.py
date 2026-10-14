import sys,unittest,json
from pathlib import Path
B=Path("scripts/baseline").resolve();D=B/"benchmarks/tools/generic_capacity_diagnostic_v2"
sys.path[:0]=[str(D/"proposed"),str(B)]
names=("test_native_public_loops","test_decorated_functions","test_function_construction",
       "test_native_diagnostics","test_application_diagnostics",
       "test_native_reserved_binding_feedback","test_static_work_feedback")
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
 skipped=len(result.skipped),successful=result.wasSuccessful())))
raise SystemExit(0 if result.wasSuccessful()else 1)
