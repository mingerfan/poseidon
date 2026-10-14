import json,sys,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
names=["test_static_tensor_transforms","test_operator_spatial_graph","test_execution_input_domain_spatial","test_pytorch_source_package","test_application_source_package_inventory","test_pytorch_spatial_cli"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful())))
raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
