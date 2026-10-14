"""Pinned transport/component regression; generation and backend are test doubles."""
import sys,json,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[3];sys.path.insert(0,str(B))
names=['test_pytorch_spatial_cli', 'test_pytorch_public_joins', 'test_pytorch_model_intake', 'test_application_source_package_inventory', 'test_static_tensor_transforms', 'test_pytorch_source_package', 'test_pytorch_static_broadcast', 'test_pytorch_spatial_source', 'test_pytorch_source_package_intake', 'test_logical_view_normalization', 'test_provenance_json_transport', 'test_application_component', 'test_application_model_source', 'test_model_source_provenance', 'test_model_source_bundle', 'test_program_execution', 'test_candidate_bundle', 'test_staged_bundle', 'test_application_model_source_context', 'test_public_weight_context']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
 skipped=len(result.skipped),passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped))))
raise SystemExit(0 if result.wasSuccessful()else 1)
