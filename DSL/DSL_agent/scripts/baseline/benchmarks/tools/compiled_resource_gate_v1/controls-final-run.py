import unittest,json,pathlib
names=['test_compiled_execution_precheck','application_planning.tests.test_compiled_resources',
 'test_validation_adapter','test_program_execution','test_pytorch_gelu_source_capture',
 'test_pytorch_approximation_capture','test_resource_feedback_chain',
 'test_application_diagnostics','test_application_component','test_cli_feedback_policy']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
summary=dict(tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),actual_http=0,compilations=0,encrypted_execution=0)
pathlib.Path(__file__).with_name('controls-integrity-final-summary.json').write_text(json.dumps(summary,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
