"""Batch controls use no source capture, SDK rebuild, FHE or model API."""
import io,json,time,unittest
import runner_scope as scope,test_scope,test_runner_scope,test_run
out=scope.HERE/"runtime-controls.json";log=scope.HERE/"runtime-controls.log"
if out.exists()or log.exists():raise SystemExit("preserve existing runtime controls")
started=time.monotonic();buffer=io.StringIO();suite=unittest.TestSuite()
for module in(test_scope,test_runner_scope,test_run):
 suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
log.write_text(buffer.getvalue())
record=dict(format="poseidon-einsum-live-runtime-controls-v1",tests_run=result.testsRun,failed=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
 status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
 proposal_binding=scope.PROPOSAL_BINDING,tools=scope.tools(),manifest_sha256=scope.sha(scope.MANIFEST))
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n");print(json.dumps(record,sort_keys=True))
raise SystemExit(0 if record["status"]=="passed"else 1)
