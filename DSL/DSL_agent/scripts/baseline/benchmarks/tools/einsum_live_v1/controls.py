"""Run bounded pure-data controls; no source capture, compile, FHE or HTTP."""
import io,json,pathlib,time,unittest
import scope,test_scope
out=scope.HERE/"controls.json";log=scope.HERE/"controls.log"
if out.exists()or log.exists():raise SystemExit("preserve existing controls")
started=time.monotonic();buffer=io.StringIO()
result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_scope))
log.write_text(buffer.getvalue())
record=dict(format="poseidon-einsum-live-scope-controls-v1",tests_run=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
 status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
 proposal_binding=scope.read(scope.PROPOSAL)["binding"],
 tools={n:scope.sha(scope.HERE/n)for n in("scope.py","prepare.py","test_scope.py","controls.py")})
out.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n");print(json.dumps(record,sort_keys=True))
raise SystemExit(0 if record["status"]=="passed"else 1)
