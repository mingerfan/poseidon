"""Unpaid continuation control acceptance. Never opens a Provider transport."""
import io,json,sys,time,unittest
from pathlib import Path
import continuation_scope as scope,test_continuation
def main():
 out=scope.HERE/"controls-v1.json";log=scope.HERE/"controls-v1.log"
 if out.exists()or log.exists():raise ValueError("preserve_existing_controls")
 before={p:scope.sha(p)for p in scope.retired_snapshot()["evidence"]}
 started=time.monotonic();stream=io.StringIO()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_continuation))
 log.write_text(stream.getvalue())
 scope.require(before=={p:scope.sha(p)for p in before},"retired_evidence_changed_during_controls")
 record=dict(format="poseidon-original-live-continuation-controls-v2",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
  tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
  failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
  tool_hashes=scope.tools(),manifest_sha256=scope.sha(scope.MANIFEST),retired_evidence_preserved=True,
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,network_configuration_changes=0)
 record["binding"]=scope.digest(record)
 with out.open("x")as f:json.dump(record,f,sort_keys=True,indent=2)
 print(json.dumps({k:v for k,v in record.items()if k!="tool_hashes"}),flush=True)
 return 0 if record["status"]=="passed"else 1
if __name__=="__main__":raise SystemExit(main())
