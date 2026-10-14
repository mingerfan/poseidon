"""Pure-data and process-supervision acceptance; no model API, source capture or FHE."""
import argparse,io,json,time,unittest
import runner_scope as scope,test_campaign
def main(label="controls"):
 if label not in ("controls","runtime-controls"):raise ValueError("control_record_name")
 out=scope.HERE/(label+".json");log=scope.HERE/(label+".log")
 if out.exists()or log.exists():raise SystemExit("preserve existing controls")
 started=time.monotonic();buffer=io.StringIO()
 result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_campaign))
 log.write_text(buffer.getvalue())
 record=dict(format="poseidon-source-expansions-controls-v1",tests_run=result.testsRun,failed=len(result.failures),
  errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
  status="passed"if result.wasSuccessful()and not result.skipped else"failed",
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,source_captures=0,
  proposal_binding=scope.PROPOSAL_BINDING,tools=scope.tools(),manifest_sha256=scope.sha(scope.MANIFEST))
 record["binding"]=scope.digest(record)
 with out.open("x")as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write("\n")
 print(json.dumps({k:v for k,v in record.items()if k!="tools"},sort_keys=True))
 return 0 if record["status"]=="passed"else 1
if __name__=="__main__":
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--label",choices=("controls","runtime-controls"),default="controls")
 raise SystemExit(main(parser.parse_args().label))
