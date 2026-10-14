import argparse,hashlib,io,json,pathlib,re,time,unittest
import test_manifest
HERE=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument("--record-name",default="controls");args=parser.parse_args()
if re.fullmatch(r"[a-z0-9-]+",args.record_name)is None:raise SystemExit("record name")
OUT=HERE/(args.record_name+".json");LOG=HERE/(args.record_name+".log")
if OUT.exists()or LOG.exists():raise SystemExit("preserve existing controls")
started=time.monotonic();buffer=io.StringIO()
result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_manifest))
LOG.write_text(buffer.getvalue())
record=dict(format="poseidon-source-manifest-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 tests_run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,source_executed=False,compiler_calls=0,encrypted_calls=0,provider_calls=0,
 source_sha256=hashlib.sha256(test_manifest.SCRIPT.read_bytes()).hexdigest(),
 test_sha256=hashlib.sha256((HERE/"test_manifest.py").read_bytes()).hexdigest())
OUT.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n");print(json.dumps(record,sort_keys=True))
if not result.wasSuccessful():print(buffer.getvalue()[-6500:])
raise SystemExit(0 if record["status"]=="passed"else 1)
