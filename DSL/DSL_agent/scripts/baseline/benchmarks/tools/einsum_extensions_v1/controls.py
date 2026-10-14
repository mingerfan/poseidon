import argparse,hashlib,io,json,pathlib,time,unittest
import test_extension
HERE=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument("--record-name",default="controls-round1");args=parser.parse_args()
if args.record_name not in("controls-round1","controls-round2","controls-round3"):raise SystemExit("bounded record name")
path=HERE/(args.record_name+".json");log=path.with_suffix(".log")
if path.exists()or log.exists():raise SystemExit("preserve controls")
start=time.monotonic();buffer=io.StringIO()
result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_extension))
log.write_text(buffer.getvalue())
record=dict(format="poseidon-einsum-extension-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 tests_run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-start,
 sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in[HERE/"planner.py",HERE/"test_extension.py",pathlib.Path(__file__)]},
 provider_calls=0,compiler_calls=0,encrypted_calls=0,production_source_policy=False)
path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
print(json.dumps(record,sort_keys=True))
if not result.wasSuccessful():print(buffer.getvalue()[-15000:])
raise SystemExit(0 if record["status"]=="passed"else 1)
