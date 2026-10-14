"""Pure checks only. No Torch, compiler, Provider or backend execution."""
import io,json,pathlib,time,unittest,sys,hashlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
import test_boolean_arithmetic
from benchmark_graph import digest
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((HERE/"before.json").read_text())
def unchanged():
 assert all(sha(ROOT/"scripts/baseline"/p)==h for p,h in before["production_python"].items())
 assert all(sha(ROOT/p)==h for p,h in before["previous_source_draft"].items())
 assert all(sha(ROOT/p)==h for p,h in before["previous_v2_python"].items())
 assert sha(pathlib.Path(before["previous_v2_capture_report"]["path"]))==before["previous_v2_capture_report"]["sha256"]
unchanged();started=time.monotonic();stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_boolean_arithmetic))
unchanged();record=dict(format="poseidon-public-boolean-controls-v1",tests_run=result.testsRun,
 passed=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),failed=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-started,
 status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 draft_files={p.name:sha(p)for p in HERE.glob("*.py")},production_activation=False,new_source_captures=0,
 provider_calls=0,compiler_calls=0,encrypted_calls=0,previous_evidence_unchanged=True,network_configuration_changes=0)
record["binding"]=digest(record)
with(HERE/"controls-static-v2.log").open("x")as f:f.write(stream.getvalue())
with(HERE/"controls-static-v2.json").open("x")as f:json.dump(record,f,indent=2,allow_nan=False,sort_keys=True)
print(json.dumps({k:v for k,v in record.items()if k!="draft_files"}),flush=True)
raise SystemExit(0 if record["status"]=="passed"else 1)
