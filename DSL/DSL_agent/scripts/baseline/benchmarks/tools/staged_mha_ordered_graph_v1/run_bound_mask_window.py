"""One existing Nix window: new controls, then affected free backend cases."""
import hashlib,json,subprocess,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"contract_tests"));sys.path.insert(0,str(HERE))
sys.path.append(str(HERE.parents[2]))
from benchmark_graph import canonical,digest,require
from hecate_python_env import WORK
def main():
 out=WORK/"results/agent-mha-bound-mask-controls-v1";require(not out.exists(),"Preserve controls evidence");out.mkdir()
 files=[HERE/"trusted_bound_mask_compaction.py",HERE/"contract_tests/test_bound_mask_compaction.py",
 HERE/"contract_tests/test_schedule_compaction.py",HERE/"backend_bound_mask_acceptance.py",Path(__file__)]
 frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 suite=unittest.TestSuite()
 for name in ("test_bound_mask_compaction","test_schedule_compaction.ScheduleCompaction"):
  suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
 began=time.monotonic()
 with (out/"unittest.log").open("w") as stream:
  result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in frozen.items()),"Control tools changed")
 report=dict(format="poseidon-mha-bound-mask-controls-v1",tests=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 tool_sources=frozen,source_frozen=True,elapsed_seconds=time.monotonic()-began,
 api_calls=0,compiler_calls=0,encrypted_execution=False,production_applied=False)
 report["binding"]=digest(report);(out/"report.json").write_bytes(canonical(report))
 print("NEW_CONTROLS",json.dumps(report),flush=True)
 if not result.wasSuccessful() or result.skipped:return 1
 return subprocess.run([sys.executable,"-B",str(HERE/"backend_bound_mask_acceptance.py"),
 str(WORK/"results/agent-mha-ordered-backend-bound-mask-v1"),
 str(WORK/"results/agent-mha-ordered-capture-v2")],timeout=1240).returncode
if __name__=="__main__":raise SystemExit(main())
