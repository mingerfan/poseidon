import hashlib,json,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from application_identity import current_identity_manifest
from test_staged_application_reuse import StagedContextControls
class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.successes=[]
    def addSuccess(self,test):
        self.successes.append(test.id());super().addSuccess(test)
start=time.monotonic();identity=current_identity_manifest()
suite=unittest.defaultTestLoader.loadTestsFromTestCase(StagedContextControls)
with(HERE/"application-context-controls.log").open("x")as f:
    r=unittest.TextTestRunner(stream=f,verbosity=2,resultclass=Result).run(suite)
d=dict(format="poseidon-generic-context-controls-v1",status="passed"if r.wasSuccessful()else"failed",
 planned=suite.countTestCases(),executed=r.testsRun,passed=len(r.successes),
 failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),
 successful_tests=r.successes,seconds=time.monotonic()-start,sdk_identity=identity["manifest_sha256"],
 source_sha256={n:hashlib.sha256((ROOT/"scripts/baseline"/n).read_bytes()).hexdigest()
                for n in["test_staged_application_reuse.py","test_application_identity.py"]},
 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 log_sha256=hashlib.sha256((HERE/"application-context-controls.log").read_bytes()).hexdigest(),
 compiler_calls=0,encrypted_calls=0,provider_calls=0,
 loader="Existing pinned HECATE_PYTHON_LIBRARY_PATH applied to this process only",
 scope="Original10 setup-blocked controls only; no source capture or real backend qualification")
d["not_run"]=d["planned"]-d["executed"]
d["binding"]=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()
with(HERE/"application-context-controls.json").open("x")as f:json.dump(d,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in d.items()if k!="successful_tests"},sort_keys=True))
sys.exit(0 if r.wasSuccessful()else 1)
