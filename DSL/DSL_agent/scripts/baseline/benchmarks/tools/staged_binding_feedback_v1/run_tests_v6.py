"""Run staged Agent checks against existing dependencies, without Nix re-entry."""
import hashlib
import importlib
import io
import json
import platform
import sys
import time
import unittest
from pathlib import Path

STAGE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
LIVE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-approved-pending-continuation-r1099")
sys.path.insert(0,str(ROOT/"scripts/baseline"))
sys.path.insert(0,str(STAGE))
sys.path.insert(0,str(STAGE/"targets"))
sys.path.insert(0,"/home/lhohy/poseidon-work/platforms/aarch64-linux/venvs/hecate-2.0.1-cpu/lib/python3.10/site-packages")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze():
    plan=json.loads((LIVE/"plan.json").read_text())
    counts={}
    for name,base in (("source_hashes",ROOT),("tool_hashes",Path("/")),("retained_hashes",Path("/"))):
        expected=plan[name]
        for relative,h in expected.items():
            if sha(base/relative)!=h:
                raise ValueError("live freeze mismatch: "+name)
        counts[name]=len(expected)
    return counts

if __name__=="__main__":
    target=STAGE/"acceptance-r1124.json"
    if target.exists():
        raise SystemExit("fresh test acceptance required")
    before=freeze()
    import numpy
    if numpy.__version__!="1.25.2":
        raise ValueError("locked existing NumPy required")
    suite=unittest.TestSuite()
    for name in ("test_staged","test_application_diagnostics","test_decorated_functions",
                 "test_native_array_core","test_unified_poseidon_guidance","test_application_component"):
        module=importlib.import_module(name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    started=time.monotonic()
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    output=dict(format="poseidon-staged-binding-feedback-tests-v1",
                tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                seconds=time.monotonic()-started,
                failures=[dict(test=str(t),traceback=message) for t,message in result.failures+result.errors],
                runtime=dict(python=platform.python_version(),numpy=numpy.__version__,
                             numpy_source=str(Path(numpy.__file__).resolve()),system=platform.system(),machine=platform.machine()),
                verification_scope="AST, context, task controls and mocked component compatibility only; no new provider, compilation, encrypted execution or decryption",
                production_applied=False,provider_calls=0,compilations=0,encrypted_executions=0,
                freeze_before=before,freeze_after=freeze(),
                source_hashes={str(p.relative_to(STAGE)):sha(p) for p in list((STAGE/"targets").glob("*.py"))+[STAGE/"test_staged.py",Path(__file__)]},
                legacy_test_hashes={name:sha(ROOT/"scripts/baseline"/(name+".py")) for name in (
                    "test_application_diagnostics","test_decorated_functions","test_native_array_core","test_unified_poseidon_guidance","test_application_component")})
    target.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    (STAGE/"tests-r1124.log").write_text(log.getvalue())
    print(json.dumps({k:output[k] for k in ("tests","passed","failed","errors","skipped","seconds","runtime","freeze_after")}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
