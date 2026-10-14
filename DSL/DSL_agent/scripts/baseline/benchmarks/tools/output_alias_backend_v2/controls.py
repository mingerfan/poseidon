from pathlib import Path
import sys,unittest,json
B=Path(__file__).resolve().parents[3];T=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
mods=["test_unified_output_aliases","test_agent_component","test_program_execution"]
r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(mods))
v=dict(format="poseidon-output-alias-adapter-controls-v2",status="passed"if r.wasSuccessful()and not r.skipped else"failed",
tests=r.testsRun,modules=mods,failures=[(str(t),s)for t,s in r.failures],
errors=[(str(t),s)for t,s in r.errors],skipped=[(str(t),s)for t,s in r.skipped],api_calls=0,compiler_calls=0,encrypted_calls=0)
(T/"controls.json").write_text(json.dumps(v,sort_keys=True))
print(json.dumps({k:v if type(v)is int else len(v)if type(v)is list else v for k,v in v.items()}))
raise SystemExit(v["status"]!="passed")
