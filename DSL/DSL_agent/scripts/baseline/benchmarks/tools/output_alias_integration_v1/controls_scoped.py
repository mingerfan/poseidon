from pathlib import Path
import sys,json,unittest
BASE=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
mods=["test_unified_output_aliases","test_unified_capacity_wide"]
r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(mods))
record=dict(tests=r.testsRun,modules=mods,failures=[(str(t),s)for t,s in r.failures],
errors=[(str(t),s)for t,s in r.errors],skipped=[(str(t),s)for t,s in r.skipped],
paid_calls=0,compilations=0,encrypted_calls=0)
(HERE/"controls-final-scoped.json").write_text(json.dumps(record,sort_keys=True))
print(json.dumps({k:v if type(v)is int else len(v)for k,v in record.items()}))
raise SystemExit(not(r.wasSuccessful()and not r.skipped))
