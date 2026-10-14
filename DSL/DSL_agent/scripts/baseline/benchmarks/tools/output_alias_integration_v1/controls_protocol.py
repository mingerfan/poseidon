from pathlib import Path
import sys,unittest,json
B=Path(__file__).resolve().parents[3];T=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
suite=unittest.defaultTestLoader.loadTestsFromName("test_unified_capacity_wide.WideCapacityTests.test_output_protocol_and_full_decode_are_checked")
r=unittest.TextTestRunner(verbosity=1).run(suite)
record=dict(tests=r.testsRun,failures=[(str(t),s)for t,s in r.failures],
errors=[(str(t),s)for t,s in r.errors],skipped=[(str(t),s)for t,s in r.skipped],api_calls=0,encrypted_calls=0)
(T/"controls-protocol-final.json").write_text(json.dumps(record))
raise SystemExit(not(r.wasSuccessful()and not r.skipped))
