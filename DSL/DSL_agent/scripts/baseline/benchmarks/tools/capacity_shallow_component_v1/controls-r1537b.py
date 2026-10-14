import sys,pathlib,json,unittest,hashlib,time,io
B=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(B))
from application_backend import PoseidonBackend
HERE=pathlib.Path(__file__).resolve().parent
before=PoseidonBackend().identity_manifest()
modules=('test_capacity_broadcast_runs','test_capacity_layout_costs','test_capacity_shallow_packing',
 'test_capacity_static_index_packing','test_capacity_binding_storage','test_unified_mixed_layout','test_poseidon_sandbox_policy','test_poseidon_backend_identity','test_bundle_manifest_storage','test_candidate_bundle')
start=time.time()
suite=unittest.TestLoader().loadTestsFromNames(modules)
with (HERE/'controls-r1537b.log').open('x')as out:
 result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
after=PoseidonBackend().identity_manifest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
report=dict(format='poseidon-shallow-packing-controls-v1',sdk=before['manifest_sha256'],
 source_unchanged=before==after,modules=modules,tests=result.testsRun,failed=len(result.failures),
 errors=len(result.errors),skipped=len(result.skipped),passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 seconds=time.time()-start,api_calls=0,compiler_calls=0,encrypted_calls=0,
 log_sha256=hashlib.sha256((HERE/'controls-r1537b.log').read_bytes()).hexdigest())
report['binding']=digest(report)
(HERE/'controls-r1537b.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report),flush=True)
raise SystemExit(0 if result.wasSuccessful()and before==after and not result.skipped else 1)
