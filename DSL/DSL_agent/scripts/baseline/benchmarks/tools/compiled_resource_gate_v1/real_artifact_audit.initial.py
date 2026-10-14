import json,hashlib,pathlib
from application_planning.compiled_resources import analyze_bound_artifacts,canonical
root=pathlib.Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-v12-source-live-rebound-v2')
out=pathlib.Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-compiled-resource-gate-artifact-audit')
out.mkdir(exist_ok=False)
rows=[]
for name in ['linear','norm','split_weight']:
 packages=list((root/name).glob('tasks/*/program'))
 assert len(packages)==1
 folder=packages[0]
 request=json.loads((folder/'request.json').read_bytes())
 report=analyze_bound_artifacts(folder/'artifacts',request)
 (out/(name+'.json')).write_bytes(canonical(report))
 rows.append(dict(case=name,status=report['status'],
  lower_bytes=report['facts']['known_retained_payload_lower_bytes'],
  budget_bytes=report['limits']['memory_budget_bytes'],
  readiness=report['execution_readiness'],binding=report['binding'],
  source='historical true Agent compiled artifact; fresh analysis only',
  encrypted_execution_performed_this_audit=False))
 print(rows[-1],flush=True)
(out/'report.json').write_bytes(canonical(dict(status='complete',cases=rows,api_calls=0,compilations=0,encrypted_execution=0)))
